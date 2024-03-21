import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

import torch
import json
from pathlib import Path
import os
import shutil
import argparse
import os
import torch
import torch.distributed as dist
from tqdm import tqdm

# import ddp_utils
import numpy as np
import re
import ddp_utils as du

device = "cuda"
seed = 23   
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_grad_enabled(False)

def add_instruct_message(verb, object, response_content):
    # 새로운 instruct_message 생성
    messages = [
        {"role": "user", "content": "In the image, a person is standing in a workshop with a light-colored wooden surfboard. The individual is wearing a gray shirt and a white apron, as well as blue ear protection on their head. They are meticulously working on the surfboard, likely preparing it for use or maintenance. The workshop appears to be equipped with various tools and materials, indicating that this might be a dedicated space for crafting or repairing objects like the surfboard. A hammer can be seen among these tools, suggesting that the person might be engaged in some form of carpentry or repair work. The overall scene suggests a moment of focus and dedication to craftsmanship, as the person carefully works on the surfboard within the well-stocked workshop. The text provides a detailed description of the image. Your task is to read it and determine if there is a human-object interaction based on the questions I'm asking. From the text, can you infer that a person is opening a microwave? Please begin your response with 'yes' or 'no', followed by your explanation."},
        {"role": "assistant", "content": "No, The situation described in the article describes a person working on a wooden surfboard in a workshop. There is no mention of opening the microwave at all in the article, so the answer is no."},
    ]
    instruct_message = {
        "role": "user",  
        "content": response_content 
    }

    # messages 리스트에 instruct_message 추가
    messages.append(instruct_message)

    return messages

def save_responses_to_file(responses_dict, prompt_each_dir):

    # responses_dict를 JSON 문자열로 변환
    responses_str = json.dumps(responses_dict, indent=4)
    
    # 저장할 파일 경로 설정
    file_path = os.path.join(prompt_each_dir, "feedbackfiltering.txt")
    
    # 파일에 응답 문자열 저장
    with open(file_path, "w") as file:
        file.write(responses_str)

@du.ddp_on_and_off
def main():
    # du.ddp_init()

    #LLM
    llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(f"cuda:{du.get_rank()}").eval()
    llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    #VLM
    prompt_path = Path("/home/uvll/jjunsss/dataset/total_new_crawling")
    vlm_model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).to(f"cuda:{du.get_rank()}").eval()
    vlm_tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)
    crawl_paths = list(prompt_path.iterdir())

    sliced_crawl = du.ddp_data_split(len(crawl_paths), crawl_paths)
    if du.get_world_size()>2: dist.barrier()
    for idx, prompt_each in enumerate(sliced_crawl):
        print(f"processing : {idx+1} / {len(list(prompt_path.iterdir()))}")
        responses = {'yes': [], 'no': []}  # 응답을 저장할 dictionary 초기화

        image_paths = list(prompt_each.glob("*/*.jpg")) + list(prompt_each.glob("*/*.jpeg")) + list(prompt_each.glob("*/*.png"))
        image_paths = [str(path) for path in image_paths]

        query = str(prompt_each.name)

        wording = query[query.find("person") + 7:]
        a_index = wording.find(" a ")
        an_index = wording.find(" an ")

        if a_index != -1 and an_index != -1:
            cut_index = min(a_index, an_index)
        elif a_index != -1:
            cut_index = a_index
        elif an_index != -1:
            cut_index = an_index
        else:
            cut_index = len(wording)

        verb = wording[:cut_index].strip()
        object = wording[cut_index+2:].strip()
        try:
            with torch.no_grad():
                for img_path in tqdm(image_paths, disable=not du.is_main_process()):
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        if img.width > 1800 or img.height > 1800:
                            # If the image is too large, append the response and continue to the next image
                            responses['no'].append({'path': img_path, 'response': 'too large size image'})
                            continue  # Skip the current iteration and move to the next image
                    
                    vlm_text = f"<ImageHere>Please provide a detailed description of this image."
                    vlm_image = img_path
                    with torch.cuda.amp.autocast():
                        # vlm_model.to(device)
                        response, _ = vlm_model.chat(vlm_tokenizer, query=vlm_text, image=vlm_image, history=[], do_sample=True)
                    # print(response)
                    
                    if verb != "and":
                        response = response + f" The text provides a detailed description of the image. Your task is to read it and determine if there is a human-object interaction based on the questions I'm asking. From the text, can you infer that a person is {verb} a {object}? Please begin your response with 'yes' or 'no', followed by your explanation."
                    else :
                        response = response + f" The text provides a detailed description of the image. Your task is to read it and determine if there is a human-object interaction based on the questions I'm asking. From the text, can you infer that a person is not interacting with a {object}, even though both the person and {object} are present in the image? Please begin your response with 'yes' or 'no', followed by your explanation."

                    instruct_question = add_instruct_message(verb, object, response)
                    encodeds = llm_tokenizer.apply_chat_template(instruct_question, return_tensors="pt")

                    model_inputs = encodeds.to(device)
                    llm_model.to(device)


                    generated_ids = llm_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
                    llm_response = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                    llm_response = llm_response.split("[/INST]")[-1].strip()
                    # vlm_model.to("cpu")
                    # llm_model.to("cpu")

                    yes_no_pattern = r'^(yes|no)\b'
                    while True:
                        if re.match(yes_no_pattern, llm_response.lower()):
                            break 
                        generated_ids = llm_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
                        llm_response = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        llm_response = llm_response.split("[/INST]")[-1].strip()

                    if  llm_response.lower().startswith("yes"):
                        responses['yes'].append({'path': img_path, 'response': llm_response})
                    elif  llm_response.lower().startswith("no"):
                        responses['no'].append({'path': img_path, 'response': llm_response})

                    torch.cuda.empty_cache()
                
        except Exception as e:
            responses['no'].append({'path': img_path, 'response': f'error generated: {e}' })
            continue

        if du.get_world_size()>2: dist.barrier()
        save_responses_to_file(responses, prompt_each)
        if du.get_world_size()>2: dist.barrier()
        
if __name__ == "__main__":
    main()
