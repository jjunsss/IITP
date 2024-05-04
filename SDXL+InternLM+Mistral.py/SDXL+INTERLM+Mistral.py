import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.utils import load_image

# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import ddp_utils as du
import tqdm
import torch.distributed as dist
import random
# from inf import *
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os
import re
import json

import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer 
from auto_gptq.modeling._base import BaseGPTQForCausalLM

seed = 23   
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

NEG_PROMPT = "[deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry, cgi, doll, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), 3d, illustration, cartoon, (doll:0.9), octane, (worst quality, low quality:1.4), EasyNegative, bad-hands-5, (bad and mutated hands:1.3), (bad hands), missing fingers, multiple limbs, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, (deformed fingers:1.2), (long fingers:1.2), comic, zombie, sketch, muscles, sinewy, bad anatomy, censored, signature, monochrome, text, watermark, sketch, duplicate, bad-artist-anime, loli, mature"

def load_categories_from_json(file_path):
    """
    Load category data from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def add_instruct_message(verb, object, response_content, paraph=False):
    # 새로운 instruct_message 생성
    paraphrasing_prompt = [
        {"role": "user", "content": "'original_prompt': A photo of a person swinging a remote, the person is wearing a light-colored shirt and slacks. The setting appears to be an indoor space with furniture and various personal items scattered around. 'Requirements': Please paraphrase the above description. The goal is to express the same content in a different way."},
        {"role": "assistant", "content": "In the image, an individual is seen wielding a remote control, clad in a light-toned top and matching trousers. The interior scene is adorned with household furnishings and a variety of personal effects."},
        {"role": "user", "content": "'original_prompt': A photo of a person swinging a remote, wearing a purple shirt and gray pants. The television in front of him displays a scene from an animated game with an intense battle happening on the screen. 'Requirements': Please paraphrase the above description. The goal is to express the same content in a different way."},
        {"role": "assistant", "content": "The picture captures a moment where a person is actively using a remote, dressed in a violet shirt paired with charcoal trousers. In the backdrop, a television screen vividly showcases an animated conflict, drawing attention to the intense gaming action."},
    ]
    messages = [
                {"role": "user", "content": "'Descriptions': In the image, a person is standing in a workshop with a light-colored wooden surfboard. The individual is wearing a gray shirt and a white apron, as well as blue ear protection on their head. They are meticulously working on the surfboard, likely preparing it for use or maintenance. The workshop appears to be equipped with various tools and materials, indicating that this might be a dedicated space for crafting or repairing objects like the surfboard. A hammer can be seen among these tools, suggesting that the person might be engaged in some form of carpentry or repair work. The overall scene suggests a moment of focus and dedication to craftsmanship, as the person carefully works on the surfboard within the well-stocked workshop. 'Question': The text provides a detailed description of the image. Your task is to read it and determine if there is a human-object interaction based on the questions I'm asking. Can you definitively determine from the text whether a person is performing the action 'opening' on the object 'microwave'? If either the person or the object is not present, or if you cannot clearly determine the opening action, respond with 'no'. Only respond with 'yes' if you can definitively determine the action, person, and microwave (including things similar to the microwave, e.g., sunglasses can be included in glass) based on the text. Please begin your response with 'yes' or 'no', followed by your explanation."},
                {"role": "assistant", "content": "No, The situation described in the article describes a person working on a wooden surfboard in a workshop. There is no mention of opening the microwave at all in the article, so the answer is no."},
                {"role": "user", "content": "'Descriptions': The image features a man working with ski bindings in a shop. He is in the process of adjusting the bindings on a pair of skis using a tool. There are multiple pairs of skis placed around the room, some of which are closer to the foreground, while others are positioned further away.\n\nThe man is standing at a work table, which spans a significant portion of the scene. The skis are laid flat on the table, and the workspace is well-organized for the task at hand. The shop appears to be dedicated to ski maintenance and repair, providing an ideal environment for the skier to get his bindings adjusted. 'Question': The text provides a detailed description of the image. Your task is to read it and determine if there is a human-object interaction based on the questions I'm asking. Can you definitively determine from the text whether a person is performing the action 'adjusting' on the object 'skis'? If either the person or the object is not present, or if you cannot clearly determine the adjusting action, respond with 'no'. Only respond with 'yes' if you can definitively determine the action, person, and skis (including things similar to the skis, e.g., sunglasses can be included in glass) based on the text. Please begin your response with 'yes' or 'no', followed by your explanation."},
                {"role": "assistant", "content": "Yes, the description provided clearly indicates that the man is in the process of adjusting the bindings on a pair of skis using a tool. This activity is central to the scene depicted, with the man working at a table in a shop dedicated to ski maintenance and repair. The detailed setup of the shop and the multiple pairs of skis placed around further emphasize the focus on ski adjustment and maintenance."},
    ]

    instruct_message = {
        "role": "user",  # 또는 "assistant", 메시지의 발신자에 따라
        "content": response_content  # 여기에 추가할 메시지 내용을 입력
    }

    # messages 리스트에 instruct_message 추가
    if paraph == True:
        paraphrasing_prompt.append(instruct_message)
        return paraphrasing_prompt
    else :
        messages.append(instruct_message)
        return messages

def paraphrase_prompt(llm_model, llm_tokenizer, origin_prompt):
    device = "cuda"
    
    llm_prompt = "'original_prompt':" + origin_prompt + "'Requirements': Please paraphrase the above description. The goal is to express the same content in a different way. Please tell your response within 66 words"
    
    paraph_pormpt = add_instruct_message(None, None, llm_prompt)
    encodeds = llm_tokenizer.apply_chat_template(paraph_pormpt, return_tensors="pt")

    while True:
        model_inputs = encodeds.to(device)
        llm_model.to(device)
        
        generated_ids = llm_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        llm_response = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        paraphrased_prompt = llm_response.split("[/INST]")[-1].strip()
        if word_count(paraphrased_prompt) <= 65 and word_count(paraphrased_prompt) >= 10:
            return paraphrased_prompt


auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
torch.set_grad_enabled(False)

class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]

################################################################################################

def generate_random_attributes(template_database):
    """
    Selects random attributes from each category in the template database to
    generate a unique prompt for each iteration.
    """
    random_attributes = {key: random.choice(value) for key, value in template_database.items()}
    return random_attributes

def load_reference_images(file_path):
    """
    Load reference images from a file.
    """
    reference_images = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            key = parts[0]
            images = parts[1].strip("[]").replace("'", "").split(', ')
            reference_images[key] = images
    return reference_images


# Example usage
reference_images = load_reference_images('hico_20160224_det/hoidataset/medium-rare-image-list_refined.txt')

def select_reference_image(category):
    """
    Select a random reference image for a given category.
    """
    if category in reference_images and reference_images[category]:
        return random.choice(reference_images[category])
    
    raise Exception(f"Unknown reference image category: {category}")

def call_the_image(category):
    """
    Simulate calling an image based on the category. In a real scenario, this could be
    replaced with actual logic to generate or retrieve an image.
    """
    ref_image = select_reference_image(category)
    # Here, we just return the name of the reference image. In practice, this should
    # retrieve the actual image, e.g., loading it into memory.
    return ref_image

def word_count(s):
    return len(s.split())


def update_vlm_query(category, verb, obj):

    reference_image = call_the_image(category)

    #! prompt changed ver 5 (too mnay people and many objects)
    # new_query = f"<ImageHere>Please provide a detailed description of this image."
    new_query = f"<ImageHere>Please provide a detailed description of the image, focusing on the main person who is {verb} a {obj}. Include specific features and attributes of the person and the {obj}, as well as the overall environment and context of the interaction. Remain objective and concrete. Please focus on describing one person and one {obj}. The total word count should not exceed 66 words. Follow this template for your answer: 'A photo of a person {verb} a {obj}, [insert your description].' Please adhere to the provided template in your answer."
    
    return new_query, reference_image

def filtering_details(verb, obj, img_name, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer, g_model, g_processor):
    device = "cuda"
    vlm_image = img_name
    vlm_query = f"<ImageHere>Please provide a detailed description of this image. Specifically, focus on the person, object, interaction each other, and all background in the image"
    
    #* for checking using grounding dino
    check_item = dino_filtering(g_model, g_processor, img_name, verb, obj)
    if check_item == False :
        return False
    
    #* for describing whole image using VLM(e.g. internLM)
    response = model_eval(vlm_model, vlm_tokenizer, vlm_image, vlm_query)
    
    #* for checking using LLM(e.g. mistral)
    if verb != "and":
        llm_prompt = "'Descriptions':" + response + "'Question':" + f" This text provides a detailed description of the image. Your task is to determine if there is a human-object interaction based on the questions I'm asking. Can you definitively determine from the text whether a person is performing the action '{verb}' on the object '{obj}'? If either the person or the {obj} is not present, or if you cannot clearly determine the {verb} action, respond with 'no'. Only respond with 'yes' if you can definitively determine the action, person, and {obj} (including things similar to the {obj}, e.g., sunglasses can be included in glass class) based on the text. Please begin your response with 'yes' or 'no', followed by your explanation."

    else :
        llm_prompt = "'Descriptions':" + response + "'Question':" + f" Can you definitively determine from the text whether a person is not interacting with a {obj}, even though both the person and {obj} are present in the image? If either the person or the {obj} is not present, please respond with 'no'. Only respond with 'yes' if you can definitively determine the {obj}, person, but do not {verb} (i.e. no interaction). Please begin your response with 'yes' or 'no', followed by your explanation."

    instruct_question = add_instruct_message(verb, object, llm_prompt)
    encodeds = llm_tokenizer.apply_chat_template(instruct_question, return_tensors="pt")

    model_inputs = encodeds.to(device)
    llm_model.to(device)
    
    generated_ids = llm_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    llm_response = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    llm_response = llm_response.split("[/INST]")[-1].strip()

    yes_no_pattern = r'^(yes|no)\b'
    while True:
        if re.match(yes_no_pattern, llm_response.lower()):
            break 
        generated_ids = llm_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        llm_response = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        llm_response = llm_response.split("[/INST]")[-1].strip()

    if llm_response.lower().startswith("yes"):
        return True
    else : 
        return False

def construct_input_prompt(user_prompt):
    SYSTEM_MESSAGE = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    IMAGE_TOKEN = "Human: <image>\n" #<image> denotes an image placehold.
    USER_PROMPT = f"Human: {user_prompt}\n"

    return SYSTEM_MESSAGE + IMAGE_TOKEN + USER_PROMPT + "AI: "

def model_eval(model, tokenizer, image_path, prompt):
    text = prompt
    image = image_path
    with torch.cuda.amp.autocast(): 
        response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=True) 
        
    return response

def model_loader():
    vlm_model = InternLMXComposer2QForCausalLM.from_quantized('internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True, device=f"cuda:{du.get_rank()}").eval()
    vlm_tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True)
    return vlm_model, vlm_tokenizer

def SDXL_loader():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"# Use the correct ckpt for your step setting!

    torch.cuda.empty_cache()
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(f"cuda:{du.get_rank()}", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=f"cuda:{du.get_rank()}"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(f"cuda:{du.get_rank()}")
    # pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    # pipe.set_ip_adapter_scale(0.6)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return unet, pipe

def generate_response_for_image(model, tokenizer, ref_img, vlm_text):
    while True:
        response = model_eval(model, tokenizer, ref_img, vlm_text)
        if word_count(response) <= 65 and response.startswith(f"A photo of a person") and word_count(response) >= 10:
            break
    return response

def count_files_in_directory(directory_path):
    """
        Counts the number of files in the given directory path, excluding subdirectories.
    """
    return sum(1 for item in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, item)))

def find_max_gen_idx(directory_path):
    max_idx = 0
    pattern = re.compile(r'(\d+)\.png$')

    for filename in os.listdir(directory_path):
        match = pattern.search(filename)
        if match:
            current_idx = int(match.group(1))
            max_idx = max(max_idx, current_idx)

    return max_idx

def calling_grounding_dino():
    model_id = "IDEA-Research/grounding-dino-base"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    return model, processor

def dino_filtering(model, processor, gen_image, verb, target_obj):
    #* initialization
    process_successful = True  # 이미지 처리 성공 여부 플래그
    coco_objects = [target_obj]
            
    # 각 객체명 뒤에 '.'을 추가하고, TEXT_PROMPT에 이어 붙입니다.
    TEXT_PROMPT = "person ." #* This is basic prompt in HOI task
    for obj in coco_objects:
        if obj != "person": #* because if target object is person, then person objects are duplicated. This is not allowed.
            TEXT_PROMPT += obj + " ."
    print(TEXT_PROMPT)

    image = Image.open(gen_image)
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    logits, phrases, boxes = results[0]["scores"], results[0]["labels"], results[0]["boxes"]
    
    if "person" not in phrases or target_obj not in phrases:
        return False
    
    # target is person scenario.
    if target_obj == "person":
        person_counts = phrases.count("person")
        if person_counts < 2 : 
            return False
        
    #* all others should got the person and target objects
    return True
    # boxes = boxes * 1024

def main():
    du.ddp_init()

    #! for category(loading number of images)
    category_data = load_categories_from_json('hico_20160224_det/hoidataset/trainval_hico_insufficient.json')  # Specify the path to your JSON file here
    non_zero_categories = {name: item for name, item in category_data.items() if item != 0}
    name_of_medium_under = [data for data in non_zero_categories.keys()]
    value_of_medium_under = [data for data in non_zero_categories.values()]
    
    #! for VLM
    model, tokenizer = model_loader()
    #! for SDXL
    unet, pipe = SDXL_loader()
    #! for LLM
    llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(f"cuda:{du.get_rank()}").eval()
    llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #! for grounding dino
    g_model, g_processor = calling_grounding_dino()

    rare_list = name_of_medium_under
    save_dir = "./8step_prompt-filtering-newnew/"
    if du.is_main_process():
        for rare in rare_list : 
            os.makedirs(save_dir + rare, exist_ok=True)

    negative_prompt = NEG_PROMPT
    sliced_rare_name = du.ddp_data_split(len(rare_list), rare_list)
    sliced_rare_value = du.ddp_data_split(len(rare_list), value_of_medium_under)
    if du.get_world_size()>2: dist.barrier()
    for idx, (rare_name, rare_value)  in tqdm(enumerate(zip(sliced_rare_name, sliced_rare_value)), disable= not du.is_main_process()) : 
        verb, obj = rare_name.split("_")
        print(f"{idx} / {len(sliced_rare_name)} tasks")
        
        rare_dir = os.path.join(save_dir, rare_name)
        
        #* Check if this category already has 50 or more generated images (considering responses.txt)
        already_images = count_files_in_directory(rare_dir)
        end_count = rare_value + 1
        pass_count = rare_value * 2 #* for passing the rareset
        revise_count = int(rare_value / 2) #* for revising all times
        paraphrising_count = revise_count + 10 #* for paraphrising
        
        if already_images >= end_count:
            print(f"Skipping {rare_name} over {pass_count}.")
            continue
        
        elif already_images >= 1:
            generated_images = already_images - 1
            gen_idx = find_max_gen_idx(rare_dir) + 1
            
            #! cut the generation process, due to too many trash image generated
            if gen_idx >= pass_count:
                responses_filename = f'{save_dir}{rare_name}/responses.txt'
                with open(responses_filename, 'a') as responses_file: 
                    responses_file.write(f'Iteration {gen_idx}: pass this verb-object due to time computation.\n')
                continue
        else :
            generated_images = 0
            gen_idx = 0
        deleted_images = 0
            
        #* generation process  
        while generated_images < rare_value:
            #! cut the generation process, due to too many trash image generated
            if gen_idx >= pass_count:
                responses_filename = f'{save_dir}{rare_name}/responses.txt'
                with open(responses_filename, 'a') as responses_file: 
                    responses_file.write(f'Iteration {gen_idx}: pass this verb-object due to time computation.\n')
                break
            
            # if generated_images % 5 == 0 or deleted_images % 5 == 0:
            if gen_idx % 3 == 0 or deleted_images > revise_count:
                vlm_text, ref_img = update_vlm_query(rare_name, verb, obj)
                ref_base_idr = "../HOI/hico_20160224_det/images/train2015/"
                ref_img = ref_base_idr + ref_img
                # load_ref_img_for_ip = load_image(ref_img)
                
                response = generate_response_for_image(model, tokenizer, ref_img, vlm_text)
                responses_filename = f'{save_dir}{rare_name}/responses.txt'
                print(f"response monitoring here : {response}")

                #* save the generate prompt
                with open(responses_filename, 'a') as responses_file: 
                    responses_file.write(f'Iteration {gen_idx}: {response}\n')
            
                if deleted_images > paraphrising_count:
                    paraphrased_prompt = paraphrase_prompt(llm_model, llm_tokenizer, response)
                    response = paraphrased_prompt
                    
                    #* save the generate prompt
                    with open(responses_filename, 'a') as responses_file: 
                        responses_file.write(f'Changed Iter. {gen_idx}: {response}\n')

            sdxl_prompt = response
            img_name = save_dir + rare_name + '/{}.png'.format(str(gen_idx).zfill(3))
            pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
            pipe(sdxl_prompt, num_inference_steps=8, guidance_scale=0, negative_prompt=negative_prompt, negative_prompt_2 = negative_prompt).images[0].save(img_name)
            
            #! filtering through VLM + LLM for long description; if over the 700 del images, then we just generate image withoud filtering.
            filter_answer = filtering_details(verb, obj, img_name, llm_model, llm_tokenizer, model, tokenizer, g_model, g_processor)
            # if deleted_images < 200:
            #     filter_answer = filtering_VLM_with_LLM(verb, obj, img_name, llm_model, llm_tokenizer, model, tokenizer, deleted_images)
            # elif deleted_images > 300 :
            #     filter_answer = True
            
            
            gen_idx += 1
            if filter_answer == False:
                os.remove(img_name)
                deleted_images += 1
            else :
                generated_images += 1
                print(f"{rare_name} dataste generated images : {generated_images}")
        

    if du.get_world_size()>2: dist.barrier()
    
if __name__ == "__main__":
    main()



