import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
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

import torch, auto_gptq
from transformers import AutoModel, AutoTokenizer 
from auto_gptq.modeling._base import BaseGPTQForCausalLM

seed = 23   
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_grad_enabled(False)

NEG_PROMPT = "[deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry, cgi, doll, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), 3d, illustration, cartoon, (doll:0.9), octane, (worst quality, low quality:1.4), EasyNegative, bad-hands-5, nsfw, (bad and mutated hands:1.3)"
NEG2 = "(bad hands), missing fingers, multiple limbs, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, (deformed fingers:1.2), (long fingers:1.2), comic, zombie, sketch, muscles, sinewy, bad anatomy, censored, signature, monochrome, text, watermark, sketch, duplicate, bad-artist-anime, loli, mature"

RARE_SET = ['drying_cat', 'washing_cat', 'licking_person', 'kissing_teddy bear', 'feeding_cat', 'painting_fire hydrant', 'painting_vase', 'feeding_zebra', 'buying_banana', 'licking_bottle', 'cleaning_oven', 'and_bear', 'hopping on_horse', 'kissing_cow', 'riding_sheep', 'hosing_potted plant', 'exiting_train', 'cleaning_dining table', 'eating_orange', 'washing_apple', 'and_toaster', 'buying_apple', 'cleaning_toilet', 'washing_toilet', 'washing_spoon', 'buying_orange', 'inspecting_orange', 'opening_oven', 'cleaning_keyboard', 'hugging_cow', 'licking_knife', 'hopping on_motorcycle', 'stabbing_person', 'washing_sheep', 'swinging_remote', 'adjusting_snowboard', 'washing_train', 'repairing_tv', 'petting_bird', 'hugging_fire hydrant', 'washing_bicycle', 'licking_wine glass', 'setting_umbrella', 'hosing_elephant', 'washing_bowl', 'operating_microwave', 'and_snowboard', 'loading_horse', 'jumping_surfboard', 'cutting_sandwich', 'kissing_elephant', 'chasing_dog', 'holding_zebra', 'repairing_umbrella', 'flushing_toilet', 'washing_bus', 'inspecting_tennis racket', 'squeezing_orange', 'licking_bowl', 'sliding_pizza', 'washing_sink', 'adjusting_skis', 'repairing_skis', 'directing_bus', 'riding_cow', 'cutting_broccoli', 'washing_airplane', 'directing_car', 'opening_microwave', 'inspecting_handbag', 'washing_surfboard', 'cleaning_sink', 'smelling_carrot', 'repairing_clock', 'cooking_sandwich', 'kissing_sheep', 'operating_toaster', 'and_zebra', 'moving_refrigerator', 'stirring_broccoli', 'dragging_surfboard', 'washing_cup', 'repairing_parking meter', 'loading_train', 'smelling_apple', 'washing_motorcycle', 'kissing_cat', 'and_cell phone', 'smelling_broccoli', 'petting_zebra', 'cooking_carrot', 'repairing_cell phone', 'throwing_baseball bat', 'cleaning_refrigerator', 'picking_orange', 'stopping at_stop sign', 'spinning_sports ball', 'tagging_person', 'cutting_hot dog', 'stirring_carrot', 'smelling_donut', 'signing_sports ball', 'washing_boat', 'repairing_toaster', 'cutting_tie', 'licking_fork', 'cutting_orange', 'losing_umbrella', 'smelling_banana', 'washing_knife', 'waving_bus', 'cutting_banana', 'washing_carrot', 'drying_dog', 'holding_toaster', 'peeling_orange', 'zipping_suitcase', 'and_hair drier', 'hugging_suitcase', 'standing on_chair', 'cleaning_bed', 'buying_pizza', 'smelling_pizza', 'chasing_cat', 'signing_baseball bat', 'washing_broccoli', 'washing_fork', 'riding_giraffe', 'repairing_hair drier', 'cleaning_microwave', 'washing_orange', 'loading_surfboard', 'opening_toilet', 'standing on_toilet', 'washing_toothbrush', 'washing_wine glass', 'jumping_car', 'repairing_mouse'] 

def add_instruct_message(verb, object, response_content):
    # 새로운 instruct_message 생성
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
    messages.append(instruct_message)

    return messages

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
reference_images = load_reference_images('../honeybee/hico_20160224_det/hoidataset/output_rare_images.txt')

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
    new_query = f"<ImageHere>Please provide a detailed description of the image, focusing on the main person who is {verb} a {obj}. Include specific features and attributes of the person and the {obj}, as well as the overall environment and context of the interaction. Remain objective and concrete. There's no need to describe multiple people. The total word count should not exceed 76 words. Follow this template for your answer: 'A photo of a person {verb} a {obj}, [insert your description].' Please adhere to the provided template in your answer."
    
    return new_query, reference_image

def filtering_VLM_with_LLM(verb, obj, img_name, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer):
    device = "cuda"
    vlm_image = img_name
    vlm_query = f"<ImageHere>Please provide a detailed description of this image. Specifically, focus on the person, object, interaction each other, and all background in the image"
    
    response = model_eval(vlm_model, vlm_tokenizer, vlm_image, vlm_query)
    
    #* for LLM 
    if verb != "and":
        llm_prompt = "'Descriptions':" + response + "'Question':" + f" This text provides a detailed description of the image. Your task is to determine if there is a human-object interaction based on the questions I'm asking. Can you definitively determine from the text whether a person is performing the action '{verb}' on the object '{obj}'? If either the person or the object is not present, or if you cannot clearly determine the {verb} action, respond with 'no'. Only respond with 'yes' if you can definitively determine the action, person, and {obj} (including things similar to the {obj}, e.g., sunglasses can be included in glass) based on the text. Please begin your response with 'yes' or 'no', followed by your explanation."
    else :
        llm_prompt = "'Descriptions':" + response + "'Question':" + f" Can you definitively determine from the text whether a person is not interacting with a {obj}, even though both the person and {obj} are present in the image? If either the person or the object is not present, please respond with 'no'. Only respond with 'yes' if you can definitively determine the action, person, but do not {obj} (i.e. no interaction). Please begin your response with 'yes' or 'no', followed by your explanation."

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
        response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False) 
        
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
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return unet, pipe

def generate_response_for_image(model, tokenizer, ref_img, vlm_text):
    while True:
        response = model_eval(model, tokenizer, ref_img, vlm_text)
        if word_count(response) <= 65 and response.startswith(f"A photo of a person") and word_count(response) >= 10:
            break
    return response

def main():
    du.ddp_init()

    #! for VLM
    model, tokenizer = model_loader()
    #! for SDXL
    unet, pipe = SDXL_loader()
    #! for LLM
    llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(f"cuda:{du.get_rank()}").eval()
    llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    rare_list = RARE_SET
    save_dir = "./8step_prompt-filtering/"
    if du.is_main_process():
        for rare in rare_list : 
            os.makedirs(save_dir + rare, exist_ok=True)

    negative_prompt = NEG_PROMPT
    sliced_rare = du.ddp_data_split(len(rare_list), rare_list)
    if du.get_world_size()>2: dist.barrier()
    for idx, rare in tqdm(enumerate(sliced_rare), disable= not du.is_main_process()) : 
        verb, obj = rare.split("_")
        print(f"{idx} / {len(sliced_rare)} tasks")
        
        generated_images = 0
        deleted_images = 0
        gen_idx = 0
        while generated_images < 50:
            gen_idx += 1
            if generated_images % 5 == 0 or deleted_images % 5 == 0:
                vlm_text, ref_img = update_vlm_query(rare, verb, obj)
                ref_base_idr = "../HOI/hico_20160224_det/images/train2015/"
                ref_img = ref_base_idr + ref_img
                
                response = generate_response_for_image(model, tokenizer, ref_img, vlm_text)
                responses_filename = f'{save_dir}{rare}/responses.txt'
                print(f"response monitoring here : {response}")
                with open(responses_filename, 'a') as responses_file: 
                    responses_file.write(f'Iteration {gen_idx}: {response}\n')
                
            #! SDXL processing
            sdxl_prompt = response
            
            #! SDXL-FreeU technique.
            img_name = save_dir + rare + '/{}.png'.format(str(gen_idx).zfill(3))
            pipe(sdxl_prompt, num_inference_steps=8, guidance_scale=0, negative_prompt=negative_prompt, negative_prompt_2 = NEG2).images[0].save(img_name)
            
            #! filtering through VLM + LLM for long descriptions
            filter_answer = filtering_VLM_with_LLM(verb, obj, img_name, llm_model, llm_tokenizer, model, tokenizer)
            
            if filter_answer == False:
                os.remove(img_name)
                deleted_images += 1
            else :
                generated_images += 1
                print(f"{rare} dataste generated images : {generated_images}")
            

    if du.get_world_size()>2: dist.barrier()
    
if __name__ == "__main__":
    main()



