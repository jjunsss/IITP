import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import ddp_utils as du
import tqdm
import torch.distributed as dist
import random
# from inf import *
from tqdm import tqdm
from pathlib import Path
from pipeline.interface import get_model
from PIL import Image
import os


seed = 23   
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_grad_enabled(False)

NEG_PROMPT = "[deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry, cgi, doll, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), 3d, illustration, cartoon, (doll:0.9), octane, (worst quality, low quality:1.4), EasyNegative, bad-hands-5, nsfw, (bad and mutated hands:1.3)"
NEG2 = "(bad hands), missing fingers, multiple limbs, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, (deformed fingers:1.2), (long fingers:1.2), comic, zombie, sketch, muscles, sinewy, bad anatomy, censored, signature, monochrome, text, watermark, sketch, duplicate, bad-artist-anime, loli, mature"

RARE_SET = ['drying_cat', 'washing_cat', 'licking_person', 'kissing_teddy bear', 'feeding_cat', 'painting_fire hydrant', 'painting_vase', 'feeding_zebra', 'buying_banana', 'licking_bottle', 'cleaning_oven', 'and_bear', 'hopping on_horse', 'kissing_cow', 'riding_sheep', 'hosing_potted plant', 'exiting_train', 'cleaning_dining table', 'eating_orange', 'washing_apple', 'and_toaster', 'buying_apple', 'cleaning_toilet', 'washing_toilet', 'washing_spoon', 'buying_orange', 'inspecting_orange', 'opening_oven', 'cleaning_keyboard', 'hugging_cow', 'licking_knife', 'hopping on_motorcycle', 'stabbing_person', 'washing_sheep', 'swinging_remote', 'adjusting_snowboard', 'washing_train', 'repairing_tv', 'petting_bird', 'hugging_fire hydrant', 'washing_bicycle', 'licking_wine glass', 'setting_umbrella', 'hosing_elephant', 'washing_bowl', 'operating_microwave', 'and_snowboard', 'loading_horse', 'jumping_surfboard', 'cutting_sandwich', 'kissing_elephant', 'chasing_dog', 'holding_zebra', 'repairing_umbrella', 'flushing_toilet', 'washing_bus', 'inspecting_tennis racket', 'squeezing_orange', 'licking_bowl', 'sliding_pizza', 'washing_sink', 'adjusting_skis', 'repairing_skis', 'directing_bus', 'riding_cow', 'cutting_broccoli', 'washing_airplane', 'directing_car', 'opening_microwave', 'inspecting_handbag', 'washing_surfboard', 'cleaning_sink', 'smelling_carrot', 'repairing_clock', 'cooking_sandwich', 'kissing_sheep', 'operating_toaster', 'and_zebra', 'moving_refrigerator', 'stirring_broccoli', 'dragging_surfboard', 'washing_cup', 'repairing_parking meter', 'loading_train', 'smelling_apple', 'washing_motorcycle', 'kissing_cat', 'and_cell phone', 'smelling_broccoli', 'petting_zebra', 'cooking_carrot', 'repairing_cell phone', 'throwing_baseball bat', 'cleaning_refrigerator', 'picking_orange', 'stopping at_stop sign', 'spinning_sports ball', 'tagging_person', 'cutting_hot dog', 'stirring_carrot', 'smelling_donut', 'signing_sports ball', 'washing_boat', 'repairing_toaster', 'cutting_tie', 'licking_fork', 'cutting_orange', 'losing_umbrella', 'smelling_banana', 'washing_knife', 'waving_bus', 'cutting_banana', 'washing_carrot', 'drying_dog', 'holding_toaster', 'peeling_orange', 'zipping_suitcase', 'and_hair drier', 'hugging_suitcase', 'standing on_chair', 'cleaning_bed', 'buying_pizza', 'smelling_pizza', 'chasing_cat', 'signing_baseball bat', 'washing_broccoli', 'washing_fork', 'riding_giraffe', 'repairing_hair drier', 'cleaning_microwave', 'washing_orange', 'loading_surfboard', 'opening_toilet', 'standing on_toilet', 'washing_toothbrush', 'washing_wine glass', 'jumping_car', 'repairing_mouse'] 



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
reference_images = load_reference_images('hico_20160224_det/hoidataset/output_rare_images.txt')

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
    new_query = f"Please provide a detailed description of the image, focusing on the main person who is {verb} a {obj}. Include specific features and attributes of the person and the {object}, as well as the overall mood and context of the interaction. Keep in mind that you don't need to describe many people. The total word count should be within 76 words. Structure your answer as follows template: 'A photo of a person {verb} a {obj}, [insert your description].' "
    
    return new_query, reference_image

def construct_input_prompt(user_prompt):
    SYSTEM_MESSAGE = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    IMAGE_TOKEN = "Human: <image>\n" #<image> denotes an image placehold.
    USER_PROMPT = f"Human: {user_prompt}\n"

    return SYSTEM_MESSAGE + IMAGE_TOKEN + USER_PROMPT + "AI: "

def model_eval(model, tokenizer, processor, image_path, prompt):
    image_list = [image_path]
    images = [Image.open(_).convert("RGB") for _ in image_list]
    prompts = [construct_input_prompt(prompt)]

    inputs = processor(text=prompts, images=images)
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512
            }
    
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
        sentence = tokenizer.batch_decode(res, skip_special_tokens=True)    

    return sentence[0]

def model_loader():
    # Load trained model
    ckpt_path = "checkpoints/13B-C-Abs-M576/last"
    model, tokenizer, processor = get_model(ckpt_path, use_bf16=True)
    model.cuda()
    return model, tokenizer, processor

def main():
    du.ddp_init()
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"# Use the correct ckpt for your step setting!

    torch.cuda.empty_cache()
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(f"cuda:{du.get_rank()}", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=f"cuda:{du.get_rank()}"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(f"cuda:{du.get_rank()}")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    #! for VLM
    model, tokenizer, processor = model_loader()


    rare_list = RARE_SET
    save_dir = "./8step_prompt2/"
    if du.is_main_process():
        for rare in rare_list : 
            os.makedirs(save_dir + rare, exist_ok=True)

    negative_prompt = NEG_PROMPT
    sliced_rare = du.ddp_data_split(len(rare_list), rare_list)
    if du.get_world_size()>2: dist.barrier()
    for idx, rare in tqdm(enumerate(sliced_rare), disable= not du.is_main_process()) : 
        verb, obj = rare.split("_")
        
        for i in range(50) : 
            print(f"{idx / len(sliced_rare)} tasks / generating {i+1} / {60}")
            
            if i % 5 == 0:
                vlm_text, ref_img = update_vlm_query(rare, verb, obj)
                ref_base_idr = "../HOI/hico_20160224_det/images/train2015/"
                ref_img = ref_base_idr + ref_img
                
                while True:
                    response = model_eval(model, tokenizer, processor, ref_img, vlm_text)
                    # condition 1: 단어 수가 77개 이하이며, 특정 형식으로 시작하는지 확인 + more then basically sentence (like A pho... ,)
                    if word_count(response) <= 67 and response.startswith(f"A photo of a person {verb} a {obj}") and  word_count(response) >= 10:
                        break  # 조건을 만족하면 루프 탈출
                                
                responses_filename = f'{save_dir}{rare}/responses.txt'
                print(f"response monitoring here : {response}")
                with open(responses_filename, 'a') as responses_file: 
                    responses_file.write(f'Iteration {i}: {response}\n')
                
            #! SDXL processing
            sdxl_prompt = response
            
            #! SDXL-FreeU technique.
            # pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            pipe(sdxl_prompt, num_inference_steps=8, guidance_scale=0, negative_prompt=negative_prompt, negative_prompt_2 = NEG2).images[0].save(save_dir + rare + '/{}.png'.format(str(i).zfill(3)))
    if du.get_world_size()>2: dist.barrier()
    
if __name__ == "__main__":
    main()



''''
f"Please describe the image, focusing on the main person's interaction ({verb}) with the {obj}. Provide a detailed description of the dynamics of their interaction, including specific features of both the person and the {obj}, as well as the setting and context of their engagement. The description should capture how the person is {verb} the {obj}. 

Please detailed describe the image, focusing on the main person {verb} a {obj}. Include specific features, attributes of the person and {object}, as well as the entire environment mood and context of the interaction, where only one person and one interaction. Structure your analysis as follows: 'A photo of a person {verb} a {obj}, [insert your desscribe] within the number of all words 76 words.

prompt2 : f"Please provide a detailed description of the image, focusing on the main person who is {verb} a {obj}. Include specific features and attributes of the person and the {object}, as well as the overall mood and context of the interaction. Keep in mind that you don't need to describe many people. The total word count should be within 76 words. Structure your answer as follows template: 'A photo of a person {verb} a {obj}, [insert your description].' "
'''