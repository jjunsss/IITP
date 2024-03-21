import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import ddp_utils as du
import tqdm
import torch.distributed as dist
import random
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt') 

seed = 23   
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_grad_enabled(False)

NEG_PROMPT = "[deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry, cgi, doll, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), 3d, illustration, cartoon, (doll:0.9), octane, (worst quality, low quality:1.4), EasyNegative, bad-hands-5, nsfw, (bad and mutated hands:1.3), (bad hands), missing fingers, multiple limbs, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, (deformed fingers:1.2), (long fingers:1.2), comic, zombie, sketch, muscles, sinewy, bad anatomy, censored, signature, monochrome, text, watermark, sketch, duplicate, bad-artist-anime, loli, mature"

RARE_SET = ['drying_cat', 'washing_cat', 'licking_person', 'kissing_teddy bear', 'feeding_cat', 'painting_fire hydrant', 'painting_vase', 'feeding_zebra', 'buying_banana', 'licking_bottle', 'cleaning_oven', 'and_bear', 'hopping on_horse', 'kissing_cow', 'riding_sheep', 'hosing_potted plant', 'exiting_train', 'cleaning_dining table', 'eating_orange', 'washing_apple', 'and_toaster', 'buying_apple', 'cleaning_toilet', 'washing_toilet', 'washing_spoon', 'buying_orange', 'inspecting_orange', 'opening_oven', 'cleaning_keyboard', 'hugging_cow', 'licking_knife', 'hopping on_motorcycle', 'stabbing_person', 'washing_sheep', 'swinging_remote', 'adjusting_snowboard', 'washing_train', 'repairing_tv', 'petting_bird', 'hugging_fire hydrant', 'washing_bicycle', 'licking_wine glass', 'setting_umbrella', 'hosing_elephant', 'washing_bowl', 'operating_microwave', 'and_snowboard', 'loading_horse', 'jumping_surfboard', 'cutting_sandwich', 'kissing_elephant', 'chasing_dog', 'holding_zebra', 'repairing_umbrella', 'flushing_toilet', 'washing_bus', 'inspecting_tennis racket', 'squeezing_orange', 'licking_bowl', 'sliding_pizza', 'washing_sink', 'adjusting_skis', 'repairing_skis', 'directing_bus', 'riding_cow', 'cutting_broccoli', 'washing_airplane', 'directing_car', 'opening_microwave', 'inspecting_handbag', 'washing_surfboard', 'cleaning_sink', 'smelling_carrot', 'repairing_clock', 'cooking_sandwich', 'kissing_sheep', 'operating_toaster', 'and_zebra', 'moving_refrigerator', 'stirring_broccoli', 'dragging_surfboard', 'washing_cup', 'repairing_parking meter', 'loading_train', 'smelling_apple', 'washing_motorcycle', 'kissing_cat', 'and_cell phone', 'smelling_broccoli', 'petting_zebra', 'cooking_carrot', 'repairing_cell phone', 'throwing_baseball bat', 'cleaning_refrigerator', 'picking_orange', 'stopping at_stop sign', 'spinning_sports ball', 'tagging_person', 'cutting_hot dog', 'stirring_carrot', 'smelling_donut', 'signing_sports ball', 'washing_boat', 'repairing_toaster', 'cutting_tie', 'licking_fork', 'cutting_orange', 'losing_umbrella', 'smelling_banana', 'washing_knife', 'waving_bus', 'cutting_banana', 'washing_carrot', 'drying_dog', 'holding_toaster', 'peeling_orange', 'zipping_suitcase', 'and_hair drier', 'hugging_suitcase', 'standing on_chair', 'cleaning_bed', 'buying_pizza', 'smelling_pizza', 'chasing_cat', 'signing_baseball bat', 'washing_broccoli', 'washing_fork', 'riding_giraffe', 'repairing_hair drier', 'cleaning_microwave', 'washing_orange', 'loading_surfboard', 'opening_toilet', 'standing on_toilet', 'washing_toothbrush', 'washing_wine glass', 'jumping_car', 'repairing_mouse'] 



################################################################################################

def count_sentences(text):
    """
       주어진 텍스트에서 문장의 수를 계산합니다.
    """
    sentences = sent_tokenize(text)
    return len(sentences)

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
reference_images = load_reference_images('../hico_20160224_det/hoidataset/output_rare_images.txt')

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

# Implement logic to update the VLM query based on reference images every 5 generations
def update_vlm_query(category, verb, obj):
    # new_query = f"<ImageHere>Can you analyze the image, focusing on the person and {verb} a {obj}? Please highlight the interaction with the {obj}, emphasizing the dynamics of this action in detail. Use the template 'A photo of the person {verb} a {obj}, [your answer]' for a structured response. Please fill out the [your answer] and ensure that the description is clear and concise for image generation, keeping it within 77 words and more only one sentences."

    reference_image = call_the_image(category)

    #! prompt changed ver 5 (too mnay people and many objects)
    new_query = f"<ImageHere>Please analyze the image, focusing on the main person's interaction with the {obj}. Provide a detailed description of the dynamics of their interaction, including specific features of both the person and the {obj}, as well as the setting and context of their engagement. The description should capture the essence of the human-object interaction, emphasizing how the person is engaging with the {obj}. Structure your analysis as follows: 'A photo of the person {verb} {obj}, [insert your analysis here].' Ensure that the analysis is comprehensive enough to serve as a descriptive prompt for image generation, highlighting a single person's interaction with a single {obj}."
    
    return new_query, reference_image
        

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
    # init model and tokenizer
    vlm_model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).cuda().eval()
    vlm_tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)
    


    rare_list = RARE_SET
    save_dir = "./8step_changeprompt_v5/"
    if du.is_main_process():
        for rare in rare_list : 
            os.makedirs(save_dir + rare, exist_ok=True)

    negative_prompt = NEG_PROMPT
    sliced_rare = du.ddp_data_split(len(rare_list), rare_list)
    if du.get_world_size()>2: dist.barrier()
    for idx, rare in tqdm.tqdm(enumerate(sliced_rare), disable= not du.is_main_process()) : 
        verb, obj = rare.split("_")
        
        for i in range(50) : 
            print(f"{idx / len(sliced_rare)} tasks / generating {i+1} / {60}")
            
            if i % 5 == 0:
                vlm_text, ref_img = update_vlm_query(rare, verb, obj)
                ref_base_idr = "/home/uvll/jjunsss/HOI/hico_20160224_det/images/train2015/"
                ref_img = ref_base_idr + ref_img
                
                while True:
                    with torch.cuda.amp.autocast():
                        response, _ = vlm_model.chat(vlm_tokenizer, query=vlm_text, image=ref_img, history=[], do_sample=True)
                    
                    # condition 1: 단어 수가 77개 이하이며, 특정 형식으로 시작하는지 확인 + more then basically sentence (like A pho... ,)
                    if word_count(response) <= 67 and not response.startswith(f"A photo of the person {verb} {obj},") and  word_count(response) >= 10:
                        break  # 조건을 만족하면 루프 탈출
                                
                responses_filename = f'{save_dir}{rare}/responses.txt'
                with open(responses_filename, 'a') as responses_file: 
                    responses_file.write(f'Iteration {i}: {response}\n')
                
            #^ SDXL processing
            print(f"response monitoring here : {response}")
            sdxl_prompt = response
            
            #TODO: add additional VLM filtering (classify image matching to prompt)
            #* FreeU technique.
            # pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            pipe(sdxl_prompt, num_inference_steps=8, guidance_scale=0, negative_prompt=negative_prompt ).images[0].save(save_dir + rare + '/{}.png'.format(str(i).zfill(3)))
    if du.get_world_size()>2: dist.barrier()
    
if __name__ == "__main__":
    main()




"""
    #! prompt changed ver 1
    # new_query = f"<ImageHere>Analyze the image, focusing on the interaction between the person and {obj}. Detail the dynamics of their {verb} with {obj} in a single, comprehensive sentence. The total response must not exceed 77 words. Structure your analysis as follows: 'A photo of the person {verb} a {obj}, [insert detailed analysis here].' This description should succinctly encapsulate the essence of the human-object interaction, while also clearly and insightfully highlighting the characteristics of both the person and {obj}."
    #! prompt changed ver 2
    # new_query = f"<ImageHere>Analyze the image, focusing on the interaction between the person and {obj}. Detail the dynamics of their {verb} with {obj} in a single, comprehensive sentence. The total response must not exceed 77 words. Structure your analysis as follows: 'A photo of the person {verb} a {obj}, [insert detailed analysis here].' This description should succinctly encapsulate the essence of the human-object interaction."
    #! prompt changed ver 3
    # new_query = f"<ImageHere>Analyze the image, focusing on the interaction between the person and {obj}. Detail the dynamics of their {verb} with {obj} in a single, comprehensive sentence. The total response must not exceed 77 words. Structure your analysis as follows: 'A photo of the person {verb} a {obj}, [insert detailed analysis here].'"
    #! prompt changed ver 4 (too mnay people and many objects)
    # new_query = f"<ImageHere>Analyze the image, focusing on the person's interaction with {obj}. Write a comprehensive sentence detailing the dynamics of their {verb} with {obj}. Your description should capture the essence of the human-object interaction and the background and surroundings. Structure your analysis as follows: 'A photo of the person {verb} {obj}, [insert your answer here]."
"""
