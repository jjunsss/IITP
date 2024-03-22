# %matplotlib inline

from PIL import Image
import matplotlib.pyplot as plt

import torch
import json
from pathlib import Path
from pipeline.interface import get_model
import os
import shutil
import argparse
import os
import torch
import torch.distributed as dist
from tqdm import tqdm

import ddp_utils
import numpy as np
import re


def construct_input_prompt(user_prompt):
    SYSTEM_MESSAGE = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    IMAGE_TOKEN = "Human: <image>\n" #<image> denotes an image placehold.
    USER_PROMPT = f"Human: {user_prompt}\n"

    return SYSTEM_MESSAGE + IMAGE_TOKEN + USER_PROMPT + "AI: "

def init_image_pil(img_path):
    # PIL로 이미지를 로드하고 RGB로 변환
    img = Image.open(img_path).convert('RGB')
    
    # # 원하는 크기로 이미지를 리사이즈 (예: 가로가 512 이상인 경우)
    # if img.width > 512:
    #     ratio = 512 / float(img.width)
    #     new_height = int(float(img.height) * ratio)
    #     img = img.resize((512, new_height), Image.ANTIALIAS)
    
    # numpy 배열로 변환
    np_img = np.array(img)
    
    return np_img

'''
    "Please give me a designated person action description based on neigbor relationship objects. \
    Does the person carrying a person? If you think so, please tell me 'yes'. But if you don't think so, please tell me 'no'."
'''
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
    
@ddp_utils.ddp_on_and_off
def main(cfg):
    # init(cfg)
    
    with open("appendix/dict_results.json") as f:
        dino_file = json.load(f)

    # load person-object matching verbs docs
    with open("appendix/hoi-verbs-ex.json") as f:
        pair_docs = json.load(f)
        
    #! load SEEM json fiel
    with open("appendix/hoi-verbs-ex.json") as f:
        ''' image / path / mask '''
        seem_docs = json.load(f)
        
    # Load trained model
    ckpt_path = "checkpoints/13B-C-Abs-M576/last"
    model, tokenizer, processor = get_model(ckpt_path, use_bf16=True)
    model.cuda()
    print("Model initialization is done.")

    # load crawling dataset
    find_path = Path("/home/uvll/jjunsss/dataset/total_new_crawling")
    image_paths = list(find_path.glob("**/*.jpg")) + list(find_path.glob("**/*.jpeg")) + list(find_path.glob("**/*.png"))
    image_paths = [str(path) for path in image_paths]  # Path 객체를 문자열로 변환
    print(f"total : {len(image_paths)}")


    image_length = len(image_paths)

    
    rank = ddp_utils.get_rank()
    world_size = ddp_utils.get_world_size()
    per_image_length = image_length / world_size
    start_idx = rank * int(per_image_length)
    end_idx = start_idx + per_image_length if rank != world_size - 1 else image_length
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    my_slice = image_paths[start_idx:end_idx]
    print(f"sliced total : {len(my_slice)}")
    
    pbar = tqdm(len(my_slice), disable=not ddp_utils.is_main_process())
    pbar.set_description("processing", )
    
    pass_count = 0
    del_count = 0
    for image_path in my_slice:
        try:
            image_list = [image_path]
            query = image_path.split("/")[-3]

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
            
            images = [Image.open(_).convert("RGB") for _ in image_list]
            
            # if verb == "and":
            #     '''
            #         and is no interaction.
            #     '''
            #     custome_p = f"Please give me a person action description based on nearby objects. Does the person no interact with {object}? If you think so, please tell me 'yes'. But if you don't think so, please tell me 'no'."
            # else:
            #     custome_p = f"Please give me a person action description based on nearby objects. Does the person {verb} a {object}? If you think so, please tell me 'yes'. But if you don't think so, please tell me 'no'."
            
            custome_p = "Please provide a detailed description of this image, including all objects, people, background, interactions, and relationships."
            prompts = [construct_input_prompt(custome_p)]

            inputs = processor(text=prompts, images=images)
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # generate kwargs (the same in transformers) can be passed in the do_generate()
            generate_kwargs = {
                'do_sample': True,
                'top_k': 5,
                'max_length': 512
            }

            with torch.no_grad():
                res = model.generate(**inputs, **generate_kwargs)
            sentence = tokenizer.batch_decode(res, skip_special_tokens=True)

            print(f"prompt: {sentence[0]}")
            continue
            pattern = r'\bno\b'

            # 결과에 따라 이미지를 처리
            if re.search(pattern, sentence[0].lower()):
                # 이미지가 있는 폴더와 동일한 위치에 not-matched 폴더 생성
                not_matched_folder = os.path.join(os.path.dirname(image_path), "not-matched")
                # os.makedirs(not_matched_folder, exist_ok=True)

                # 이미지를 not-matched 폴더로 이동
                new_image_path = os.path.join(not_matched_folder, os.path.basename(image_path))
                # shutil.move(image_path, new_image_path)
                print(f"Image moved to {new_image_path}")

            pbar.update(1)
            pass_count += 1
            
        except Exception as e:
            print(f"Error with image {image_path}: {e}")

            # 에러 이미지를 error 폴더로 이동
            error_folder = os.path.join(os.path.dirname(image_path), "error")
            # os.makedirs(error_folder, exist_ok=True)

            error_image_path = os.path.join(error_folder, os.path.basename(image_path))
            # shutil.move(image_path, error_image_path)
            print(f"Error image moved to {error_image_path}")
            pbar.update(1)
            del_count += 1
            continue
        
    
        
    print(f"pass imgs : {pass_count} \t del imgs : {del_count}")
    # ddp_utils.stop_ddp()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('DDP', add_help=False)
    cfg = parser.parse_args()
    main(cfg)