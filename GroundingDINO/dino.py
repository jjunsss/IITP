import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from PIL import Image
import matplotlib.pyplot as plt

import torch
import json
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

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
import os
import torch
import ddp_utils as du
import torch.distributed as dist

device = "cuda"
seed = 23   
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_grad_enabled(False)

TEXT_PROMPT = "airplane . apple . backpack . banana . baseball bat . baseball glove . bear . bed . bench . bicycle . bird . boat . book . bottle . bowl . broccoli . bus . cake . car . carrot . cat . cell phone . chair . clock . couch . cow . cup . dining table . dog . donut . elephant . fire hydrant . fork . frisbee . giraffe . hair drier . handbag . horse . hot dog . keyboard . kite . knife . laptop . microwave . motorcycle . mouse . orange . oven . parking meter . person . pizza . potted plant . refrigerator . remote . sandwich . scissors . sheep . sink . skateboard . skis . snowboard . spoon . sports ball . stop sign . suitcase . surfboard . teddy bear . tennis racket . tie . toaster . toilet . toothbrush . traffic light . train . truck . tv . umbrella . vase . wine glass . zebra ."

def save_responses_to_file(responses_dict, prompt_each_dir):
    """
    각 폴더별 응답을 .txt 파일로 저장합니다.
    """
    # responses_dict를 JSON 문자열로 변환
    responses_str = json.dumps(responses_dict, indent=4)
    
    # 저장할 파일 경로 설정
    file_path = os.path.join(prompt_each_dir, "feedbackfiltering.txt")
    
    # 파일에 응답 문자열 저장
    with open(file_path, "w") as file:
        file.write(responses_str)

def check_file(prompt_each_dir):
    # path check
    file_path = os.path.join(prompt_each_dir, "feedbackfiltering.txt")
    file_check = os.path.exists(file_path)   
    return file_check
     
@du.ddp_on_and_off
def main():
    # du.ddp_init()
    
    # 모델 및 기타 설정 로드
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.40

    #Data Loader
    prompt_path = Path("/home/user/jjunsss/HOI/GroundingDINO/dataset/total_new_crawling")
    crawl_paths = list(prompt_path.iterdir())

    sliced_crawl = du.ddp_data_split(len(crawl_paths), crawl_paths)
    if du.get_world_size()>2: dist.barrier()
    for idx, prompt_each in enumerate(sliced_crawl):
        print(f"processing : {idx+1} / {sliced_crawl}")
        
        if check_file(prompt_each):
            print(f"processing done")
            continue
        
        results = {"positive": [], "negative": []}
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
        target_object = wording[cut_index+2:].strip()

        with torch.no_grad():
            for image_path in tqdm(image_paths, disable=not du.is_main_process()):
                process_successful = True  # 이미지 처리 성공 여부 플래그
                image_source, image = load_image(image_path)
                if image is None:
                    results['negative'].append(os.path.basename(image_path))
                    continue  
                
                temp_text_conf = TEXT_TRESHOLD
                while True:
                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=TEXT_PROMPT,
                        box_threshold=temp_text_conf,
                        text_threshold=TEXT_TRESHOLD
                    )

                    check_phrase = [phrase for phrase in phrases if phrase not in TEXT_PROMPT]
                    if not check_phrase or not phrases:
                        break
                    temp_text_conf += 0.05 
                
                name = os.path.relpath(image_path)
                hoi_category = os.path.basename(os.path.dirname(image_path))
                
                if "person" not in phrases or target_object not in phrases:
                    # os.remove(image_path)
                    process_successful = False
                    continue  
                    
                # target is person scenario.
                if target_object == "person":
                    person_counts = phrases.count("person")
                    if person_counts < 2 : 
                        # os.remove(image_path)
                        process_successful = False
                        continue  
                    
                if process_successful:  # 이미지 처리 성공 시
                    # torch change to list format
                    if isinstance(boxes, torch.Tensor) :
                        boxes = boxes.tolist()
                    if isinstance(logits, torch.Tensor) :
                        logits = logits.tolist()
                    if isinstance(phrases, torch.Tensor) :
                        phrases = phrases.tolist()
                    result_item = {'boxes': boxes, 'logits': logits, 'phrases': phrases, 'image_name': name}
                    results['positive'].append(result_item)
                else:  # 이미지 처리 실패 시
                    results['negative'].append(os.path.basename(image_path))
                            
        # except Exception as e:
        #     results['negative'].append({'path': image_path, 'response': f'error generated: {e}' })
        #     continue

        save_responses_to_file(results, prompt_each)
    
    print(f"{du.get_rank()} stand by here")
    
    if du.get_world_size()>2: dist.barrier()
    return
    
if __name__ == "__main__":
    main()