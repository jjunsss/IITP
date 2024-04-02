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
def save_responses_to_file(responses_dict, prompt_each_dir):
    """
    각 폴더별 응답을 .txt 파일로 저장합니다.
    """
    # responses_dict를 JSON 문자열로 변환
    responses_str = json.dumps(responses_dict, indent=4)
    
    # 저장할 파일 경로 설정
    file_path = os.path.join(prompt_each_dir, "dino_results.txt")
    
    # 파일에 응답 문자열 저장
    with open(file_path, "w") as file:
        file.write(responses_str)
        
def normalize_bbox(bboxes):
    """
    Normalize the bounding box coordinates.
    Args:
    - bboxes: A list of bounding boxes in the format [cx, cy, w, h] with normalized coordinates.
    - img_width: The width of the image.
    - img_height: The height of the image.
    
    Returns:
    - A list of bounding boxes in the format [cx, cy, w, h] with actual pixel coordinates.
    """
    bboxes = bboxes * 1024    
    return bboxes

def main():
    du.ddp_init()
    
    # 모델 및 기타 설정 로드
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    TEXT_PROMPT = "dog ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # 현재 디렉토리의 모든 jpg, jpeg, png 이미지 파일 검색
    prompt_path = Path("/home/user/jjunsss/HOI/synthetic/8step_prompt3/")
    list_prompts_path = list(prompt_path.iterdir())
    
    my_slice = du.ddp_data_split(len(list_prompts_path), list_prompts_path)

    if du.get_world_size() >= 2 : dist.barrier()
    for idx, prompt_each in enumerate(my_slice):
        results = {"positive": [], "negative": []}
        print(f"processing : {idx+1} / {len(my_slice)}")
        
        img_paths = list(prompt_each.glob("*.png"))
        img_paths = [str(path) for path in img_paths]
        
        query = str(prompt_each.name)
        
        verb, target_object = query.split("_")[0], query.split("_")[1]
        
        coco_objects = [target_object]
        
        # 각 객체명 뒤에 '.'을 추가하고, TEXT_PROMPT에 이어 붙입니다.
        TEXT_PROMPT = "person ."
        for obj in coco_objects:
            TEXT_PROMPT += obj + " ."
        print(TEXT_PROMPT)

        # if not os.path.exists('annotated_images'):
        #     os.makedirs('annotated_images')
        pbar = tqdm(total=len(img_paths))
        pbar.set_description(f"GPU {du.get_rank()} img counts :")
        
        # iteration processing
        for image_path in img_paths:
            process_successful = True  # 이미지 처리 성공 여부 플래그
            try:
                image_source, image = load_image(image_path)
            except OSError as e:
                print(f"Error loading image {image_path}: {e}")
                # 이미지 처리에 실패하면 negative 리스트에 추가
                results['negative'].append(os.path.basename(image_path))
                continue  
            
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

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
                # positive 리스트에 결과 추가
                # torch change to list format
                boxes = boxes * 1024 #synthetic image has constant image size
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
                

            pbar.update(1)

        save_responses_to_file(results, prompt_each)
    if du.get_world_size() >= 2 : dist.barrier()

if __name__ == "__main__":
    main()