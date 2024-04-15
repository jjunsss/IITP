# %matplotlib inline

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

from pycocotools.coco import COCO
import argparse

# import ferret body
from ferret.eval.model_lvis import *
import ddp_utils

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"

# COCO 카테고리 ID와 객체 이름 매핑
coco_categories = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
    10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter",
    15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
    25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball",
    38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush"
}
voc_categories = {1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog",
    13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

def load_coco_json(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data['info'], coco_data['licenses'], coco_data['images'], {img['id']: img for img in coco_data['images']}, coco_data["categories"]

# def region_bbox(img, bbox_loc=[0.0, 0.0, 0.0, 0.0], category_id=None) -> None:
#     x, y, w, h = bbox_loc  # COCO bbox format: x, y, width, height
#     x_2 = x + w
#     y_2 = y + h
    
#     # Normalize bbox coordinates
#     img_w, img_h = img.width, img.height
#     norm_x1 = x / img_w
#     norm_y1 = y / img_h
#     norm_x2 = x_2 / img_w
#     norm_y2 = y_2 / img_h
    
#     # object_name = voc_categories.get(category_id, "object")
#     object_name = coco_categories.get(category_id, "object")
#     # question_prompt = "What Is the object <location> of the image a 'person' or a 'robot'?"
#     # question_prompt = f"Is the object <location> of the image a '{object_name}'?"
#     question_prompt = f"Considering the object <location> of the image, would you classify it as a '{object_name}' category without any doubt? Please respond with only 'yes' or 'no'."

#     # Calculate normalized width and height ratios
#     ratio_w = VOCAB_IMAGE_W / img_w
#     ratio_h = VOCAB_IMAGE_H / img_h

#     # Directly use bbox for region feature without checking region_format
#     box_x1 = int(x)
#     box_y1 = int(y)
#     box_x2 = int(x_2)
#     box_y2 = int(y_2)
#     region_coordinate_raw = [box_x1, box_y1, box_x2, box_y2]
#     segment_mask = None
    
#     box_x1_textvocab = int(norm_x1 * VOCAB_IMAGE_W)
#     box_y1_textvocab = int(norm_y1 * VOCAB_IMAGE_H)
#     box_x2_textvocab = int(norm_x2 * VOCAB_IMAGE_W)
#     box_y2_textvocab = int(norm_y2 * VOCAB_IMAGE_H)
#     # region_coordinate_raw = [box_x1_textvocab, box_y1_textvocab, box_x2_textvocab, box_y2_textvocab]

#     advanced_question = question_prompt.replace('<location>', '[{}, {}, {}, {}]'.format(box_x1_textvocab, box_y1_textvocab, box_x2_textvocab, box_y2_textvocab))
#     # If add_region_feature is True, generate the region mask
#     if args.add_region_feature:
#         region_question = advanced_question.replace('of the image', f'{DEFAULT_REGION_FEA_TOKEN} of the image')
#         generated_mask = generate_mask_for_feature(region_coordinate_raw, raw_w=int(img_w), raw_h=int(img_h), mask=segment_mask)
#         region_mask = [generated_mask]
#     else:
#         region_mask = None
#         region_question = None

#     return region_question, region_mask

def region_bbox(img, bbox_locs, category_ids):
    object_descriptions = []
    region_masks = []

    # Iterating over each object's bbox and category_id
    for i, (bbox_loc, category_id) in enumerate(zip(bbox_locs, category_ids)):
        x, y, w, h = bbox_loc
        x_2 = x + w
        y_2 = y + h
        
        img_w, img_h = img.width, img.height
        norm_x1 = x / img_w
        norm_y1 = y / img_h
        norm_x2 = x_2 / img_w
        norm_y2 = y_2 / img_h        
        
        box_x1 = int(x)
        box_y1 = int(y)
        box_x2 = int(x_2)
        box_y2 = int(y_2)    
        region_coordinate_raw = [box_x1, box_y1, box_x2, box_y2]
        segment_mask = None
        
        box_x1_textvocab = int(norm_x1 * VOCAB_IMAGE_W)
        box_y1_textvocab = int(norm_y1 * VOCAB_IMAGE_H)
        box_x2_textvocab = int(norm_x2 * VOCAB_IMAGE_W)
        box_y2_textvocab = int(norm_y2 * VOCAB_IMAGE_H)
        
        
        img_w, img_h = img.width, img.height
        norm_bbox = [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]

        object_name = coco_categories.get(category_id, "object")
        # Create a description for each object
        description = f"object [{box_x1_textvocab}, {box_y1_textvocab}, {box_x2_textvocab}, {box_y2_textvocab}] <region_fea> of the image as '{object_name}'"
        object_descriptions.append(description)

        if args.add_region_feature:
            # Generate mask for this bbox
            mask = generate_mask_for_feature(region_coordinate_raw,  raw_w=int(img_w), raw_h=int(img_h), mask=segment_mask)
            region_masks.append(mask)
    
    # Formulate a single comprehensive question with all object descriptions
    if len(object_descriptions) > 1:
        last_description = object_descriptions.pop()
        comprehensive_question = f"Considering the image, would you classify " + ", ".join(object_descriptions) + ", and " + last_description + "? Please respond with 'yes' or 'no' for each object."
    else:
        comprehensive_question = f"Considering the image, would you classify " + object_descriptions[0] + "? Please respond with 'yes' or 'no'."

    return comprehensive_question, region_masks

# @ddp_utils.ddp_on_and_off
def main(args):
    ddp_utils.ddp_init()
    
    # Model
    pseudo_lableset_path = os.path.join(args.data_path, f"annotations/{args.devide_ratio}/pseudo_coco_{args.devide_ratio}_{args.task_index}.json")
    existing_json_path = os.path.join(args.data_path, f"annotations/{args.devide_ratio}/total_pseudo_coco_{args.devide_ratio}.json")
    coco = COCO(pseudo_lableset_path)
    
    # 기존 annotation ID 저장을 위한 집합
    existing_ann_ids = set()

    # 기존 JSON 파일이 존재하면 로드
    if os.path.exists(existing_json_path):
        with open(existing_json_path, 'r') as f:
            data = json.load(f)
            # 각 annotation의 'id'를 existing_ann_ids 집합에 추가
            for ann in data["annotations"]:
                existing_ann_ids.add(ann['id'])
                
    # 모든 annotations에서 이미지 ID 추출
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)
    img_ids = list(set([ann['image_id'] for ann in annotations]))
    
    my_slice = ddp_utils.ddp_data_split(len(img_ids), img_ids)
    
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    model.cuda()
    model.orig_forward = model.forward
    
    kept_annotations = []
    filtered_count = 0
    remained_count = 0
    error_count = 0
    for image_id in tqdm(my_slice, disable=not ddp_utils.is_main_process()):
        img_info = coco.loadImgs(image_id)[0]  # 이미지 정보 로드
        img_path = os.path.join(args.data_path, "train2017" , img_info['file_name'])
        # try :
        with torch.no_grad():
            image = Image.open(img_path).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt', do_resize=True, 
                                                    do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]
            anns = coco.getAnnIds(image_id) 
            
            anns_list=[]
            for idx, ann_id in enumerate(anns):
                ann = coco.loadAnns(ann_id)[0]
                anns_list.append(ann)
                
                #* check for next iteraction
                if len(anns_list) < 2:
                    continue
                
                print(f"ann processing : {idx + 1} / {len(anns)}")

                # 이미 처리된 annotation ID인 경우 건너뛰기
                # if ann['id'] in existing_ann_ids:
                #     print(f"Annotation ID {ann['id']} already exists. Skipping.")
                #     remained_count += 1
                #     continue
                
                # if ann["score"] >= 0.5:
                #     kept_annotations.append(ann)  # 해당 annotation 추가
                #     print(f"remained annotations: over score (0.5)")
                #     remained_count += 1
                #     continue
                bbox_locs = [item["bbox"] for item in anns_list]
                categories = [item["category_id"] for item in anns_list]
                if args.region_format == "box":
                    region_question, region_masks = region_bbox(image, bbox_locs, categories)
                    region_masks = [[region_mask_i.cuda().half() for region_mask_i in region_masks]]
                    qs = region_question
                else:
                    region_masks = None
                
                # if model.config.mm_use_im_start_end:
                #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                # else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                    
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
                model.forward = partial(
                    model.orig_forward,
                    region_masks=region_masks
                )
                
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    region_masks=region_masks,
                )
                
                model.forward = model.orig_forward
                
                input_token_len = input_ids.shape[1]
                # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                # if n_diff_input_output > 0:
                    # print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
                    
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                
                # print(outputs)
                
                # 패턴에 따른 폴더 분류 로직 개선
                yes_pattern = r'^yes\b'
                # pattern = [(yes_pattern, "ferret_matched")]
                    
                # 모델의 출력이 "yes"인 경우에만 해당 annotation을 kept_annotations에 추가
                if re.search(yes_pattern, outputs.lower()):
                    kept_annotations.append(ann)  # 해당 annotation 추가
                    print(f"remained annotations: print yes pattern")
                    remained_count += 1
                else:
                    print(f"deleted annotations: print no pattern")
                    filtered_count += 1
            
                anns_list = []
            # pass_count += 1  # pass_count 처리 로직 (기존 코드에 있던 부분)
            
        # except Exception as e:
        #     image_path = Path(img_path)  # 이미지 원본 경로
        #     error_count += 1
        #     print(f"Error with image {image_path}: {e}")
    
    if ddp_utils.get_world_size() > 1: dist.barrier()
    print(f"ramaining img counts: {remained_count}, removed counts: {filtered_count}, error counts: {error_count}")
    json_folder_path = os.path.join(args.data_path, 'annotations', 'multi', f'{args.devide_ratio}', "0.3-0.5")
    
    # 폴더가 없다면 생성
    if not os.path.exists(json_folder_path):
        os.makedirs(json_folder_path, exist_ok=True)

    # JSON 파일 이름을 정의
    json_file_name = f'ferret_pseudo_labels_{args.devide_ratio}_{args.task_index}_{ddp_utils.get_rank()}.json'
    json_dir = os.path.join(json_folder_path, json_file_name)

    # JSON 파일로 COCO 포맷 데이터 저장
    with open(json_dir, 'w') as f:
        json.dump(kept_annotations, f, indent=4)
    print(f"each {json_dir} has been successfully created.")

    
    if ddp_utils.get_world_size() > 1: dist.barrier()
    if ddp_utils.is_main_process():
        merge_ddp_jsons(pseudo_lableset_path, json_folder_path, args.devide_ratio)
    if ddp_utils.get_world_size() > 1: dist.barrier()
    
    # print(f"total {json_dir} has been successfully created.")

def merge_ddp_jsons(base_json_path, json_path, devide_ratio):
    # 기본 틀 로드
    with open(base_json_path, 'r') as f:
        base_data = json.load(f)
    # 기본 틀에서 'annotations' 필드는 빈 리스트로 초기화
    base_data['annotations'] = []

    ddp_json_paths = [ f'ferret_pseudo_labels_{devide_ratio}_{args.task_index}_{idx}.json' for idx in range(4)]
    
    # DDP JSON 파일들을 순회하며 'annotations' 내용을 합침
    for ddp_json_path in ddp_json_paths:
        each_dir = os.path.join(json_path, ddp_json_path)
        print(each_dir)
        with open(each_dir, 'r') as f:
            ddp_data = json.load(f)
        base_data['annotations'].extend(ddp_data)

    json_path = os.path.join(json_path, f"ferret_total_pseudo_coco_{devide_ratio}_{args.task_index}.json")
    # 새로운 JSON 파일로 저장
    with open(json_path, 'w') as f:
        json.dump(base_data, f, indent=4)
    print("total pseudo label saved complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=1)
    parser.add_argument("--devide_ratio", type=int, default=4040)
    parser.add_argument("--model-path", type=str, default="/jjunsss/ml-ferret/model/ferret-7b-v1.3")
    parser.add_argument("--model-base", type=str, default=None) 
    parser.add_argument("--image_path", type=str, default="dataset/cocoval2017")
    parser.add_argument("--data_path", type=str, default="/jjunsss/COCODIR")
    parser.add_argument("--conv-mode", type=str, default="ferret_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--add_region_feature", action="store_true", default=True)
    parser.add_argument("--no_coor", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--region_format", type=str, default="box", choices=["point", "box", "free_shape"])
    args = parser.parse_args()

    main(args)

