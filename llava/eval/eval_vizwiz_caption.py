import os
import argparse
import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
from collections import defaultdict

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def create_coco_type(annotation_file, result_file, output_dir):
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    coco_results = []
    image_id = 1
    for result in results:
        pred = result['text']

        coco_results.append({
            "image_id": int(image_id),
            "caption": pred
        })
        image_id += 1
    output_file = 'pred_coco_type.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f_out:
        json.dump(coco_results, f_out, indent=4)
    return output_path, total

def load_json(file_path):
    """load JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_captions(pred_file, val_file, output_file):
    pred_data = load_json(pred_file)
    val_data = load_json(val_file)

    val_dict = defaultdict(list)
    for item in val_data['annotations']:
        val_dict[item['image_id']].append(item['caption'])

    merged_data = []
    for pred_item in pred_data:
        image_id = pred_item['image_id']
        pred_caption = pred_item['caption']

        gt_captions = val_dict.get(image_id, [])

        merged_data.append({
            "pred": pred_caption,
            "ground_truth": gt_captions
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

def eval_single(output_file, annotation_file, total):
    coco = COCO(annotation_file)  # Ground truth JSON file
    coco_res = coco.loadRes(output_file)  # Prediction JSON file

    coco_eval = COCOEvalCap(coco, coco_res)

    coco_eval.evaluate()

    metrics_to_print = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    results = []
    for metric, score in coco_eval.eval.items():
        if metric in metrics_to_print:
            score_percentage = score * 100.
            print(f"{metric}: {score_percentage:.2f}")
            results.append(score_percentage)
        

    print('Samples: {}\nAverage: {:.2f}%\n'.format(total, sum(results) / len(results)))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nBleu_1: {:.2f}\nBleu_2: {:.2f}\nBleu_3: {:.2f}\nBleu_4: {:.2f}\nMETEOR: {:.2f}\nROUGE_L: {:.2f}\nCIDEr: {:.2f}\nAverage: {:.2f}\n'.format(
                total, results[0], results[1], results[2], results[3], results[4], results[5], results[6], sum(results) / len(results)))

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        output_file, total = create_coco_type(args.annotation_file, args.result_file, args.output_dir)
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, args.annotation_file, ans_gt_file)
        eval_single(output_file, args.annotation_file, total)
