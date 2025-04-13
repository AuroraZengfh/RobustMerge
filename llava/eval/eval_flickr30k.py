import os
import argparse
import json
import re
import nltk
from nltk.translate import bleu_score
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def calculate_metrics(reference_captions, generated_caption):
    # calculate BLEU score
    reference_captions_tokenized = [word_tokenize(ref) for ref in reference_captions]
    bleu_1 = bleu_score.sentence_bleu(reference_captions_tokenized, word_tokenize(generated_caption), weights=(1, 0, 0, 0))
    bleu_2 = bleu_score.sentence_bleu(reference_captions_tokenized, word_tokenize(generated_caption), weights=(0.5, 0.5, 0, 0))
    bleu_3 = bleu_score.sentence_bleu(reference_captions_tokenized, word_tokenize(generated_caption), weights=(1/3, 1/3, 1/3, 0))
    bleu_4 = bleu_score.sentence_bleu(reference_captions_tokenized, word_tokenize(generated_caption), weights=(0.25, 0.25, 0.25, 0.25))
    
    # calculate METEOR score
    meteor = meteor_score(reference_captions, generated_caption)

    # calculate ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_caption, reference_captions[0])
    rouge_l = rouge_scores[0]['rouge-l']['f']

    # calculate CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score({0: reference_captions}, {0: [generated_caption]})


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

import json

def load_captions_from_output(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    captions = {}
    for item in data:
        image_id = item['image_id']
        caption = item['caption']
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)
    
    return captions

def load_captions_from_gt(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    captions = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)
    
    return captions


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
        eval_single(output_file, args.annotation_file, total)
