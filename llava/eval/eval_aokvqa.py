import os
import argparse
import json
import re
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    answer_gt_file = []
    for result in results:
        annotation = annotations[result['question_id']]
        pred = result['text']
        ground_truth = annotation['answer']
        if pred.upper() == ground_truth.upper():
            right += 1
        answer_gt_file.append({
        "pred": pred,
        "ground_truth": ground_truth
        })
    ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
    with open(ans_gt_file, "w", encoding="utf-8") as f:
        json.dump(answer_gt_file, f, ensure_ascii=False, indent=4)

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

    return ans_gt_file

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        ans_gt_file = eval_single(args.annotation_file, args.result_file)
