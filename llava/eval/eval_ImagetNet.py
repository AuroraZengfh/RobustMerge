import os
import argparse
import json
import re
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


def eval_single(test_file, result_file):
    annotations = json.load(open(test_file))
    answers = [test['answer'] for test in annotations]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    false_answers = []
    answer_gt_file = []
    for index in tqdm(range(total)):
        text = answers[index]
        label = results[index]
        if (text.upper() in label['text'].upper()) or (label['text'].upper() in text.upper()):
            right += 1
        else:
            label['ground_truth'] = text
            false_answers.append(label)
        answer_gt_file.append({
        "pred": label['text'],
        "ground_truth": text
        })
    ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
    with open(ans_gt_file, "w", encoding="utf-8") as f:
        json.dump(answer_gt_file, f, ensure_ascii=False, indent=4)
        
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
            json.dump(false_answers,f,indent=4)

    return ans_gt_file

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        ans_gt_file = eval_single(args.test_file, args.result_file)