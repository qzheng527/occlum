import json
import argparse
import numpy as np 
# from collections import defaultdict
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, default=None)
    args = parser.parse_args()

    arr = np.arange(0, 4, 0.3)
    x = np.arange(0, 5, 0.1)
    calibration_y = 1 / (1 + np.exp(-x))

    chat_rank_total = [0] * len(arr)
    chat_rank_correct = [0] * len(arr)

    open_asis_total = [0] * len(arr)
    open_asis_correct = [0] * len(arr)

    with open(os.path.join(args.score_path, "pred_scores.jsonl"), 'r', encoding='utf8') as fin:
        for line in fin:
            line_dict = json.loads(line)
            chosen_score = line_dict['chosen_score']
            rejected_score = line_dict['rejected_score']
            score = abs(chosen_score - rejected_score)
            for ind in reversed(range(len(arr))):
                if score >= arr[ind]:
                    break
                else:
                    pass
            
            open_asis_total[ind] += 1

            if chosen_score > rejected_score:
                open_asis_correct[ind] += 1
        
        # ipdb.set_trace()
        open_asis_acc = [open_asis_correct[ind] / open_asis_total[ind] for ind in range(len(arr))]
    
    open_asis_acc = np.array(open_asis_acc)
    plt.plot(arr, open_asis_acc, label="model_rank")
    plt.plot(x, calibration_y, label="calibration", linestyle="--")
    plt.xlabel("score difference")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.score_path, "score.png"))



