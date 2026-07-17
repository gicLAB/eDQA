import os
import re
import pickle
import numpy as np
from collections import defaultdict

def average_acc_by_prefix(folder_path):
    """
    Reads pickle files named like xxx_seed_<number>.pkl from folder_path,
    extracts 'acc' values, and computes average acc per xxx prefix.
    """

    # regex to match xxx_seed_<number>.pkl
    pattern = re.compile(r"(.+)_seed_\d+\_bit_8_ext.pkl")

    acc_values = defaultdict(list)

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if not match:
            continue
        '''
        else:
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                print(data)
        continue
        '''

        prefix = match.group(1)
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if "acc" not in data:
                print(f"Warning: 'acc' key not found in {filename}")
                continue

            acc_values[prefix].append(data["acc"])

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    avg_acc = {}
    for p, values in acc_values.items():
        # compute averages
        if values:
            avg_acc[p] = (np.mean(values), np.std(values))
    #print(avg_acc)
    return avg_acc


if __name__ == "__main__":
    folder = "/home/wenhao/DQA_Exp_Ext/classification/ResNet32_Noisy_local/experiment_res"
    avg_results = average_acc_by_prefix(folder)

    avg_results_sorted = dict(sorted(avg_results.items()))
    for prefix, avg in avg_results_sorted.items():
        print(f"{prefix}: average acc = {avg[0]:.2f}, {avg[1]:.2f}")
