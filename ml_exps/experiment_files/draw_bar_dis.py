import os
import json
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({'font.size': 15})

def plot_json_folder(json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    #file = 'DQA_res_abl_vit_im_bit_4_dist'
    #file = 'DQA_res_abl_vit_cifar_bit_4_dist'
    #file = 'DQA_res_abl_resnet18_im_bit_4_dist'
    #file = 'DQA_res_abl_resnet18_cifar_bit_4_dist'
    #file = 'DQA_res_abl_mobilev2_im_bit_4_dist'
    #file = 'DQA_res_abl_mobilev2_cifar_bit_4_dist'
    #file = 'DQA_res_abl_resnet32_im_bit_4_dist'
    #file = 'DQA_res_abl_resnet32_cifar_bit_3_dist'

    for filename in os.listdir(json_folder):
        if not filename.endswith("dist.pkl"):
            continue
        print(filename)
        #if file not in filename:
        #    continue

        json_path = os.path.join(json_folder, filename)

        # Load JSON
        with open(json_path, "rb") as f:
            p_data = pickle.load(f)
            #print(p_data)
        data = p_data

        if not data:
            print(f"Skipping empty file: {filename}")
            continue

        # Compute averages
        sums = defaultdict(float)
        count = len(data)

        for item in data:
            for key, value in item.items():
                sums[key] += value

        averages = {key: sums[key] / count for key in sums}
        averages = dict(sorted(averages.items()))
        # Plot bar chart
        bits_values = {1: '0.125', 2: '0.25', 3: '0.375', 4: '0.5', 5: '0.625', 6: '0.75', 7: '0.875', 0: '0.0'}
        keys = [bits_values[i] for i in averages.keys()]
        values = list(averages.values())
        #print(keys, values)
        plt.figure(figsize=(8, 5))
        plt.bar(keys, values, color='orange', width=0.5)
        plt.xlabel("Key")
        plt.ylabel("Average Value (log scale)")
        #plt.title(f"Averages for {filename}")
        plt.yscale("log")  # <-- log scale
        plt.xticks(rotation=45)
        #plt.xlim(0, 0.9)
        plt.ylim(500000, 400000000000)
        plt.xlabel("Shift-Out Errors")
        plt.ylabel("Avg. Frequencies")
        plt.grid(axis='y')
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(
            output_folder, filename.replace(".pkl", ".png")
        )
        plt.savefig(output_path)
        plt.close()

        print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    json_folder = "/home/wenhao/DQA_Exp_Ext/classification/ResNet32_Dqa_sep_overhead_size_correct_local/experiment_res"
    output_folder = "/home/wenhao/DQA_Exp_Ext/classification/ResNet32_Dqa_sep_overhead_size_correct_local/non_bold_dis"

    plot_json_folder(json_folder, output_folder)
