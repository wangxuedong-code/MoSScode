# model_trainer.py - 模型训练
import json
from model import models
import os

if __name__ == "__main__":


    PARAM_GRID = {
        "data":["Heart_Disease.xlsx",],
    }
    num_list = [[8, 2], ]  # num_list = [[8, 2], [7, 3], [6, 4]]

    for group in num_list:
        breaklB = group[0]
        breaklS = group[1]
        for data in PARAM_GRID["data"]:
            para_C = 16
            para_K = 0.625
            para_δ = 0.45
            para_ϵ = 55
            para_k = 3
            para_σ = 1
            all_results, metrics_mean, metrics_std, Y_sample, Accuracy = models(data, para_C, para_K, para_δ, para_ϵ, para_k, para_σ, breaklB, breaklS)
            filename = f"results-{breaklB}-{breaklS}/Main-experiment-MOSS-{data}-{Accuracy.round(3)}.json"
            folder_path = os.path.dirname(filename)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(filename, 'w') as f:
                json.dump({"fold_results": all_results, "mean": metrics_mean, "std": metrics_std,
                           "Y_sample": Y_sample}, f, indent=2)














