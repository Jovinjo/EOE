import os
import numpy as np
import logging
import pandas as pd

# Score Saving and Loading
def save_scores(args, scores, dataset_name):
    os.makedirs(args.log_directory, exist_ok=True) 
    with open(os.path.join(args.log_directory, f'{dataset_name}_scores.npy'), 'wb') as f:
        np.save(f, scores)

def load_scores(args, dataset_name):
    with open(os.path.join(args.log_directory, f'{dataset_name}_scores.npy'), 'rb') as f:
        scores = np.load(f)
    return scores

# Logger
def setup_log(args):
    log = logging.getLogger(__name__)
    if not log.handlers: 
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        fileHandler = logging.FileHandler(os.path.join(args.log_directory, "ood_eval_info.log"), mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        log.setLevel(logging.DEBUG)
        log.addHandler(fileHandler)
        log.addHandler(streamHandler)
        log.debug(f"#########{args.name}############")
    return log

# DataFrame Logging
def save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list):
    os.makedirs(args.log_directory, exist_ok=True)

    fpr_list = [float(f"{100 * fpr:.2f}") for fpr in fpr_list]
    auroc_list = [float(f"{100 * auroc:.2f}") for auroc in auroc_list]
    aupr_list = [float(f"{100 * aupr:.2f}") for aupr in aupr_list]

    data = {k: v for k, v in zip(out_datasets, zip(fpr_list, auroc_list, aupr_list))}
    data["AVG"] = [
        float(f"{np.mean(fpr_list):.2f}"),
        float(f"{np.mean(auroc_list):.2f}"),
        float(f"{np.mean(aupr_list):.2f}")
    ]

    df = pd.DataFrame.from_dict(data, orient="index", columns=["FPR95", "AUROC", "AUPR"])
    df.to_csv(os.path.join(args.log_directory, f"{args.name}_{args.json_number}.csv"))