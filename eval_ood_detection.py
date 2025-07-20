import os
os.environ['CURL_CA_BUNDLE'] = ''
import argparse
import numpy as np
import torch
from scipy import stats

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader
from utils.generate_llm_class import load_llm_classes
from utils.args_pool import *
from utils.synonym_fetcher import SynonymFetcher 

def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    out_datasets = dataset_mappings.get(args.in_dataset, [])
    test_loader = set_val_loader(args, preprocess)
    test_labels = list(get_test_labels(args, test_loader))
    print("Original ID Labels:", test_labels)

    if args.score == 'EOE':
        print("Using LLM-generated OOD candidate classes...")
        base_ood_labels = load_llm_classes(args, test_labels)

        if args.use_synonyms:
            print("Augmenting OOD candidates with synonyms...")
            synonym_fetcher = SynonymFetcher()
            class_type = synonym_fetcher.dataset_info[args.in_dataset]["class_type"]

            base_ood_labels = [lbl.strip().lower() for lbl in base_ood_labels]
            original_count = len(base_ood_labels)

            synonym_labels = []
            for lbl in base_ood_labels:
                syn = synonym_fetcher.fetch(class_type, lbl, args.in_dataset).strip().lower()
                print(f"  → {lbl!r}  → synonym: {syn!r}")
                if syn != lbl:
                    synonym_labels.append(syn)

            combined = base_ood_labels + synonym_labels
            filtered = [label for label in combined if label and label.lower() not in {"none", "null"}]
            llm_labels = list(dict.fromkeys(filtered))  # Deduplicate
            print(f"OOD classes before synonym expansion: {original_count}")
            print(f"OOD classes after synonym + deduplication: {len(llm_labels)}")
        else:
            llm_labels = base_ood_labels
    else:
        llm_labels = []

    print(f"\n Final ID Labels passed to CLIP: {test_labels}")
    print(f"OOD candidate labels: {llm_labels}\n")

    in_score = get_ood_scores_clip(args, net, test_loader, test_labels, llm_labels)
    auroc_list, aupr_list, fpr_list = [], [], []

    for out_dataset in out_datasets:
        log.debug(f"Evaluating OOD dataset {out_dataset}")
        ood_loader = set_ood_loader(args, out_dataset, preprocess)
        out_score = get_ood_scores_clip(args, net, ood_loader, test_labels, llm_labels)

        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)

    log.debug('\n\n Mean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)

def process_args():
    parser = argparse.ArgumentParser(
        description='Leverage LLMs for OOD Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--in_dataset', default='cub100_ID', choices=ALL_ID_DATASET, help='in-distribution dataset')
    parser.add_argument('--root_dir', default="datasets", help='root dir of datasets')
    parser.add_argument('--ensemble', action='store_true', default=False, help='CLIP text prompt engineering')
    parser.add_argument('--L', type=int, default=100, help='length of envisioned OOD class labels')
    parser.add_argument('--beta', type=float, default=0.25, help='beta for scoring')
    parser.add_argument('--ood_task', type=str, default='far', choices=ALL_OOD_TASK, help='choose OOD task')
    parser.add_argument('--generate_class', action='store_true', help='generate or load envisioned OOD class')
    parser.add_argument('--json_number', type=int, default=0, help='which json to load')
    parser.add_argument('--llm_model', default="gpt-3.5-turbo", choices=ALL_LLM, help='LLM model')
    parser.add_argument('--name', default="eval_ood", help='run name ID')
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--T', type=float, default=1.0, help='temperature')
    parser.add_argument('--model', default='CLIP', choices=['CLIP'], help='model arch')
    parser.add_argument('--CLIP_ckpt', default='ViT-B/16', choices=['ViT-B/16'], help='CLIP checkpoint')
    parser.add_argument('--score', default='EOE', choices=['EOE', 'MCM', 'energy', 'max-logit'], help='score method')
    parser.add_argument('--score_ablation', default='MAX', choices=['MAX', 'MSP', 'energy', 'max-logit', 'EOE'], help='ablation mode')
    parser.add_argument('--feat_dim', type=int, default=512, help='feature dim')
    parser.add_argument('--use_synonyms', action='store_true', help='add synonyms using LLM')
    parser.add_argument('--prompt_pool_id', type=int, default=0, help='prompt ensemble index')

    args = parser.parse_args()
    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)
    return args

if __name__ == '__main__':
    main()
