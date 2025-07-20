import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as sk
import torch.nn.functional as F
from transformers import CLIPTokenizer

# Prompt Ensemble
def clip_text_ens(net, tokenizer, test_labels, prompt_pool):
    prompts = [template(label) for label in test_labels for template in prompt_pool]
    text_inputs = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    features = net.get_text_features(
        input_ids=text_inputs['input_ids'].cuda(),
        attention_mask=text_inputs['attention_mask'].cuda()
    ).float()

    text_features = torch.zeros(len(test_labels), features.shape[1]).cuda()
    num_templates = len(prompt_pool)
    for i in range(len(test_labels)):
        for j in range(num_templates):
            idx = i * num_templates + j
            normed = features[idx] / features[idx].norm(dim=-1, keepdim=True)
            text_features[i] += normed
    return text_features

# Filtering Text Embeddings for EOE
def pre_filter(net, tokenizer, test_labels, gpt_labels, args, prompt_pool=None):
    net.eval()
    if not args.ensemble:
        def encode(labels):
            prompts = [f"a photo of a {c}" for c in labels]
            return tokenizer(prompts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")

        test_in = encode(test_labels)
        gpt_in = encode(gpt_labels)

        test_feats = net.get_text_features(**{k: v.cuda() for k, v in test_in.items()}).float()
        gpt_feats  = net.get_text_features(**{k: v.cuda() for k, v in gpt_in.items()}).float()
    else:
        test_feats = clip_text_ens(net, tokenizer, test_labels, prompt_pool)
        gpt_feats  = clip_text_ens(net, tokenizer, gpt_labels,  prompt_pool)

    return torch.cat((test_feats, gpt_feats), dim=0)

def remove_overlap_class(test_labels, gpt_labels):
    words_set = {word for phrase in test_labels for word in phrase.split()}
    return [phrase for phrase in gpt_labels if not any(word in words_set for word in phrase.split())]

# Compute OOD Scores
def get_ood_scores_clip(args, net, loader, test_labels, gpt_labels, softmax=True):
    net.eval()
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    id_class_nums = len(test_labels)

    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    for images, _ in tqdm(loader, total=len(loader)):
        images = images.cuda()
        image_features = net.get_image_features(pixel_values=images).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if args.score in ['MCM', 'energy', 'max-logit']:
            if not args.ensemble:
                text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
                text_features = net.get_text_features(**{k: v.cuda() for k, v in text_inputs.items()}).float()
            else:
                text_features = clip_text_ens(net, tokenizer, test_labels, args.prompt_pool)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            output = image_features @ text_features.T

        elif args.score == 'EOE':
            if args.ood_task == 'near':
                gpt_labels = remove_overlap_class(test_labels, gpt_labels)
            total_features = pre_filter(net, tokenizer, test_labels, gpt_labels, args, args.prompt_pool)
            total_features /= total_features.norm(dim=-1, keepdim=True)
            output = image_features @ total_features.T

        smax = to_np(F.softmax(output / args.T, dim=1)) if softmax else to_np(output / args.T)

        # scoring logic
        if args.score == 'EOE':
            if args.score_ablation == 'EOE':
                s_id = np.max(smax[:, :id_class_nums], axis=1)
                s_ood = np.max(smax[:, id_class_nums:], axis=1)
                _score.append(-(s_id - args.beta * s_ood))
            elif args.score_ablation == 'MAX':
                id_max = torch.max(output[:, :id_class_nums], dim=1)[0]
                ood_max = torch.max(output[:, id_class_nums:], dim=1)[0]
                mask = ood_max > id_max
                output[:, :id_class_nums][mask] = 1.0 / id_class_nums
                output = output[:, :id_class_nums]
                smax = to_np(F.softmax(output / args.T, dim=1))
                _score.append(-np.max(smax, axis=1))
            elif args.score_ablation == 'MSP':
                _score.append(-np.max(smax[:, :id_class_nums], axis=1))
            elif args.score_ablation == 'energy':
                energy = args.T * (
                    torch.logsumexp(output[:, :id_class_nums] / args.T, dim=1) -
                    torch.logsumexp(output[:, id_class_nums:] / args.T, dim=1)
                )
                _score.append(-to_np(energy))
            elif args.score_ablation == 'max-logit':
                diff = torch.max(output[:, :id_class_nums], 1)[0] - torch.max(output[:, id_class_nums:], 1)[0]
                _score.append(-to_np(diff))
            else:
                raise NotImplementedError

        elif args.score == 'MCM':
            _score.append(-np.max(smax, axis=1))
        elif args.score == 'energy':
            _score.append(-to_np(args.T * torch.logsumexp(output / args.T, dim=1)))
        elif args.score == 'max-logit':
            _score.append(-to_np(torch.max(output, 1)[0]))

    return concat(_score)[:len(loader.dataset)].copy()

# Metrics
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    if not np.allclose(out[-1], np.sum(arr, dtype=np.float64), rtol=rtol, atol=atol):
        raise RuntimeError('Unstable cumsum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1):
    y_true = (y_true == pos_label)
    idx = np.argsort(y_score)[::-1]
    y_score, y_true = y_score[idx], y_true[idx]
    distinct_idxs = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_idxs, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    recall = tps / tps[-1]
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    recall, fps = np.r_[recall[sl], 1], np.r_[fps[sl], 0]
    return fps[np.argmin(np.abs(recall - recall_level))] / np.sum(~y_true)

def get_measures(pos, neg, recall_level=0.95):
    pos, neg = np.array(pos), np.array(neg)
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])
    auroc = sk.roc_auc_score(labels, scores)
    aupr = sk.average_precision_score(labels, scores)
    fpr = fpr_and_fdr_at_recall(labels, scores, recall_level)
    return auroc, aupr, fpr

# Final Evaluation Printer
def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    auroc, aupr, fpr = get_measures(-in_score, -out_score)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    print_measures(log, auroc, aupr, fpr, args.score)

def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log is None:
        print(f'FPR{int(100 * recall_level)}:\t\t\t{100 * fpr:.2f}')
        print(f'AUROC: \t\t\t{100 * auroc:.2f}')
        print(f'AUPR:  \t\t\t{100 * aupr:.2f}')
    else:
        log.debug(f'\t\t\t\t{method_name}')
        log.debug(f'  FPR{int(100 * recall_level)} AUROC AUPR')
        log.debug(f'& {100*fpr:.2f} & {100*auroc:.2f} & {100*aupr:.2f}')