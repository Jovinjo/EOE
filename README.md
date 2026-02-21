<h1 align="center">Experiments on Envisioning Outlier Exposure (EOE)</h1>

![](img/framework.png)

This repository contains experiments **based on** the paper **"Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection"** published at **ICML 2024** by Cao et al. It is build on the [codebase](https://github.com/Aboriginer/EOE) and paper [arXiv](https://arxiv.org/pdf/2406.00806)

### Recognition
This Anomaly Detection project was awarded **Best Pitch Presentation**. [View Certificate](recognition-certificate.pdf) [View Final Presentation](final-presentation.pdf)

### Abstract
Detecting out-of-distribution (OOD) samples is an important step when deploying machine learning models in real-world settings, where unexpected inputs can often occur. The EOE (Envisioning Outlier Exposure) method tackles this challenge in a zero-shot setting, meaning it doesn’t require access to OOD data during training. Instead, it uses large language models (LLMs) like GPT to generate imagined outlier labels that are visually similar but semantically different from the known classes. These envisioned labels help expand the classifier’s understanding of what “unknown” might look like. In this project, I explored how well EOE works across three common OOD tasks: far, near, and fine-grained detection. I followed the original setup for far and fine-grained tasks but used CIFAR-10 and CIFAR-100 for near-OOD detection. I also extended the original method by adding semantic variants pipeline to the envisioned outlier labels. This helped improve the model’s ability to detect subtle differences, especially in fine-grained tasks. Overall, the results confirm EOE’s effectiveness and show that enriching the outlier label pool with diverse terms can further boost performance.

## Setup
### Dependencies
```bash
pip install -r requirements.txt
# please add your openai in .env
touch .env
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
```


### Dataset Preparation

#### In-distribution Datasets

We consider the following datasets used in our experiments:

- **In-distribution (ID) datasets**:  
  [`CUB-200`](http://www.vision.caltech.edu/datasets/cub_200_2011/),  
  [`Oxford-IIIT Pet`](https://www.robots.ox.ac.uk/~vgg/data/pets/),  
  `CUB-100 (ID)`, `Oxford-Pet-18 (ID)`,  
  [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html)

- **Out-of-distribution (OOD) datasets**:  
  [`DTD (Describable Textures Dataset)`](https://www.robots.ox.ac.uk/~vgg/data/dtd/),  
  `CUB-100 (OOD)`, `Oxford-Pet-19 (OOD)`,  
  [`CIFAR-100`](https://www.cs.toronto.edu/~kriz/cifar.html)

Specifically:
1. `CUB-200` and `Oxford-IIIT Pet` are used as ID datasets for **far-OOD detection**, paired with the `Texture` dataset as OOD.
2. `CUB-100` and `Oxford-Pet-18` are subsets of the above datasets, used for **fine-grained OOD detection**, where both ID and OOD classes come from the same domain.  
   The selected class indices are provided in:  
   - `data/CUB-100/selected_100_classes.pkl`  
   - `data/Oxford-Pet-18/selected_18_classes.pkl`
3. `CIFAR-10` is used as the ID dataset for **near-OOD detection**, paired with `CIFAR-100`.

All datasets should be placed inside the `./datasets` directory. CIFAR-10/100 will be automatically downloaded when running the code.

## Quick Start

The main script for evaluating OOD detection is `eval_ood_detection.py`. Below are the main arguments:

- `--in_dataset`: In-distribution dataset (e.g., `cub100_ID`, `pet18_ID`, `cifar10`)
- `--ood_task`: OOD task type (`far`, `near`, `fine_grained`)
- `--score`: OOD detection score (`EOE`, `MCM`, `energy`, `max-logit`)
- `--generate_class`: Generate envisioned OOD class labels using LLMs
- `--L`: Number of envisioned OOD class labels (e.g., `300` for far/fine-grained, `3` for near)
- `--llm_model`: LLM model name (default: `gpt-3.5-turbo`)
- `--score_ablation`: Use specific score function for ablation (`EOE`, `MAX`, `MSP`, `energy`, `max-logit`)
- `--ensemble`: Use CLIP prompt ensembling
- `--use_synonyms`: Add synonyms to envisioned labels using LLM
- `--prompt_pool_id`: Prompt variant to use with `--ensemble`
- `--gpu`: GPU index (e.g., `0`)
- `--seed`: Random seed (default: `5`)
- `--T`: Temperature for softmax (default: `1.0`)
