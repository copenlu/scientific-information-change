import torch
import random
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
import pandas as pd
from datasets import Dataset
import wandb
import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from datasets import load_dataset
import ipdb

import torch.nn.functional as F

from utils.data_processor import read_covert_dataset, read_covidfact_dataset
from utils.metrics import acc_f1, compute_regression_metrics
from utils.rank_metrics import mean_average_precision, mean_reciprocal_rank


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def bm25(claims, evidence):
    corpus = [e[1].split(" ") for e in evidence]
    bm25 = BM25Okapi(corpus)
    ranked_lists = []
    for claim,ev_id in tqdm(claims):
        # Create dataset to pass through model
        preds = bm25.get_scores(claim.split(" "))
        # Get order
        rank = np.argsort(preds)[::-1]
        # Get labels
        labels = np.zeros(len(evidence))
        labels[ev_id] = 1
        ranked_lists.append(labels[rank])

    return {'MAP': mean_average_precision(ranked_lists), 'MRR': mean_reciprocal_rank(ranked_lists)}


def sts_model(claims, evidence, args):
    #Load the model
    model = SentenceTransformer(args.model_name)
    evidence_embeddings = model.encode([e[1] for e in evidence])
    claim_embeddings = model.encode([c[0] for c in claims])

    scores = util.cos_sim(claim_embeddings, evidence_embeddings)
    ranked_lists = []
    for row_score,(claim, ev_id) in zip(scores, claims):
        # Get order
        rank = np.argsort(row_score.numpy())[::-1]
        # Get labels
        labels = np.zeros(len(evidence))
        labels[ev_id] = 1
        ranked_lists.append(labels[rank])

    return {'MAP': mean_average_precision(ranked_lists), 'MRR': mean_reciprocal_rank(ranked_lists)}


def trained_model(claims, evidence, args):
    tk = AutoTokenizer.from_pretrained(args.base_model, model_max_length=512)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    training_args = TrainingArguments(
        output_dir=args.output_dir
    )
    collator = DataCollatorWithPadding(tk)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator
    )
    def preprocess(examples):
        batch = tk(examples['claim'], text_pair=examples['evidence'], truncation=True)
        return batch
    # Iterate through each claim and get a ranked list
    ranked_lists = []
    for claim,ev_id in tqdm(claims):
        # Create dataset to pass through model
        pairs = [[e[0], claim, e[1]] for e in evidence]
        dframe = pd.DataFrame(pairs, columns=['id', 'claim', 'evidence'])
        dataset = Dataset.from_pandas(dframe)
        dataset = dataset.map(preprocess, batched=True)
        pred_output = trainer.predict(dataset)
        preds = pred_output.predictions.squeeze()
        # Get order
        rank = np.argsort(preds)[::-1]
        # Get labels
        labels = np.zeros(len(evidence))
        labels[ev_id] = 1
        ranked_lists.append(labels[rank])

    return {'MAP': mean_average_precision(ranked_lists), 'MRR': mean_reciprocal_rank(ranked_lists)}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        help="The location of the COVERT data",
                        type=str, required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='copenlu/citebert')
    parser.add_argument("--base_model",
                        help="The base model",
                        type=str, default='copenlu/citebert')
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--dataset",
                        help="The name of the dataset to use",
                        type=str, default='covert', choices=['covert', 'covidfact'])

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True, type=str)
    parser.add_argument("--eval_type", help="Decide which test samples to test on", type=str, default='ours', choices=['ours', 'sts', 'bm25'])


    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    run_name = args.run_name
    eval_type = args.eval_type
    dataset_name = args.dataset
    config = {
        'run_name': run_name,
        'seed': seed,
        'model_name': model_name,
        'output_dir': args.output_dir,
        'tags': args.tags,
        'eval_type': args.eval_type,
        'dataset': dataset_name
    }

    run = wandb.init(
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tags
    )

    # Enforce reproducibility
    # Always first
    enforce_reproducibility(seed)

    # Load the data
    if dataset_name == 'covert':
        claim_label,evidence = read_covert_dataset(args.data_loc)
    elif dataset_name == 'covidfact':
        claim_label, evidence = read_covidfact_dataset(args.data_loc)

    if eval_type == 'ours':
        metrics = trained_model(claim_label, evidence, args)
    elif eval_type == 'bm25':
        metrics = bm25(claim_label, evidence)
    elif eval_type == 'sts':
        metrics = sts_model(claim_label, evidence, args)


    wandb.log(metrics)
    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(metrics))
