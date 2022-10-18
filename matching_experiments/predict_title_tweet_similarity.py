import torch
import random
import numpy as np
import argparse
import pandas as pd
from functools import partial
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import AutoConfig
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
import pandas as pd
from datasets import Dataset
from collections import defaultdict
import ipdb
import os
from sklearn.model_selection import train_test_split
import glob
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from utils.data_processor import read_dataset
from utils.trainer import CustomTrainer
from utils.metrics import compute_f1, acc_f1, compute_regression_metrics


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


def preprocess_matching_data(tk, examples):
    batch = tk(examples['title'], text_pair=examples['tweet_text'], truncation=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweet_data_loc",
                        help="The location of the data to predict on",
                        type=str, required=True)
    parser.add_argument("--s2orc_metadata_loc",
                        help="Location of s2orc metadata",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument("--base_model",
                        help="For the tokenizer",
                        type=str, default='copenlu/citebert')
    parser.add_argument("--output_file", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", help="The dropout probability", type=float, default=0.1)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--n_gpu", help="The number of gpus to use", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)


    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    # Enforce reproducibility
    # Always first
    enforce_reproducibility(seed)

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    allowable_categories = ['Computer Science', 'Psychology', 'Biology', 'Medicine']
    doi_to_category = {}
    with open(args.s2orc_metadata_loc) as f:
        for l in f:
            paper = json.loads(l)
            if any(c in allowable_categories for c in paper['mag_field_of_study']):
                doi_to_category[paper['doi']] = paper['mag_field_of_study']

    tweet_data = pd.read_csv(args.tweet_data_loc, lineterminator='\n')
    tweet_data = tweet_data[tweet_data['doi'].isin(doi_to_category)].sample(30000)
    dataset = Dataset.from_pandas(tweet_data)

    # Create the tokenizer and model
    tk = AutoTokenizer.from_pretrained(args.base_model, model_max_length=512)

    dataset = dataset.map(partial(preprocess_matching_data, tk), batched=True)

    collator = DataCollatorWithPadding(tk)

    # Create the training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=None,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.n_epochs,
        seed=seed,
        output_dir='./output'
    )

    # Get the dataset
    config = AutoConfig.from_pretrained(model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator
    )
    # pred_output = trainer.predict(dataset)
    # predictions = np.clip(pred_output.predictions, 1.0, 5.0)
    #
    # # Get the original data and attach the scores
    # tweet_data['score'] = predictions.squeeze()

    # Get and plot distributions
    # cs_scores = [row['score'] for i,row in tweet_data.iterrows() if 'Computer Science' in doi_to_category[row['doi']]]
    # psych_scores = [row['score'] for i,row in tweet_data.iterrows() if 'Psychology' in doi_to_category[row['doi']]]
    # med_scores = [row['score'] for i,row in tweet_data.iterrows() if 'Medicine' in doi_to_category[row['doi']]]
    # bio_scores = [row['score'] for i,row in tweet_data.iterrows() if 'Biology' in doi_to_category[row['doi']]]
    #
    # fig, ax = plt.subplots()
    # for a,c,l in zip([cs_scores, psych_scores, med_scores, bio_scores],
    #                     ['blue', 'red', 'green', 'purple'],
    #                     ['Computer Science', 'Psychology', 'Medicine', 'Biology']):
    #     sns.kdeplot(a, ax=ax, color=c, label=l)
    # plt.legend(loc='upper left')
    #
    # plt.savefig(args.output_file)

    cs_scores = [row['levenshtein'] for i, row in tweet_data.iterrows() if 'Computer Science' in doi_to_category[row['doi']]]
    psych_scores = [row['levenshtein'] for i, row in tweet_data.iterrows() if 'Psychology' in doi_to_category[row['doi']]]
    med_scores = [row['levenshtein'] for i, row in tweet_data.iterrows() if 'Medicine' in doi_to_category[row['doi']]]
    bio_scores = [row['levenshtein'] for i, row in tweet_data.iterrows() if 'Biology' in doi_to_category[row['doi']]]

    fig, ax = plt.subplots()
    for a, c, l in zip([cs_scores, psych_scores, med_scores, bio_scores],
                       ['blue', 'red', 'green', 'purple'],
                       ['Computer Science', 'Psychology', 'Medicine', 'Biology']):
        sns.kdeplot(a, ax=ax, color=c, label=l)
    plt.legend(loc='upper left')

    plt.savefig(args.output_file)