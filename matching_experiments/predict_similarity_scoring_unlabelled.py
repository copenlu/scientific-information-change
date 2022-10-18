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
from collections import defaultdict
import json
import wandb
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

from utils.data_processor import read_unlabelled_tweet_dataset
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        help="The location of the data to predict on",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument("--problem_type",
                        help="The problem type",
                        type=str, choices=['regression', 'single_label_classification'], default='regression')
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
    problem_type = args.problem_type
    num_labels = 1 if problem_type == 'regression' else 2
    # Enforce reproducibility
    # Always first
    enforce_reproducibility(seed)

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Create the tokenizer and model
    tk = AutoTokenizer.from_pretrained(model_name)

    dataset = read_unlabelled_tweet_dataset(args.data_loc, tk)

    collator = DataCollatorWithPadding(tk)

    # Get the F1 metric
    compute_metric = compute_regression_metrics if problem_type == 'regression' else partial(acc_f1, 'binary')

    # Create the training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=None,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.n_epochs,
        seed=seed,
        output_dir='./output'
    )

    # Get the dataset
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, problem_type=problem_type)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator
    )
    pred_output = trainer.predict(dataset)
    predictions = pred_output.predictions
    # Group the scores by tweet ID and sentence
    tweet_id_to_scores = defaultdict(list)
    for id_,pred in zip(dataset['tweet_id'], predictions):
        tweet_id_to_scores[id_].append(float(pred))

    # Open the original data
    with open(args.data_loc) as f:
        data = [json.loads(l) for l in f]

    for row in data:
        for tweet in row['full_tweets']:
            tweet['sentence_scores'] = tweet_id_to_scores[tweet['tweet_id']]

    # Get the original data and attach the scores
    with open(args.output_file, 'wt') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

    sns.kdeplot(predictions.squeeze())
    plt.savefig('data/dev-dist.png')


