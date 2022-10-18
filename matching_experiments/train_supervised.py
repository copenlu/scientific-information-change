import torch
import random
import numpy as np
import argparse
import json
import os
from functools import partial
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import AutoConfig
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
import wandb
import pandas as pd
import ipdb

from utils.data_processor import read_datasets
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
                        help="The location of the training data",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument("--problem_type",
                        help="The problem type",
                        type=str, choices=['regression', 'single_label_classification'], default='regression')
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", help="The dropout probability", type=float, default=0.1)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--n_gpu", help="The number of gpus to use", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True, type=str)
    parser.add_argument("--test_filter", help="Decide which test samples to test on", type=str, default='none', choices=['none', 'easy', 'hard'])
    parser.add_argument("--train_split", help="Decide which data to train on", type=str, default='all', choices=['all', 'tweets', 'news'])
    parser.add_argument("--test_split", help="Decide which test split to use", type=str, default='all', choices=['all', 'tweets', 'news'])
    parser.add_argument("--use_context", help="Flag to switch to using the context", action='store_true')


    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    problem_type = args.problem_type
    test_filter = args.test_filter
    use_context = args.use_context
    num_labels = 1 if problem_type == 'regression' else 2
    train_split = args.train_split
    test_split = args.test_split
    # Enforce reproducibility
    # Always first
    enforce_reproducibility(seed)
    config = {
        'run_name': args.run_name,
        'seed': seed,
        'model_name': model_name,
        'output_dir': args.output_dir,
        'tags': args.tags,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'epochs': args.n_epochs,
        'seed': args.seed,
        'problem_type': args.problem_type,
        'test_filter': args.test_filter,
        'use_context': use_context,
        'train_split': train_split,
        'test_split': test_split
    }

    run = wandb.init(
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tags
    )

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    categories = ['Medicine', 'Biology', 'Psychology', 'Computer_Science']

    # Create the tokenizer and model
    if 'scibert' in model_name or 'citebert' in model_name:
        tk = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    else:
        tk = AutoTokenizer.from_pretrained(model_name)

    dataset = read_datasets(args.data_loc, tk, problem_type, test_filter, train_split, test_split, use_context)

    collator = DataCollatorWithPadding(tk)

    # Get the F1 metric
    compute_metric = compute_regression_metrics if problem_type == 'regression' else partial(acc_f1, 'binary')

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=None,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.n_epochs,
        logging_dir=args.output_dir,
        save_strategy='epoch',
        seed=seed,
        run_name=args.run_name,
        load_best_model_at_end=True,
        report_to=['tensorboard', 'wandb']
    )

    # Get the dataset
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, problem_type=problem_type)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    if problem_type == 'regression':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=compute_metric,
            data_collator=collator
        )
    else:
        # Get label weights
        labels = dataset["train"]['label']
        weight = torch.tensor(len(labels) / (2 * np.bincount(labels)))
        weight = weight.type(torch.FloatTensor)
        # Create the trainer and train
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=compute_metric,
            data_collator=collator,
            weight=weight
        )

    train_output = trainer.train()
    model.save_pretrained(args.output_dir)
    tk.save_pretrained(args.output_dir)
    pred_output = trainer.predict(dataset['test'])
    pred_metrics = pred_output.metrics
    wandb.log(pred_metrics)
    if test_split == 'all':
        # Get tweet performance
        tweet_data = dataset['test'].filter(lambda example: example['source'] == 'tweets')
        tweet_metrics = trainer.predict(tweet_data, metric_key_prefix='tweet')
        pred_metrics.update(tweet_metrics.metrics)
        # Rank the errors for analysis
        preds = tweet_metrics.predictions.squeeze()
        labels = tweet_data['final_score_hard']
        AE = np.abs(preds - np.array(labels))
        ranking = np.argsort(AE)[::-1]
        analysis_dframe = [[tweet_data['News Finding'][r], tweet_data['Paper Finding'][r], preds[r], labels[r]] for r in ranking]
        analysis_dframe = pd.DataFrame(analysis_dframe, columns=['Tweet', 'Paper', 'Pred', 'Label'])
        analysis_dframe.to_csv(f"{args.metrics_dir}/tweet_errors.csv", index=False)

        # Get news performance
        news_data = dataset['test'].filter(lambda example: example['source'] == 'news')
        news_metrics = trainer.predict(news_data, metric_key_prefix='news')
        pred_metrics.update(news_metrics.metrics)


    # Iterate through the categories
    for cat in categories:
        curr_dataset = dataset['test'].filter(lambda example: cat in example['instance_id'])
        # Predict
        pred_output = trainer.predict(curr_dataset, metric_key_prefix=cat)
        pred_metrics.update(pred_output.metrics)
        tweet_curr = curr_dataset.filter(lambda example: example['source'] == 'tweets')
        pred_output = trainer.predict(tweet_curr, metric_key_prefix=cat + '_tweet')
        pred_metrics.update(pred_output.metrics)
        news_curr = curr_dataset.filter(lambda example: example['source'] == 'news')
        pred_output = trainer.predict(news_curr, metric_key_prefix=cat + '_news')
        pred_metrics.update(pred_output.metrics)
        wandb.log(pred_metrics)

    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(pred_metrics))
