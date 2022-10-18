import argparse
import random
from functools import partial
import wandb
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

from utils.data_processor import read_datasets
from utils.metrics import acc_f1, compute_regression_metrics


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
                        type=str, default='coderpotter/adversarial-paraphrasing-detector')
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True, type=str)

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--test_filter", help="Decide which test samples to test on", type=str, default='none', choices=['none', 'easy', 'hard'])


    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    run_name = args.run_name
    test_filter = args.test_filter
    config = {
        'run_name': run_name,
        'seed': seed,
        'model_name': model_name,
        'output_dir': args.output_dir,
        'tags': args.tags,
        'test_filter': test_filter
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

    categories = ['Medicine', 'Biology', 'Psychology', 'Computer_Science']


    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Create the tokenizer and model
    tk = AutoTokenizer.from_pretrained(model_name)
    # config = AutoConfig.from_pretrained(model_name, num_labels=3)
    # Initialization warning is apparently normal: https://github.com/huggingface/transformers/issues/5421
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    dataset = read_datasets(args.data_loc, tk, test_filter=test_filter)
    labels = np.array(dataset['test']['label'])
    dataset = dataset.remove_columns("label")

    collator = DataCollatorWithPadding(tk)

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        max_grad_norm=None,
        logging_dir=args.output_dir,
        save_strategy='epoch',
        seed=seed,
        run_name=args.run_name,
        load_best_model_at_end=True,
        report_to=['tensorboard', 'wandb']
    )

    # Get the dataset
    # Create the trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=collator
    )

    pred_output = trainer.predict(dataset['test'])
    logits_orig = pred_output.predictions
    # Take a softmax over logits
    probs = F.softmax(torch.tensor(logits_orig), -1)
    similarity = probs[:, 1]
    # Convert to range [1,5]
    preds = (similarity * 4) + 1
    metrics = compute_regression_metrics((preds, labels), prefix='unsupervised_')

    tweet_selector = [source == 'tweets' for source in dataset['test']['source']]
    tweet_idx = np.where(tweet_selector)
    tweet_metrics = compute_regression_metrics((preds[tweet_idx], labels[tweet_idx]),
                                               prefix='tweet_')
    metrics.update(tweet_metrics)

    news_selector = [source == 'news' for source in dataset['test']['source']]
    news_idx = np.where(news_selector)
    news_metrics = compute_regression_metrics((preds[news_idx], labels[news_idx]),
                                              prefix='news_')
    metrics.update(news_metrics)

    for cat in categories:
        selector = [cat in instance_id for instance_id in dataset['test']['instance_id']]
        # Predict
        idx = np.where(selector)
        metrics_cat = compute_regression_metrics((preds[idx], labels[idx]), prefix=cat + '_')
        metrics.update(metrics_cat)
        tweets = list(np.array(selector) & np.array(tweet_selector))
        idx = np.where(tweets)
        metrics_cat = compute_regression_metrics((preds[idx], labels[idx]), prefix=cat + '_tweet_')
        metrics.update(metrics_cat)
        news = list(np.array(selector) & np.array(news_selector))
        idx = np.where(news)
        metrics_cat = compute_regression_metrics((preds[idx], labels[idx]),
                                                 prefix=cat + '_news_')
        metrics.update(metrics_cat)

    wandb.log(metrics)

    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(metrics))
