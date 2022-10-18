import torch
import random
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import wandb
import json
import os
import ipdb

import torch.nn.functional as F

from utils.data_processor import read_dataset_raw
from utils.metrics import compute_regression_metrics
from utils.data_processor import LABEL_COLUMN


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
                        help="The location of the test data",
                        type=str, required=True)
    parser.add_argument("--eval_data_loc",
                        help="The location of the validation data",
                        type=str, required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='all-MiniLM-L6-v2')
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True, type=str)
    parser.add_argument("--test_filter", help="Decide which test samples to test on", type=str, default='none', choices=['none', 'easy', 'hard'])

    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    run_name = args.run_name
    test_filter = args.test_filter

    categories = ['Medicine', 'Biology', 'Psychology', 'Computer_Science']

    config = {
        'run_name': run_name,
        'seed': seed,
        'model_name': model_name,
        'output_dir': args.output_dir,
        'tags': args.tags,
        'test_filter': args.test_filter
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
    val_data = read_dataset_raw(args.eval_data_loc)
    test_data = read_dataset_raw(args.data_loc)
    if args.test_filter == 'easy':
        test_data = test_data[test_data['instance_id'].str.contains('easy')]
    elif args.test_filter == 'hard':
        test_data = test_data[~test_data['instance_id'].str.contains('easy')]

    # Load the model
    model = SentenceTransformer(model_name)

    paper_embeddings = model.encode(list(test_data['Paper Finding']))
    news_embeddings = model.encode(list(test_data['News Finding']))

    scores = F.cosine_similarity(torch.Tensor(paper_embeddings), torch.Tensor(news_embeddings), dim=1).clip(min=0).squeeze().cpu().numpy()
    # Convert to range [1,5], assume anything below 0 is 0
    preds = (scores * 4) + 1

    labels = test_data[LABEL_COLUMN]

    metrics = compute_regression_metrics((preds,labels), prefix='unsupervised_')
    tweet_selector = test_data['source'] == 'tweets'
    tweet_metrics = compute_regression_metrics((preds[tweet_selector], labels[tweet_selector]), prefix='tweet_')
    metrics.update(tweet_metrics)

    news_selector = test_data['source'] == 'news'
    news_metrics = compute_regression_metrics((preds[news_selector], labels[news_selector]), prefix='news_')
    metrics.update(news_metrics)

    for cat in categories:
        selector = test_data['instance_id'].str.contains(cat)
        # Predict
        metrics_cat = compute_regression_metrics((preds[selector], labels[selector]), prefix=cat + '_')
        metrics.update(metrics_cat)
        tweet_selector = selector & (test_data['source'] == 'tweets')
        metrics_cat = compute_regression_metrics((preds[tweet_selector], labels[tweet_selector]), prefix=cat + '_tweet_')
        metrics.update(metrics_cat)
        news_selector = selector & (test_data['source'] == 'news')
        metrics_cat = compute_regression_metrics((preds[news_selector], labels[news_selector]),
                                                 prefix=cat + '_news_')
        metrics.update(metrics_cat)
        wandb.log(metrics)

    wandb.log(metrics)
    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(metrics))