import torch
from torch import nn
import random
import numpy as np
import argparse
import json
import os
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses, evaluation, models
from torch.utils.data import DataLoader
import wandb
import pandas as pd
import ipdb

from utils.data_processor import read_datasets_sentence_transformers, read_dataset_raw, filter_data_sentence_transformers
from utils.data_processor import LABEL_COLUMN
from utils.metrics import compute_regression_metrics


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
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", help="The dropout probability", type=float, default=0.1)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--n_gpu", help="The number of gpus to use", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True, type=str)
    parser.add_argument("--test_filter", help="Decide which test samples to test on", type=str, default='none', choices=['none', 'easy', 'hard'])
    parser.add_argument("--train_split", help="Decide which data to train on", type=str, default='all',
                        choices=['all', 'tweets', 'news'])
    parser.add_argument("--test_split", help="Decide which test split to use", type=str, default='all',
                        choices=['all', 'tweets', 'news'])

    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
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

    # Create the model
    model = SentenceTransformer(model_name)

    dataset = read_datasets_sentence_transformers(args.data_loc, args.test_filter, train_split, test_split)
    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=args.batch_size)
    dev_data = read_dataset_raw(f"{args.data_loc}/dev.csv")
    dev_sentences1 = list(dev_data['Paper Finding'])
    dev_sentences2 = list(dev_data['News Finding'])
    dev_scores = [(s-1)/4 for s in dev_data[LABEL_COLUMN]]
    evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_sentences1, dev_sentences2, dev_scores)

    train_loss = losses.CosineSimilarityLoss(model)
    # Same loss used to train mpnet model
    #train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.n_epochs,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),
        output_path=args.output_dir,
        save_best_model=True,
        optimizer_params={'lr': args.learning_rate}
    )

    # Test
    test_data = read_dataset_raw(f"{args.data_loc}/test.csv")
    test_data = filter_data_sentence_transformers(test_data, args.test_filter, test_split)
    if args.test_filter == 'easy':
        test_data = test_data[test_data['instance_id'].str.contains('easy')]
    elif args.test_filter == 'hard':
        test_data = test_data[~test_data['instance_id'].str.contains('easy')]

    paper_embeddings = model.encode(list(test_data['Paper Finding']))
    news_embeddings = model.encode(list(test_data['News Finding']))

    scores = F.cosine_similarity(torch.Tensor(paper_embeddings), torch.Tensor(news_embeddings), dim=1).clip(
        min=0).squeeze().cpu().numpy()
    # Convert to range [1,5], assume anything below 0 is 0
    preds = (scores * 4) + 1
    labels = test_data[LABEL_COLUMN]

    metrics = compute_regression_metrics((preds, labels), prefix='test_')
    if test_split == 'all':
        tweet_selector = test_data['source'] == 'tweets'
        tweet_metrics = compute_regression_metrics((preds[tweet_selector], labels[tweet_selector]), prefix='tweet_')
        metrics.update(tweet_metrics)
        tweet_data = test_data[tweet_selector]
        tweets = list(tweet_data['News Finding'])
        paper = list(tweet_data['Paper Finding'])
        preds_tweet = preds[tweet_selector]
        labels_tweet = list(labels[tweet_selector])
        AE = np.abs(preds_tweet - np.array(labels_tweet))
        assert len(tweets) == len(paper)
        assert len(tweets) == len(preds_tweet)
        assert len(tweets) == len(labels_tweet)
        ranking = np.argsort(AE)[::-1]
        analysis_dframe = [[tweets[r], paper[r], preds_tweet[r], labels_tweet[r]] for r in
                           ranking]
        analysis_dframe = pd.DataFrame(analysis_dframe, columns=['Tweet', 'Paper', 'Pred', 'Label'])
        analysis_dframe.to_csv(f"{args.metrics_dir}/tweet_errors.csv", index=False)

        news_selector = test_data['source'] == 'news'
        news_metrics = compute_regression_metrics((preds[news_selector], labels[news_selector]), prefix='news_')
        metrics.update(news_metrics)

    wandb.log(metrics)

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

    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(metrics))