import torch
import random
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer, util
from transformers import DataCollatorWithPadding
from collections import defaultdict
import json
import wandb
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sns.set()

from utils.data_processor import clean_tweet


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
                        type=str, choices=['tweets', 'news'], default='tweets')
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

    with open(args.data_loc) as f:
        data = [json.loads(l) for l in f]

    model = SentenceTransformer(model_name)
    predictions = []

    if problem_type == 'tweets':
        for row in tqdm(data):
            tweet_embeddings = model.encode([clean_tweet(t) for t in row['tweets']])
            paper_embeddings = model.encode(list(row['paper_sentences']))
            # Convert to range [1,5]
            scores = (util.cos_sim(tweet_embeddings, paper_embeddings).clip(min=0, max=1) * 4) + 1
            for s,tweet in zip(scores, row['full_tweets']):
                tweet['paper_sentence_scores'] = s.tolist()
                predictions.extend(s.tolist())
    elif problem_type == 'news':
        for row in tqdm(data):
            news_text = [sent['text'] for url in row['news'] for sent in row['news'][url]]
            news_embeddings = model.encode(news_text)
            paper_text = [p['text'] for p in row['paper']]
            paper_embeddings = model.encode(paper_text)
            # Convert to range [1,5]
            scores = (util.cos_sim(news_embeddings, paper_embeddings).clip(min=0, max=1) * 4) + 1
            j = 0
            for url in row['news']:
                for sent in row['news'][url]:
                    sent['paper_sentence_scores'] = scores[j].tolist()
                    predictions.extend(scores[j].tolist())
                    j += 1

    # Get the original data and attach the scores
    with open(args.output_file, 'wt') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

    fig = plt.figure(figsize=(6,5))
    sns.kdeplot(predictions)
    plt.tight_layout()
    plt.savefig('data/dev-dist.png')


