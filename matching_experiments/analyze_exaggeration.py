from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict
from functools import partial
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets
import numpy as np
from utils.metrics import acc_f1
import random
import argparse
from utils.model import calibrate_temperature
import torch.nn.functional as F
import ipdb
from torch.utils.data import DataLoader
from scipy.stats import entropy
import pandas as pd
import json
from utils.data_processor import clean_tweet
from datasets import Dataset


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


def load_and_prepare_tweets(data_loc):
    with open(data_loc) as f:
        data_raw = [json.loads(l) for l in f]
    # Get pairs of sentences, with indexes to the original data
    tweet_map = {}
    paper_map = {}
    matches = set()
    for i,row in enumerate(data_raw):
        for j,tweet in enumerate(row['full_tweets']):
            tweet['exaggeration_labels'] = [-1] * len(row['paper_sentences'])
            for k, (sent,score) in enumerate(zip(row['paper_sentences'],tweet['paper_sentence_scores'])):
                if score > 3:
                    tweet_map[(i,j)] = json.loads(tweet['tweet'])['text']
                    paper_map[(i,k)] = sent
                    matches.add((i,j,k))

    tweet_dset = [[k,v] for k,v in tweet_map.items()]
    print(len(tweet_dset))
    paper_dset = [[k, v] for k, v in paper_map.items()]
    print(len(paper_dset))
    return data_raw, Dataset.from_pandas(pd.DataFrame(tweet_dset, columns=['idx', 'finding'])), Dataset.from_pandas(pd.DataFrame(paper_dset, columns=['idx', 'finding'])), matches


def load_and_prepare_news(data_loc):
    with open(data_loc) as f:
        data_raw = [json.loads(l) for l in f]
    # Get pairs of sentences, with indexes to the original data
    news_map = {}
    paper_map = {}
    matches = set()
    for i,row in enumerate(data_raw):
        for url in row['news']:
            for k,sent in enumerate(row['news'][url]):
                sent['exaggeration_labels'] = [-1] * len(row['paper'])
                assert len(row['paper']) == len(sent['paper_sentence_scores'])
                for l, (paper_sent,score) in enumerate(zip(row['paper'],sent['paper_sentence_scores'])):
                    if score > 3:
                        news_map[(i,url,k)] = sent['text']
                        paper_map[(i,l)] = paper_sent['text']
                        matches.add((i,url,k,l))

    news_idx = list(news_map.keys())
    news_dset = list(news_map.values())
    print(len(news_dset))
    paper_dset = [[k, v] for k, v in paper_map.items()]
    print(len(paper_dset))
    return data_raw, news_idx, Dataset.from_pandas(pd.DataFrame(news_dset, columns=['finding'])), Dataset.from_pandas(pd.DataFrame(paper_dset, columns=['idx', 'finding'])), matches



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        help="The location of the data to analyze",
                        type=str, required=True)
    parser.add_argument("--insciout_test_data_loc",
                        help="Location of insciout test data for model calibration",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='/home/vcx366/computational-science-journalism/artifacts/models/mt-pet-strength-prediction-4500cls-200nli/')
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)

    parser.add_argument("--output_file", help="Output file", required=True, type=str)
    parser.add_argument("--problem_type",
                        help="The problem type",
                        type=str, choices=['tweets', 'news'], default='tweets')

    args = parser.parse_args()
    model_name = args.model_name
    seed = args.seed
    enforce_reproducibility(seed)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    tk = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def preprocess_dataset(key, examples):
        batch = tk(examples[f"{key}_conclusion"], truncation=True)
        batch['label'] = examples[f"{key}_strength"]
        return batch

    # Calibrate the softmax temperature for better uncertainty estimation
    dataset = load_dataset('json', data_files={'press_release': args.insciout_test_data_loc, 'abstract': args.insciout_test_data_loc})
    dataset.cleanup_cache_files()
    press_data = dataset['press_release'].map(partial(preprocess_dataset, 'press_release'), batched=True, remove_columns=dataset['press_release'].column_names)
    abstract_data = dataset['abstract'].map(partial(preprocess_dataset, 'abstract'), batched=True, remove_columns=dataset['abstract'].column_names)

    full_data = concatenate_datasets([press_data, abstract_data])
    loader = DataLoader(full_data, collate_fn=DataCollatorWithPadding(tk), shuffle=True)
    # Already calculated
    T = 2.847956418991089#calibrate_temperature(model, loader, device)
    print(f"Final temperature {T}")

    # Get a trainer
    collator = DataCollatorWithPadding(tk)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir='./debug'),
        data_collator=collator
    )

    # Load the data we want to analyze
    def preprocess_analysis_dataset(examples):
        return tk(examples[f"finding"], truncation=True)

    # Load the dataset depending on which split (news or tweets)
    if args.problem_type == 'tweets':
        raw_data,tweet_dset,paper_dset,match_indices = load_and_prepare_tweets(args.data_loc)
        tweet_dset = tweet_dset.map(preprocess_analysis_dataset, batched=True)
        paper_dset = paper_dset.map(preprocess_analysis_dataset, batched=True)

        tweet_preds = trainer.predict(tweet_dset).predictions
        paper_preds = trainer.predict(paper_dset).predictions

        tweet_logits = np.array(tweet_preds) / T
        paper_logits = np.array(paper_preds) / T

        tweet_map = {}
        # Put the logits for each tweet
        for idx_raw,tweet_logits,finding in zip(tweet_dset['idx'], tweet_logits, tweet_dset['finding']):
            idx = tuple(idx_raw)
            assert finding == json.loads(raw_data[idx[0]]['full_tweets'][idx[1]]['tweet'])['text']
            raw_data[idx[0]]['full_tweets'][idx[1]]['strength_logits'] = [float(n) for n in tweet_logits.squeeze()]
            tweet_map[idx] = tweet_logits
        # Logits for each paper sentence
        paper_map = {}
        for idx_raw,paper_logits,finding in zip(paper_dset['idx'], paper_logits, paper_dset['finding']):
            idx = tuple(idx_raw)
            assert finding == raw_data[idx[0]]['paper_sentences'][idx[1]]
            raw_data[idx[0]]['full_paper_sentences'][idx[1]]['strength_logits'] = [float(n) for n in paper_logits.squeeze()]
            paper_map[idx] = paper_logits
        # List of exaggeration labels for matching
        for match in match_indices:
            pred_tweet = np.argmax(tweet_map[(match[0], match[1])])
            pred_paper = np.argmax(paper_map[(match[0], match[2])])

            # Downplays
            if pred_tweet < pred_paper:
                raw_data[match[0]]['full_tweets'][match[1]]['exaggeration_labels'][match[2]] = 0
            # Same
            elif pred_tweet == pred_paper:
                raw_data[match[0]]['full_tweets'][match[1]]['exaggeration_labels'][match[2]] = 1
            # Exaggerates
            else:
                raw_data[match[0]]['full_tweets'][match[1]]['exaggeration_labels'][match[2]] = 2
        with open(args.output_file, 'wt') as f:
            for row in raw_data:
                f.write(json.dumps(row) + '\n')
    elif args.problem_type == 'news':
        raw_data,news_idx,news_dset,paper_dset,match_indices = load_and_prepare_news(args.data_loc)
        news_dset = news_dset.map(preprocess_analysis_dataset, batched=True)
        paper_dset = paper_dset.map(preprocess_analysis_dataset, batched=True)

        news_preds = trainer.predict(news_dset).predictions
        paper_preds = trainer.predict(paper_dset).predictions

        news_logits = np.array(news_preds) / T
        paper_logits = np.array(paper_preds) / T

        news_map = {}
        # Put the logits for each tweet
        for idx_raw,news_logits,finding in zip(news_idx, news_logits, news_dset['finding']):
            idx = tuple(idx_raw)
            assert finding == raw_data[idx[0]]['news'][idx[1]][idx[2]]['text']
            raw_data[idx[0]]['news'][idx[1]][idx[2]]['strength_logits'] = [float(n) for n in news_logits.squeeze()]
            news_map[idx] = news_logits
        # Logits for each paper sentence
        paper_map = {}
        for idx_raw,paper_logits,finding in zip(paper_dset['idx'], paper_logits, paper_dset['finding']):
            idx = tuple(idx_raw)
            assert finding == raw_data[idx[0]]['paper'][idx[1]]['text']
            raw_data[idx[0]]['paper'][idx[1]]['strength_logits'] = [float(n) for n in paper_logits.squeeze()]
            paper_map[idx] = paper_logits
        # List of exaggeration labels for matching
        for match in match_indices:
            pred_news = np.argmax(news_map[(match[0], match[1], match[2])])
            pred_paper = np.argmax(paper_map[(match[0], match[3])])

            assert raw_data[match[0]]['news'][match[1]][match[2]]['paper_sentence_scores'][match[3]] > 3

            # Downplays
            if pred_news < pred_paper:
                raw_data[match[0]]['news'][match[1]][match[2]]['exaggeration_labels'][match[3]] = 0
            # Same
            elif pred_news == pred_paper:
                raw_data[match[0]]['news'][match[1]][match[2]]['exaggeration_labels'][match[3]] = 1
            # Exaggerates
            else:
                raw_data[match[0]]['news'][match[1]][match[2]]['exaggeration_labels'][match[3]] = 2
        with open(args.output_file, 'wt') as f:
            for row in raw_data:
                f.write(json.dumps(row) + '\n')
    # Get the calibrated softmax and take the average of the entropy of the predictions
    # finding1_H = entropy(F.softmax(torch.tensor(finding1_preds) / T, -1).numpy(), axis=-1)
    # finding2_H = entropy(F.softmax(torch.tensor(finding2_preds) / T, -1).numpy(), axis=-1)

    # avg_H = (finding1_H + finding2_H) / 2
    # ranked = np.argsort(avg_H)

    # Get the exaggeration label
    # labels = []
    # f1_label = []
    # f2_label = []
    # for f1, f2 in zip(np.argmax(finding1_preds, -1), np.argmax(finding2_preds, -1)):
    #     f1_label.append(f1)
    #     f2_label.append(f2)
    #     if f1 == f2:
    #         labels.append('SAME')
    #     elif f1 < f2:
    #         labels.append('DOWNPLAY')
    #     else:
    #         labels.append('EXAGGERATE')
    #
    # out_data = pd.read_csv(args.data_loc)
    # out_data['finding1_strength'] = f1_label
    # out_data['finding1_calibrated_probabilities'] = finding1_probs
    # out_data['finding2_strength'] = f2_label
    # out_data['finding2_calibrated_probabilities'] = finding2_probs
    # out_data['exaggeration_label'] = labels
