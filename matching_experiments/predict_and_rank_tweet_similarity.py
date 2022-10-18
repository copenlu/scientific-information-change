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
import re
from pathlib import Path

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


def hanging_citation(text):
    """
    Checks if a sentence ends with a certain list of prepositions and ignores those that do
    :param text:
    :return:
    """
    return re.search("\s+\(?(\(\s*\)|like|reference|including|include|with|for instance|for example|see also|at|following|of|from|to|in|by|see|as|e\.?g\.?(,)?|viz(\.)?(,)?)\s*(,)*(-)*[\)\]]?\s*[.?!]\s*$", text.lower()) is not None


def preprocess_matching_data(tk, examples):
    batch = tk(examples['paper'], text_pair=examples['tweet'], truncation=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_data_loc",
                        help="The location of the data to predict on",
                        type=str, required=True)
    parser.add_argument("--unlabelled_paper_data_loc",
                        help="The location of the data to predict on",
                        type=str, required=False, default=None)
    parser.add_argument("--exclude_tweets",
                        help="The location of data to exclude",
                        type=str, default=[], nargs='+')
    parser.add_argument("--tweet_data_loc",
                        help="The location of the data to predict on",
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
    parser.add_argument("--n_samples_categories", help="The total number of samples to collect", type=int, default=100)
    parser.add_argument("--sample_dir", help="Directory to save the samples in", required=True, type=str)


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

    paper_data = pd.read_csv(args.paper_data_loc)
    paper_dois = set(paper_data['DOI'])

    #categories = ['Psychology', 'Computer Science', 'Medicine', 'Biology']
    categories = ['Computer Science', 'Psychology', 'Biology', 'Medicine']
    if args.unlabelled_paper_data_loc:
        unlabelled_csvs = []
        for cat in categories:
            unlabelled_paper_data = pd.read_csv(f"{args.unlabelled_paper_data_loc}/{cat}.csv")
            paper_dois.update(set(unlabelled_paper_data['DOI']))
            cat = cat.replace(' ', '_')
            unlabelled_paper_data['field'] = [cat]*len(unlabelled_paper_data)
            unlabelled_csvs.append(unlabelled_paper_data)

        unlabelled_paper_data = pd.concat(unlabelled_csvs)
    else:
        unlabelled_paper_data = pd.DataFrame([])

    tweet_data = pd.read_csv(args.tweet_data_loc, lineterminator='\n')
    tweet_data = tweet_data[tweet_data['doi'].isin(paper_dois)]

    # split them up by DOI
    doi_to_text = defaultdict(lambda: {'paper': [], 'paper_context': [], 'tweet': [], 'field': [], 'split': []})
    for i,sample in paper_data.iterrows():
        doi_to_text[sample['DOI']]['paper'].append(sample['Paper Finding'])
        doi_to_text[sample['DOI']]['paper_context'].append(sample['Paper Context'])
        if 'easy' in sample['instance_id']:
            cat = sample['instance_id'][:sample['instance_id'].rfind('_')][:-5]
        else:
            cat = sample['instance_id'][:sample['instance_id'].rfind('_')]
        doi_to_text[sample['DOI']]['field'].append(cat)
        doi_to_text[sample['DOI']]['split'].append(sample['split'])

    for i,sample in unlabelled_paper_data.iterrows():
        doi_to_text[sample['DOI']]['paper'].append(sample['Paper Finding'])
        doi_to_text[sample['DOI']]['paper_context'].append(sample['Paper Context'])

        cat = sample['field']
        doi_to_text[sample['DOI']]['field'].append(cat)
        doi_to_text[sample['DOI']]['split'].append('')

    for i,sample in tweet_data.iterrows():
        #if sample['levenshtein'] < 80:
        if sample['levenshtein'] < 75 and sample['title'].lower() not in sample['tweet_text'].lower():
            doi_to_text[sample['doi']]['tweet'].append(sample['tweet_text'])

    input_data = []
    for doi in doi_to_text:
        for j,p in enumerate(doi_to_text[doi]['paper']):
            for t in doi_to_text[doi]['tweet']:
                input_data.append([doi, p, t, doi_to_text[doi]['paper_context'][j], doi_to_text[doi]['field'][j], doi_to_text[doi]['split'][j]])

    dframe = pd.DataFrame(input_data, columns=['doi', 'paper', 'tweet', 'paper_context', 'field', 'split'])
    dataset = Dataset.from_pandas(dframe)

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
    # np.save('preds.npy', predictions)
    predictions = np.load('preds.npy')

    # Get the original data and attach the scores
    dframe = pd.DataFrame(input_data, columns=['doi', 'paper', 'tweet', 'paper_context', 'field', 'split'])
    dframe['score'] = predictions
    # Get the top 2 for each tweet
    out_data = []
    seen_pairs = set()
    for t,tweet in dframe.groupby('tweet'):
        ranked = list(sorted(tweet.to_numpy(), key=lambda x: x[-1], reverse=True))
        for r in ranked:
            if r[-1] > 3 and (r[1],r[2]) not in seen_pairs:
                out_data.append(r)
            seen_pairs.add((r[1],r[2]))

    out_data = pd.DataFrame(out_data, columns=['doi', 'paper', 'tweet', 'paper_context', 'field', 'split', 'score'])
    out_data.to_csv(args.output_file, index=False)

    if not os.path.exists(f"{args.sample_dir}"):
        os.makedirs(f"{args.sample_dir}")

    # Get excluded data
    seen_sentences = set()
    for dir in args.exclude_tweets:
        for csv in glob.glob(f"{dir}/*.csv"):
            curr = pd.read_csv(csv)
            seen_sentences.update(curr['Paper Finding'])
            seen_sentences.update(curr['Tweet'])
            #seen_sentences.add((curr['Paper Finding'],curr['Tweet']))

    # Separate into buckets and sample
    bucket_to_sample = defaultdict(lambda: defaultdict(list))
    leftover_to_sample = defaultdict(list)
    seen_tweets = set()
    categories = ['Computer_Science', 'Psychology', 'Biology', 'Medicine']
    #categories = ['Computer_Science', 'Psychology']
    for cat in categories:
        for i,row in dframe[dframe['field'] == cat].iterrows():
            # Normalize score to [0,1]
            score = (row['score'] - 1) / 4
            if len(row['paper']) > 20 and (row['paper'][0].isupper() or row['paper'][0] == '"') and not hanging_citation(row['paper']):
                if row['paper'] not in seen_sentences and row['tweet'] not in seen_sentences:
                    bucket_to_sample[row['field']][min(int(score * 20), 19)].append(row)
                    seen_sentences.update(set([row['paper'], row['tweet']]))
                    #seen_sentences.add((row['paper'], row['tweet']))
                # In case we need some extra data
                elif len(row['paper']) > 10 and row['tweet'] not in seen_tweets:
                    leftover_to_sample[row['field']].append(row)
                    seen_tweets.add(row['tweet'])

    # Sample and write out
    doi_counts = defaultdict(int)

    for cat in categories:
        out_data = []
        if len(bucket_to_sample[cat]) == 0:
            continue
        desired_total = args.n_samples_categories // len(bucket_to_sample[cat])
        for j,bin in enumerate(sorted(bucket_to_sample[cat], key=lambda x: len(bucket_to_sample[cat][x]))):
            size = min(len(bucket_to_sample[cat][bin]), desired_total)
            # stratify = [r['doi'] for r in bucket_to_sample[cat][bin]]
            # if size < len(bucket_to_sample[cat][bin]):
            #     _,cat_sample = train_test_split(bucket_to_sample[cat][bin], test_size=size, stratify=stratify)
            # else:
            #     cat_sample = bucket_to_sample[cat][bin]
            cat_sample = []
            # First, select any DOI that hasn't been selected
            pool = bucket_to_sample[cat][bin]
            random.shuffle(pool)
            count = 0
            while len(pool) > 0 and len(cat_sample) < size:
                j = 0
                while len(pool) > 0 and len(cat_sample) < size and j < len(pool):
                    if doi_counts[pool[j]['doi']] == count:
                        doi_counts[pool[j]['doi']] += 1
                        cat_sample.append(pool.pop(j).to_list())
                    else:
                        j += 1
                if len(pool) > 0:
                    count = min([doi_counts[p['doi']] for p in pool])
            # Now fill in based on how many times the DOI has been used
            # for item in sorted(pool, key=lambda x: doi_counts[x['doi']]):
            #     if len(cat_sample) == size:
            #         break
            #     cat_sample.append(item)
            out_data.extend(cat_sample)
            desired_total = (args.n_samples_categories // len(bucket_to_sample[cat])) + (desired_total - len(cat_sample))
            if j == len(bucket_to_sample[cat]) - 2:
                desired_total += args.n_samples_categories - (len(out_data) + desired_total)

        if len(out_data) < args.n_samples_categories:
            out_data.extend(random.sample(leftover_to_sample[cat], args.n_samples_categories - len(out_data)))
        out_data = pd.DataFrame(out_data, columns=['doi', 'Paper Finding', 'Tweet', 'Paper Context', 'field', 'split', 'score'])

        out_data.to_csv(f"{args.sample_dir}/{cat}.csv", index=None)

