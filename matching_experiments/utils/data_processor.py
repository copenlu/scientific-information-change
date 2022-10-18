import ipdb
import numpy as np
from datasets import load_dataset
from datasets import Dataset
from functools import partial
import pandas as pd
import json
from sentence_transformers import InputExample
import re


LABEL_COLUMN = 'final_score_hard'

def clean_tweet(text):
    no_html = re.sub(r'http\S+', '', text)
    return re.sub(r'@([A-Za-z]+[A-Za-z0-9-_]+)', '@username', no_html)


def prepare_context_sample(text, context):
    text_loc = context.find(text)
    text_end = text_loc + len(text)
    return f"{context[:text_loc]}{{{{{text}}}}}{context[text_end:]}"


def preprocess_matching_data(tk, type, use_context, examples):
    if use_context:
        paper = [prepare_context_sample(f,c) for f,c in zip(examples['Paper Finding'], examples['Paper Context'])]
        news = [prepare_context_sample(n,c) for n,c in zip(examples['News Finding'], examples['News Context'])]
        batch = tk(paper, text_pair=news, truncation=True)
    else:
        batch = tk(examples['Paper Finding'], text_pair=examples['News Finding'], truncation=True)
    if LABEL_COLUMN in examples or 'binary_label' in examples:
        if type == 'regression':
            batch['label'] = [float(l) for l in examples[LABEL_COLUMN]]
        else:
            batch['label'] = [int(l) for l in examples['binary_label']]
    return batch


def filter_data(data, filter, split):
    if filter == 'easy':
        data = data.filter(lambda example: 'easy' in example['instance_id'])
    elif filter == 'hard':
        data = data.filter(lambda example: 'easy' not in example['instance_id'])

    # TODO: See what the actual fields/names are
    if split == 'all':
        return data
    elif split == 'tweets':
        return data.filter(lambda example: example['source'] == 'tweets')
    elif split == 'news':
        return data.filter(lambda example: example['source'] == 'news')


def read_datasets(data_loc, tokenizer, type='regression', test_filter='none',
                  train_split='all', test_split='all', use_context=False):
    # Something like this
    dataset = load_dataset('csv', data_files={'train': f"{data_loc}/train.csv",
                                              'validation': f"{data_loc}/dev.csv",
                                              'test': f"{data_loc}/test.csv"})
    dataset['train'] = filter_data(dataset['train'], 'none', train_split)
    dataset['validation'] = filter_data(dataset['validation'], test_filter, test_split)
    dataset['test'] = filter_data(dataset['test'], test_filter, test_split)

    dataset.cleanup_cache_files()
    dataset = dataset.map(partial(preprocess_matching_data, tokenizer, type, use_context), batched=True)

    return dataset


def read_dataset(data_loc, tokenizer, type='regression'):
    # Something like this
    dataset = load_dataset('csv', data_files={'data': f"{data_loc}"})
    dataset.cleanup_cache_files()
    dataset = dataset.map(partial(preprocess_matching_data, tokenizer, type, False), batched=True,
                          remove_columns=dataset['train'].column_names)

    return dataset


def preprocess_matching_data_domains(tk, examples):
    selector = np.array(examples[LABEL_COLUMN]) > 3
    batch = tk(list(np.array(examples['Paper Finding'])[selector]) + list(np.array(examples['News Finding'])[selector]),
               truncation=True)
    batch['label'] = [0] * sum(selector) + [1] * sum(selector)
    return batch


def read_datasets_domain(data_loc, tokenizer):
    # Something like this
    dataset = load_dataset('csv', data_files={'train': f"{data_loc}/train.csv",
                                              'validation': f"{data_loc}/dev.csv",
                                              'test': f"{data_loc}/test.csv"})
    dataset.cleanup_cache_files()
    dataset = dataset.map(partial(preprocess_matching_data_domains, tokenizer), batched=True,
                          remove_columns=dataset['train'].column_names)

    return dataset


def read_dataset_raw(data_loc):
    return pd.read_csv(data_loc).fillna('')


def read_unlabelled_tweet_dataset(data_loc, tokenizer):
    with open(data_loc) as f:
        data = [json.loads(l) for l in f]

    dframe = []
    for row in data:
        for tweet in row['full_tweets']:
            for sent in row['paper_sentences']:

                dframe.append([row['doi'], tweet['tweet_id'], clean_tweet(json.loads(tweet['tweet'])['text']), sent])

    dframe = pd.DataFrame(dframe, columns=['doi', 'tweet_id', 'News Finding', 'Paper Finding'])
    dataset = Dataset.from_pandas(dframe)
    dataset = dataset.map(partial(preprocess_matching_data, tokenizer, type, False), batched=True)
    return dataset



def filter_data_sentence_transformers(data, filter, split):
    if filter == 'easy':
        data = data[data['instance_id'].str.contains('easy')]
    elif filter == 'hard':
        data = data[~data['instance_id'].str.contains('easy')]

    # TODO: See what the actual fields/names are
    if split == 'all':
        return data
    elif split == 'tweets':
        return data[data['source'] == 'tweets']
    elif split == 'news':
        return data[data['source'] == 'news']


def read_datasets_sentence_transformers(data_loc, test_filter='none', train_split='all', test_split='all'):
    # Something like this
    dataset = {}
    for split in ['train', 'dev', 'test']:
        data = read_dataset_raw(f"{data_loc}/{split}.csv")
        if split == 'train':
            data = filter_data_sentence_transformers(data, 'none', train_split)
        else:
            data = filter_data_sentence_transformers(data, test_filter, test_split)
        samples = []
        for i,row in data.iterrows():
            samples.append(InputExample(texts=[row['Paper Finding'], row['News Finding']],
                                        label=(row[LABEL_COLUMN] - 1) / 4)) # Convert score to [0,1]
        name = split if split != 'dev' else 'validation'
        dataset[name] = samples
    return dataset


def read_covert_dataset(data_loc):
    with open(data_loc) as f:
        all_data = [json.loads(l) for l in f]
    # Collect all of the evidence and use that as the corpus
    id_ = 0
    claim_labels = []
    evidence = []
    # Because there are repeats
    ev_to_id = {}
    for row in all_data:
        claim = clean_tweet(row['claim'])
        labs_curr = []
        for ev in row['evidence']:
            if ev[0] != 'NOT ENOUGH INFO' and isinstance(ev[2], str):
                if ev[2] in ev_to_id:
                    labs_curr.append(ev_to_id[ev[2]])
                else:
                    labs_curr.append(id_)
                    evidence.append([id_, ev[2]])
                    ev_to_id[ev[2]] = id_
                    id_ += 1
        claim_labels.append([claim, labs_curr])

    print(len(evidence))
    return claim_labels,evidence


def read_covidfact_dataset(data_loc):
    with open(data_loc) as f:
        all_data = [json.loads(l) for l in f]
    # Collect all of the evidence and use that as the corpus
    id_ = 0
    claim_labels = []
    evidence = []
    ev_to_id = {}
    for row in all_data:
        claim = row['claim']
        labs_curr = []
        for ev in row['evidence']:
            if ev in ev_to_id:
                labs_curr.append(ev_to_id[ev])
            else:
                labs_curr.append(id_)
                evidence.append([id_, ev])
                ev_to_id[ev] = id_
                id_ += 1
        claim_labels.append([claim, labs_curr])

    print(len(evidence))
    return claim_labels,evidence