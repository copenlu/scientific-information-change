import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import os
import random
from collections import defaultdict
from functools import partial
import ipdb

import matplotlib as mpl
import numpy as np
import scipy
import seaborn as sns
import torch
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import editdistance
import json

sns.set(font_scale=1.4)

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel

from lexicalrichness import LexicalRichness

from utils.data_processor import LABEL_COLUMN


colors = ['red', 'blue']


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


def make_ellipses(gmm, ax, clusters_to_classes):
    """
    Adds Ellipses to ax according to the gmm clusters.
    Taken from https://github.com/roeeaharoni/unsupervised-domain-clusters/blob/master/src/domain_clusters.ipynb
    """

    for n in sorted(list(clusters_to_classes.keys())):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        class_id = clusters_to_classes[n]
        class_color = colors[n]
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=class_color, linewidth=0)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def map_clusters_to_classes_by_majority(y_train, y_train_pred):
    """
    Maps clusters to classes by majority to compute the Purity metric.
    Taken from https://github.com/roeeaharoni/unsupervised-domain-clusters/blob/master/src/domain_clusters.ipynb
    """
    cluster_to_class = {}
    for cluster in np.unique(y_train_pred):
        # run on indices where this is the cluster
        original_classes = []
        for i, pred in enumerate(y_train_pred):
            if pred == cluster:
                original_classes.append(y_train[i])
        # take majority
        cluster_to_class[cluster] = max(set(original_classes), key = original_classes.count)
    return cluster_to_class


def preprocess_data(tk, examples):
    selector = np.array(examples[LABEL_COLUMN]) > 3
    batch = tk(list(np.array(examples['Paper Finding'])[selector]) + list(np.array(examples['News Finding'])[selector]), truncation=True)
    batch['source'] = [0] * sum(selector) + [1] * sum(selector)
    return batch


def read_dataset(data_loc, tokenizer):
    # Something like this
    dataset = load_dataset('csv', data_files={'train': data_loc})
    dataset.cleanup_cache_files()
    dataset = dataset.map(partial(preprocess_data, tokenizer), batched=True, remove_columns=dataset['train'].column_names)

    return dataset



if __name__ == '__main__':
    # e.g.: python domain_and_lexical_analysis.py --data_loc data/prolific_round_2/train.csv --model_name bert-base-uncased --domains paper news --output_dir domain_analysis --new_vectors --metrics_dir metrics/prolific_round_2/
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        help="The location of the training data",
                        type=str, required=True)
    parser.add_argument("--model_name", help="The name of the model being tested. Can be a directory for a local model",
                        required=True, type=str)
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--new_vectors", action="store_true", help="Whether or not to get new vectors")
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True, type=str)


    args = parser.parse_args()
    enforce_reproducibility(args.seed)

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Get the model
    tk = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    full_dataset = read_dataset(args.data_loc, tk)
    collator = DataCollatorWithPadding(tk)

    for split in ['news', 'tweets', 'full']:

        if split == 'full':
            dataset = full_dataset
        else:
            dataset = full_dataset.filter(lambda example: example['source'] == split)

        dloader = DataLoader(dataset['train'], collate_fn=collator, batch_size=args.batch_size)


        #### Domain analysis


        # Create numpy memory map
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # if args.new_vectors:
        #     memmap = np.memmap(f'{args.output_dir}/{split}_vectors.dat', dtype='float32', mode='w+', shape=(len(dataset['train']), config.hidden_size))
        #     i = 0
        #     model.eval()
        #     with torch.no_grad():
        #         for batch in tqdm(dloader):
        #             input_ids = batch['input_ids'].to(device)
        #             masks = batch['attention_mask'].to(device)
        #             source = batch['source']
        #
        #             outputs = model(input_ids)
        #             # avg pooled
        #             hidden_states = outputs['last_hidden_state'] * masks.unsqueeze(-1)
        #             seq_lens = masks.sum(-1, keepdim=True)
        #             avg_pooled = hidden_states.sum(1) / seq_lens
        #
        #             # max pooled
        #             # hidden_states = outputs[0]
        #             # hidden_states[masks == 0] = float('-inf')
        #             # max_pooled = hidden_states.max(dim=1)[0]
        #
        #             memmap[i:i+args.batch_size, :] = avg_pooled.detach().cpu().numpy()
        #             i += args.batch_size
        # else:
        #     memmap = np.memmap(f'{args.output_dir}/{split}_vectors.dat', dtype='float32', mode='r', shape=(len(dataset['train']), config.hidden_size))
        #
        # on_mem_data = np.array(memmap)
        # pca_plotting = PCA(n_components=50)
        # plot_data = pca_plotting.fit_transform(on_mem_data)
        # cats = dataset['train']['source']
        #
        #
        # # GMM
        # estimator = GaussianMixture(n_components=2, covariance_type='full', max_iter=150, random_state=args.seed, verbose=True)
        # estimator.fit(plot_data)
        # y_pred = estimator.predict(plot_data)
        # clusters_to_classes = map_clusters_to_classes_by_majority(cats, y_pred)
        # classes_to_cluster = {v:k for k,v in clusters_to_classes.items()}
        #
        # # Get representative points and measure pairwise distances
        # centers = np.empty(shape=(estimator.n_components, plot_data.shape[1]))
        # for i in range(estimator.n_components):
        #     density = scipy.stats.multivariate_normal(cov=estimator.covariances_[i], mean=estimator.means_[i]).logpdf(
        #         plot_data)
        #     centers[i, :] = plot_data[np.argmax(density)]
        #
        # # Calculate purity
        # count = 0
        # for i, pred in enumerate(y_pred):
        #     if clusters_to_classes[pred] == cats[i]:
        #         count += 1
        # train_accuracy = float(count) / len(y_pred) * 100
        # print(f"Purity: {train_accuracy}")
        # # Plot
        # fig = plt.figure(figsize=(10, 8))
        # pca_plotting = PCA(n_components=2)
        # plot_data = pca_plotting.fit_transform(on_mem_data)
        # palette = {k: colors[v] for k,v in classes_to_cluster.items()}
        # scatter = sns.scatterplot(plot_data[:,0], plot_data[:,1], hue=cats, palette=palette, edgecolor='none')
        # scatter.set(xticklabels=[])
        # scatter.set(yticklabels=[])
        # make_ellipses(estimator, scatter, clusters_to_classes)
        # fig.tight_layout()
        # fig.savefig(f'{args.output_dir}/{split}.png')


        #### Lexical analysis
        metrics = {}
        # Reload the data
        dataset = load_dataset('csv', data_files={'train': args.data_loc})
        selector = (np.array(dataset['train']['final_score_hard']) > 3) & np.array(['easy' not in id_ for id_ in dataset['train']['instance_id']])
        if split != 'full':
            source_selector = np.array([s == split for s in dataset['train']['source']])
            selector = selector & source_selector
            corpus_papers = '\n'.join(np.array(dataset['train']['Paper Finding'])[source_selector])
            corpus_news = '\n'.join(np.array(dataset['train']['News Finding'])[source_selector])
        else:
            corpus_papers = '\n'.join(np.array(dataset['train']['Paper Finding']))
            corpus_news = '\n'.join(np.array(dataset['train']['News Finding']))

        lex_papers = LexicalRichness(corpus_papers)
        lex_news = LexicalRichness(corpus_news)

        print(f"Word count (paper, news): ({lex_papers.words},{lex_news.words})")

        print(f"Uniques words (paper, news): ({lex_papers.terms},{lex_news.terms})")

        print(f"Type-token ratio (TTR) (paper, news): ({lex_papers.ttr},{lex_news.ttr})")

        print(f"Root type-token ratio (RTTR) (paper, news): ({lex_papers.rttr},{lex_news.rttr})")

        print(f"Corrected type-token ratio (CTTR) (paper, news): ({lex_papers.cttr},{lex_news.cttr})")

        print(f"Mean segmental type-token ratio (MSTTR) (paper, news): ({lex_papers.msttr(segment_window=25)},{lex_news.msttr(segment_window=25)})")

        print(f"Moving average type-token ratio (MATTR) (paper, news): ({lex_papers.mattr(window_size=25)},{lex_news.mattr(window_size=25)})")

        print(f"Measure of textual lexical diversity (MTLD) (paper, news): ({lex_papers.mtld(threshold=0.72)},{lex_news.mtld(threshold=0.72)})")

        print(f"Hypergeometric distribution diversity (HD-D) (paper, news): ({lex_papers.hdd(draws=42)},{lex_news.hdd(draws=42)})")

        print(f"Herdan's lexical diversity measure (paper, news): ({lex_papers.Herdan},{lex_news.Herdan})")

        print(f"Summer's lexical diversity (paper, news): ({lex_papers.Summer},{lex_news.Summer})")

        print(f"Dugast's lexical diversity (paper, news): ({lex_papers.Dugast},{lex_news.Dugast})")

        print(f"Maas's lexical diversity (paper, news): ({lex_papers.Maas},{lex_news.Maas})")

        metrics['paper.word_count'] = lex_papers.words
        metrics['news.word_count'] = lex_news.words

        metrics['paper.unique_words'] = lex_papers.terms
        metrics['news.unique_words'] = lex_news.terms

        metrics['paper.ttr'] = lex_papers.ttr
        metrics['news.ttr'] = lex_news.ttr

        metrics['paper.rttr'] = lex_papers.rttr
        metrics['news.rttr'] = lex_news.rttr

        metrics['paper.cttr'] = lex_papers.cttr
        metrics['news.cttr'] = lex_news.cttr

        metrics['paper.msttr'] = lex_papers.msttr(segment_window=25)
        metrics['news.msttr'] = lex_news.msttr(segment_window=25)

        metrics['paper.mattr'] = lex_papers.mattr(window_size=25)
        metrics['news.mattr'] = lex_news.mattr(window_size=25)

        metrics['paper.mtld'] = lex_papers.mtld(threshold=0.72)
        metrics['news.mtld'] = lex_news.mtld(threshold=0.72)

        metrics['paper.hdd'] = lex_papers.hdd(draws=42)
        metrics['news.hdd'] = lex_news.hdd(draws=42)

        metrics['paper.Herdan'] = lex_papers.Herdan
        metrics['news.Herdan'] = lex_news.Herdan

        metrics['paper.Summer'] = lex_papers.Summer
        metrics['news.Summer'] = lex_news.Summer

        metrics['paper.Dugast'] = lex_papers.Dugast
        metrics['news.Dugast'] = lex_news.Dugast

        metrics['paper.Maas'] = lex_papers.Maas
        metrics['news.Maas'] = lex_news.Maas


        # Looking at edit distances
        distances = []
        scores = np.array(dataset['train']['final_score_hard'])[selector]
        assert all([s > 3 for s in np.array(dataset['train']['final_score_hard'])[selector]])
        assert len(np.array(dataset['train']['Paper Finding'])[selector]) == len(np.array(dataset['train']['News Finding'])[selector])
        assert len(np.array(dataset['train']['Paper Finding'])[selector]) == sum(selector)
        for paper,news in zip(np.array(dataset['train']['Paper Finding'])[selector],np.array(dataset['train']['News Finding'])[selector]):
            dist = editdistance.eval(paper, news)
            norm_dist = dist / max(len(paper), len(news))
            distances.append(norm_dist)

        print(f"Average edit distance (ours): {sum(distances) / len(distances)}")
        metrics['ours.mean_normalized_edit_distance'] = sum(distances) / len(distances)

        # Compare to STS benchmarks
        stsb = load_dataset("glue", name='stsb', split='train')
        selector = np.array(stsb['label']) > 3
        distances_sts = []
        for sent1, sent2 in zip(np.array(stsb['sentence1'])[selector], np.array(stsb['sentence2'])[selector]):
            dist = editdistance.eval(sent1, sent2)
            norm_dist = dist / max(len(sent1), len(sent2))
            distances_sts.append(norm_dist)

        print(f"Average edit distance (STSB): {sum(distances_sts) / len(distances_sts)}")
        metrics['stsb.mean_normalized_edit_distance'] = sum(distances_sts) / len(distances_sts)

        # And compare to NLI differences
        nli = load_dataset('snli', name='plain_text', split='train')
        selector = np.array(nli['label']) == 0
        distances_nli = []
        for sent1, sent2 in zip(np.array(nli['premise'])[selector], np.array(nli['hypothesis'])[selector]):
            dist = editdistance.eval(sent1, sent2)
            norm_dist = dist / max(len(sent1), len(sent2))
            distances_nli.append(norm_dist)

        print(f"Average edit distance (SNLI): {sum(distances_nli) / len(distances_nli)}")
        metrics['snli.mean_normalized_edit_distance'] = sum(distances_nli) / len(distances_nli)


        if not os.path.exists(f"{args.metrics_dir}"):
            os.makedirs(f"{args.metrics_dir}")
        with open(f"{args.metrics_dir}/{split}_analysis.json", 'wt') as f:
            f.write(json.dumps(metrics))



