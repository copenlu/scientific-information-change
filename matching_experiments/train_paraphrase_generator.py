import argparse
import random
from functools import partial

import numpy as np
import torch
import json
import os
import ipdb
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

import wandb
from utils.metrics import compute_rouge
from rouge_score import rouge_scorer
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


def preprocess_data(tokenizer, examples):
    inputs = [c for c in examples['Paper Finding']]
    targets = [c for c in examples['News Finding']]
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=tokenizer.model_max_length, truncation=True)

    # # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # # padding in the loss.
    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #     labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #     ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        help="The location of the training data",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='facebook/bart-base')
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

    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    data_loc = args.data_loc
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
        'seed': args.seed
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

    # Create the tokenizer and model
    tk = AutoTokenizer.from_pretrained(model_name)
    # Get the dataset
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_dataset('csv', data_files={'train': f"{data_loc}/train.csv",
                                              'validation': f"{data_loc}/dev.csv",
                                              'test': f"{data_loc}/test.csv"})
    dataset = dataset.filter(lambda example: example[LABEL_COLUMN] > 3 and example['source'] == 'news')
    dataset.cleanup_cache_files()
    dataset = dataset.map(partial(preprocess_data, tk), batched=True,
                          remove_columns=dataset['train'].column_names)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tk,
        model=model,
        label_pad_token_id=-100,
        padding='longest'
    )

    # Get the F1 metric
    metric = load_metric('rouge')
    compute_metric = partial(compute_rouge, tk, metric)

    # Create the training arguments
    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        report_to=['tensorboard', 'wandb']
    )

    # Create the trainer and train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metric,
        data_collator=collator
    )

    train_output = trainer.train()
    pred_output = trainer.predict(dataset['test'])
    pred_metrics = pred_output.metrics
    wandb.log(pred_metrics)
    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(pred_metrics))

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    error_samples = []
    for i,preds in enumerate(pred_output.predictions):
        pred_text = tk.decode(preds)
        orig_text = tk.decode(dataset['test']['labels'][i])
        score = scorer.score(orig_text, pred_text)['rougeL'].fmeasure
        error_samples.append([orig_text, pred_text, score])

    error_data = pd.DataFrame(sorted(error_samples, key=lambda x: x[-1], reverse=True), columns=['Original', 'Generated', 'rL-F'])
    error_data.to_csv(f"{args.metrics_dir}/{seed}_errors.csv", index=None)
