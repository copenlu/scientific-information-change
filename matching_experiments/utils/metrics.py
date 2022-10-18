import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from typing import List, AnyStr, Tuple, Dict
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import ipdb


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return np.sum(preds == labels).astype(np.float32) / float(labels.shape[0])


def acc_f1(averaging, eval_pred) -> Dict:
    logits, labels = eval_pred
    if len(logits.shape) > 1:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits
    acc = accuracy(preds, labels)
    P, R, F1, _ = precision_recall_fscore_support(labels, preds, average=averaging)

    return {'accuracy': acc, 'precision': P, 'recall': R, 'f1': F1}


def compute_regression_metrics(eval_pred, clip_value=(1.0,5.0), prefix=''):
    predictions, labels = eval_pred
    predictions = np.clip(predictions, clip_value[0], clip_value[1])
    mse = mean_squared_error(labels, predictions)
    if len(predictions.shape) > 1:
        predictions = predictions[:,0]
    rho = pearsonr(predictions, labels.squeeze())
    psi = spearmanr(predictions, labels.squeeze())
    return {f"{prefix}mse": mse, f'{prefix}rho': rho[0], f'{prefix}rho-p': rho[1], f'{prefix}psi': psi[0], f'{prefix}psi-p': psi[1]}


def compute_f1(f1_metric, average, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels, average=average)


def compute_rouge(tokenizer, metric, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [lab.strip() for lab in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    #result = {"rouge": result["score"]}
    result = {
        'rouge1_low_p': result['rouge1'].low.precision,
        'rouge1_low_r': result['rouge1'].low.recall,
        'rouge1_low_fmeasure': result['rouge1'].low.fmeasure,
        'rouge1_mid_p': result['rouge1'].mid.precision,
        'rouge1_mid_r': result['rouge1'].mid.recall,
        'rouge1_mid_fmeasure': result['rouge1'].mid.fmeasure,
        'rouge1_high_p': result['rouge1'].high.precision,
        'rouge1_high_r': result['rouge1'].high.recall,
        'rouge1_high_fmeasure': result['rouge1'].high.fmeasure,
        'rouge2_low_p': result['rouge2'].low.precision,
        'rouge2_low_r': result['rouge2'].low.recall,
        'rouge2_low_fmeasure': result['rouge2'].low.fmeasure,
        'rouge2_mid_p': result['rouge2'].mid.precision,
        'rouge2_mid_r': result['rouge2'].mid.recall,
        'rouge2_mid_fmeasure': result['rouge2'].mid.fmeasure,
        'rouge2_high_p': result['rouge2'].high.precision,
        'rouge2_high_r': result['rouge2'].high.recall,
        'rouge2_high_fmeasure': result['rouge2'].high.fmeasure,
        'rougeL_low_p': result['rougeL'].low.precision,
        'rougeL_low_r': result['rougeL'].low.recall,
        'rougeL_low_fmeasure': result['rougeL'].low.fmeasure,
        'rougeL_mid_p': result['rougeL'].mid.precision,
        'rougeL_mid_r': result['rougeL'].mid.recall,
        'rougeL_mid_fmeasure': result['rougeL'].mid.fmeasure,
        'rougeL_high_p': result['rougeL'].high.precision,
        'rougeL_high_r': result['rougeL'].high.recall,
        'rougeL_high_fmeasure': result['rougeL'].high.fmeasure,
        'rougeLsum_low_p': result['rougeLsum'].low.precision,
        'rougeLsum_low_r': result['rougeLsum'].low.recall,
        'rougeLsum_low_fmeasure': result['rougeLsum'].low.fmeasure,
        'rougeLsum_mid_p': result['rougeLsum'].mid.precision,
        'rougeLsum_mid_r': result['rougeLsum'].mid.recall,
        'rougeLsum_mid_fmeasure': result['rougeLsum'].mid.fmeasure,
        'rougeLsum_high_p': result['rougeLsum'].high.precision,
        'rougeLsum_high_r': result['rougeLsum'].high.recall,
        'rougeLsum_high_fmeasure': result['rougeLsum'].high.fmeasure,
    }

    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 6) for k, v in result.items()}
    return result