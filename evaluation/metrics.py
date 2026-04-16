""" Evaluation metrics for the model's performances in phoneme transcription

Includes:
- A function to compute the PER (Phoneme Error Rate) using nltk's edit_distance
- A function to compute aligned PER using nltk's edit_distance_align
- A function to compute the PFER (Phoneme Feature Error Rate) using panphon
- A function to compute PER by variant of occitan """

import numpy as np
from nltk.metrics.distance import edit_distance, edit_distance_align
import panphon

variants = ['auvergnat', 'gascon', 'limousin', 'languedocien', 'provencal', 'vivaro']


# ======================================================== GET PER ==================================================================

def get_per_nltk(y_true, y_pred):
    """ Computes PER using nltk edit_distance

    Args:
        y_true (list of str): list of phonetic transcriptions by Lo Congres
        y_pred (list of str): list of phonetic transcriptions predicted by the model
        
    Returns:
        dict containing:
            - per (float): the mean PER for all transcriptions
            - scores_per (list): PER score by transcription
            - y_true (list): the list of phonemes given by Lo Congres
            - y_pred (list): the list of phonemes predicted by the model
    """

    scores_per = []
    y_true_phoneme = []
    y_pred_phoneme = []

    for true, pred in zip(y_true, y_pred):
        y_true_phoneme.extend(true)
        y_pred_phoneme.extend(pred)

        score = edit_distance(true, pred, substitution_cost=1)
        scores_per.append(score/max(len(true), len(pred)))

    mean_per = np.mean(scores_per) * 100
    
    return {
        "per": mean_per, # format later with f"{mean_per:.2f}%"
        "scores_per": scores_per,
        "y_true": y_true_phoneme,
        "y_pred": y_pred_phoneme,
    }



# ======================================================== GET ALIGNED PER ===========================================================

def get_aligned_per_nltk(y_true, y_pred):
    """ Compute PER using nltk edit_distance_align to align the phonemes before computing the PER

    Args:
        y_true (list of str): list of phonetic transcriptions by Lo Congres
        y_pred (list of str): list of phonetic transcriptions predicted by the model

    Returns:
        dict containing:
            y_true_aligned (list): list of phonemes given by Lo Congres, aligned with y_pred_aligned
            y_pred_aligned (list): list of phonemes predicted by the model, aligned with y_true_aligned
            per_score (float): total PER score (all errors / all phonemes)
    """

    substitutions = 0
    deletions = 0
    insertions = 0
    hits = 0
    total_phonemes = 0
    y_true_aligned = []
    y_pred_aligned = []

    for true, pred in zip(y_true, y_pred):
        aligned_score = edit_distance_align(true, pred, substitution_cost=1)

        prev_j, prev_k = -1, -1

        total_phonemes += len(true)

        for j, k in aligned_score:
            if j == prev_j:
                insertions += 1
            elif k == prev_k:
                deletions += 1
            else:
                if j < len(true) and k < len(pred):
                    y_true_aligned.append(true[j])
                    y_pred_aligned.append(pred[k])
                    if true[j] == pred[k]:
                        hits += 1
                    else:
                        substitutions += 1
            prev_j, prev_k = j, k

    total_errors = substitutions + deletions + insertions
    per_score = total_errors / total_phonemes
    return {
        "y_true_aligned": y_true_aligned,
        "y_pred_aligned": y_pred_aligned,
        "per": per_score,
        }



# ============================================== GET ALIGNED PER BY VARIANT ========================================================

def get_aligned_per_by_var(true, pred): # True is a dict and pred is a list
    aligned_per = {}
    aligned_data = {}

    for var in variants:
        y_true, y_pred, per = get_aligned_per_nltk(true[var], pred)
        aligned_per[var] = per
        aligned_data[var] = (y_true, y_pred)

    return aligned_per, aligned_data


def get_per_by_variant(y_true, y_pred):
    results = {}
    for var in variants:
        results[var] = get_per_nltk(
            y_true[var],
            y_pred,
        )["per"]

    return results

# ======================================================== GET PFER ==================================================================

def get_pfer_panphon(y_true, y_pred):
    dist = panphon.distance.Distance()

    scores = []

    for true, pred in zip(y_true, y_pred):
        feature_dist = dist.feature_edit_distance(true, pred)

        scores.append(feature_dist)

    score_pfer = np.mean(scores) * 100 

    return score_pfer



def pfer_var(true_pred_dict):
    results = {}
    
    for var in variants:
        pfer = get_pfer_panphon(true_pred_dict[var][0], true_pred_dict[var][1])
        results[var] = pfer
    
    return results