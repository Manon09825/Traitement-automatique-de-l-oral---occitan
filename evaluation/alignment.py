""" Alignment of different phoneme sequences and hidden state vectors

Includes:
    - substitution cost computing to get a personal substitution cost for a pair of phonemes
    - Alignment using nltk's edit_distance function
    - Alignment using PanPhon's feature-based edit_distance function
"""

import panphon.distance
import numpy as np
import torch
from nltk.metrics.distance import edit_distance_align

dst = panphon.distance.Distance()


# ==================================== COMPUTE SUBSTITUTION COST ==========================================

def get_substitution_cost(p1, p2):
    """ Computes a substitution cost based on the PanPhon module's weighted_feature_edit_distance function based on phonetic features
    Returns a personal substitution cost for each pair of phonemes
     
    Args:
        - p1 (str): one of the two phonemes to compare
        - p2 (str): the phoneme to compare with p1
         
    Returns:
        - the distance between the two phonemes based on phonetic features calculated by PanPhon """

    dist = dst.weighted_feature_edit_distance(p1, p2)

    return dist



# =================================== ALIGN TRANSCRIPTIONS ==============================================

def alignment_nltk(true, pred, vec_list=None):
    """ Aligns two phoneme sequences using ntlk's edit_distance function
    
    Args:
        - true (list): sequence of phonemes transcribed by Lo Congres
        - pred (list): sequence of phonemes predicted by the model
        
    Returns:
        - dict containing:
            - y_true_aligned (list): list of Lo Congrès phonemes aligned with the prediction
            - y_pred_aligned (list): list of predicted phonemes aligned with Lo Congrès phonemes
            - hs_aligned (list): list of hidden state vectors aligned with the prediction and Lo Congres phonemes
    """

    aligned_score = edit_distance_align(true, pred, substitution_cost=1)

    y_true_aligned = []
    y_pred_aligned = []
    hs_aligned = []

    for j, k in aligned_score:
        if j < len(true) and k < len(pred):
            y_true_aligned.append(true[j])
            y_pred_aligned.append(pred[k])
            if vec_list is not None:
                v = vec_list[k]
                hs_aligned.append(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
    
    return {
        'y_true_aligned': y_true_aligned,
        'y_pred_aligned': y_pred_aligned,
        'hs_aligned': hs_aligned
    }



def alignment_panphon(true, pred, hs = None, gap_cost=4):
    """ Aligns two phoneme sequences using PanPhon's feature-based distance
    
    Args:
        - true (list): sequence of phonemes transcribed by Lo Congres
        - pred (list): sequence of phonemes predicted by the model
        - hs (list, optional): sequence of hidden state vectors corresponding to the predictions
        - gap_cost (float, optional): gap_cost, set at 4 based on tests on a few sentences but can be changed
        
    Returns:
        - dict containing:
            - y_true_aligned (list): list of Lo Congrès phonemes aligned with the prediction
            - y_pred_aligned (list): list of predicted phonemes aligned with Lo Congrès transcriptions
            - hs_aligned (list): list of hidden state vectors aligned with the prediction and Lo Congrès
    """

    n, m = len(true), len(pred)

    score_matrix = np.zeros((n + 1, m + 1))

     # Matrix initialization
    for i in range(n + 1):
        score_matrix[i][0] = i * gap_cost
    for j in range(m + 1):
        score_matrix[0][j] = j * gap_cost

    # Matrix filling
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = get_substitution_cost(true[i-1], pred[j-1])

            score_matrix[i][j] = min(
                score_matrix[i-1][j-1] + sub_cost,
                score_matrix[i-1][j] + gap_cost,
                score_matrix[i][j-1] + gap_cost
            )

    # Backtracking to get correspondences
    y_true_aligned, y_pred_aligned = [], []
    hs_aligned = [] if hs is not None else None

    i, j = n, m

    # Substitution / hit
    while i > 0 or j > 0:
        sub_cost = get_substitution_cost(true[i-1], pred[j-1]) if (i > 0 and j > 0) else None

        if i > 0 and j > 0 and np.isclose(score_matrix[i][j], score_matrix[i-1][j-1] + sub_cost):
            y_true_aligned.append(true[i-1])
            y_pred_aligned.append(pred[j-1])

            if hs is not None:
                hs_aligned.append(hs[j-1])

            i -= 1
            j -= 1

        # Deletion
        elif i > 0 and np.isclose(score_matrix[i][j], score_matrix[i-1][j] + gap_cost):
            y_true_aligned.append(true[i-1])
            y_pred_aligned.append("-")

            if hs is not None:
                hs_aligned.append(None)

            i -= 1

        # Insertion
        else:
            y_true_aligned.append("-")
            y_pred_aligned.append(pred[j-1])

            if hs is not None:
                hs_aligned.append(hs[j-1])

            j -= 1
    
    return {
        'y_true_aligned': y_true_aligned[::-1],
        'y_pred_aligned': y_pred_aligned[::-1],
        'hs_aligned': hs_aligned[::-1] if hs is not None else None,
    }