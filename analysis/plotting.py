""" Visualizations for performance analysis

Includes:
    - plot_confusion_matrix: confusion matrix plotting
    - plot_hs_tsne: visualizationhidden state 
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd


# ======================================================= CONFUSION MATRIX ========================================================

def plot_confusion_matrix(y_true_aligned, y_pred_aligned, list_of_sent=False):
    """ Plots confusion matrix
    
    Args:
        - y_true_aligned (list): list of phonemes transcribed by Lo Congres, aligned with the model's predictions
        - y_pred_aligned (list): list of phonemes predicted by the model, aligned with Lo Congres's predictions
        
    Returns:
        - confusion matrix comparing y_pred_aligned and y_true_aligned
    """

    labels = []

    if list_of_sent:
        y_true_list, y_pred_list = [], []
        for phrase in y_true_aligned:
            for phoneme in phrase:
                if phoneme not in ["-", ":"]:
                    y_true_list.append(phoneme)
        
        for phrase in y_pred_aligned:
            for phoneme in phrase:
                if phoneme not in ["-", ":"]:
                    y_pred_list.append(phoneme)

    
        for phoneme in y_true_list + y_pred_list:
                if phoneme not in ["-", ":"]:
                    if phoneme not in labels:
                        labels.append(phoneme)

    else:               
        for phoneme in y_true_aligned + y_pred_aligned:
            if phoneme not in labels:
                labels.append(phoneme)

        y_true_list, y_pred_list = y_true_aligned, y_pred_aligned
        
    cm = confusion_matrix(y_true_list, y_pred_list, labels=labels)

    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)

    plt.title(f"Matrice de confusion")
    plt.show()



# ================================================= PLOT HIDDEN STATES VECTORS =====================================================

def plot_hs_tsne(target_phonemes, vec_list, phon_list):
    """ Visualize hidden states using t-SNE
    
    Args:
        - target_phonemes (str): phonemes to be displayed
        - vec_list (list): list of hidden states vectors
        - phon_list (list): list of phonemes corresponding to the list of hidden states
        
    Returns:
        - df_all_phonemes_hidden(pd.DataFrame): dataframe of 2D hidden state projections with labels
    """

    X = np.array(vec_list)
    y = np.array(phon_list)

    X, y = zip(*[xy for xy in zip(X, y) if xy[1] in target_phonemes])
    X = np.array(X)
    y = np.array(y)

    X_2d = TSNE(perplexity=5, random_state = 42).fit_transform(X)

    df_all_phonemes_hidden = pd.DataFrame(X_2d, columns = ['Dimension 1', 'Dimension 2'])
    df_all_phonemes_hidden['Phoneme'] = y

    plt.figure(figsize=(20,20))
    sns.scatterplot(data = df_all_phonemes_hidden, x='Dimension 1', y='Dimension 2', hue='Phoneme')
    plt.title("États cachés pour tout le corpus")
    plt.legend(loc='best')
    plt.show()

    return df_all_phonemes_hidden
