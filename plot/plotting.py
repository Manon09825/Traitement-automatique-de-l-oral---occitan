""" Plotting data vizualisations

Includes:
    - confusion matrix plotting

"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def plot_confusion_matrix(y_true, y_pred, output_mode='display'):
    """ Plots a confusion matrix
    
    Args:
        - y_true (list): list of phonemes transcribed by Lo Congrès
        - y_pred (list): list of phonemes predicted by the model
        - output_mode (str): specifies whether the confusion matrix should be saved as a png file or displayed when running the script
        
    Returns:
        - 
    """
    true, pred = [], []
    labels = []

    for true_phoneme, pred_phoneme in zip(y_true, y_pred):
        if true_phoneme not in ["-", ":"] and pred_phoneme not in ["-", ":"]:
            true.append(true_phoneme)
            pred.append(pred_phoneme)

    for phoneme in y_true + y_pred:
        if phoneme not in labels:
            labels.append(phoneme)

    fig, ax = plt.subplots(figsize=(20, 20))
    cm = confusion_matrix(true, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)
    plt.title(f"Confusion matrix")

    if output_mode == 'display':
        plt.show()
    
    elif output_mode == 'save':
        name = input("Enter a name for the png file: ")
        plt.savefig(name)