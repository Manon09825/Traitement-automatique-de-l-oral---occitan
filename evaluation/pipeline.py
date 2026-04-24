import torchaudio.transforms as T
from nltk import edit_distance
from tqdm import tqdm

def evaluate_dataset(dataset, X_test, true_phonemes, model, processor, predict_function, clean_function, closest_function, get_per_function, alignment_function):
    true_clean = clean_function(true_phonemes, list_of_lists=True)
    y_true_gascon = [tlist[1] for tlist in true_clean]
    total_pred = []

    for i in tqdm(range(len(dataset))):
        audio = dataset["audio"][i]["array"]
        sr = 16000

        audio_path = X_test[i]

        # prediction
        pred = predict_function(model, processor, audio_path)

        # cleaning
        pred_clean = clean_function(pred)
        total_pred.extend(pred_clean)

    # align
    true = []
    pred = []
    for tlist in y_true_gascon:
        for t in tlist:
            true.extend(t)

    for tlist in total_pred:
        for t in tlist:
            pred.extend(t)

    y_true_aligned, y_pred_aligned = alignment_function(true, pred)['y_true_aligned'], alignment_function(true, pred)['y_pred_aligned']

    # compute per
    per = get_per_function(y_true_gascon, total_pred, "test")
    aligned_per = get_per_function(y_true_aligned, y_pred_aligned, "test")

    return per, aligned_per
