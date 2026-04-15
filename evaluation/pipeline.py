import torch
import torchaudio.transforms as T
from nltk import edit_distance
from tqdm import tqdm

def evaluate_dataset(dataset, true_phonemes, model, processor, predict_function, clean_function, closest_function, get_per_function, X_test):
    total_distance = 0
    total_length = 0

    for i in tqdm(range(len(dataset))):
        audio = dataset["audio"][i]["array"]
        sr = 16000

        audio_path = X_test[i]

        # prediction
        pred = predict_function(model, processor, audio_path)

        # cleaning
        true_clean, pred_clean = clean_function(true_phonemes, list_of_lists=True), clean_function(pred)

        # get closest
        true_variants = true_clean[i]
        true_best = closest_function(pred_clean, true_clean)

        # compute per
        per = get_per_function(true_best, pred_clean, "test")

    return per
