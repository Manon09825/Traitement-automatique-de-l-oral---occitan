import torchaudio.transforms as T
from nltk import edit_distance
from tqdm import tqdm
import json

def evaluate_dataset(dataset, X_test, true_phonemes, model, processor, predict_function, clean_function, get_per_function, alignment_function, output_filename):
    true_clean = clean_function(true_phonemes, list_of_lists=True)
    y_true_gascon = [tlist[1] for tlist in true_clean]
    total_pred = []
    output = output_filename

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

    aligned_results = alignment_function(true, pred)
    y_true_aligned, y_pred_aligned = aligned_results['y_true_aligned'], aligned_results['y_pred_aligned']
    y_true_sub_hit, y_pred_sub_hit = aligned_results['y_true_sub_hit'], aligned_results['y_pred_sub_hit']

    # Save aligned predictions and "gold" transcription in json file
    with open(output, "w", encoding="utf-8") as file:
        json.dump("[", file, ensure_ascii=False)
        json.dump(" ".join(y_true_aligned), file, ensure_ascii=False, indent=4)
        json.dump("]", file, ensure_ascii=False)
        json.dump(f"\n", file, ensure_ascii=False)
        json.dump("[", file, ensure_ascii=False)
        json.dump(" ".join(y_pred_aligned), file, ensure_ascii=False, indent=4)
        json.dump("]", file, ensure_ascii=False)


    # compute per
    per = get_per_function(y_true_gascon, total_pred, "test")
    aligned_per = get_per_function(y_true_aligned, y_pred_aligned, "test")
    aligned_per_sub_hit = get_per_function(y_true_sub_hit, y_pred_sub_hit, "test")

    print("true aligned:", y_true_aligned)
    print("pred aligned:", y_pred_aligned)

    print("true without insertions and deletions:", y_true_sub_hit)
    print("pred without insertions and deletions: ", y_pred_sub_hit)

    return {'per': per,
            'aligned_per': aligned_per,
            'aligned_per_sub_hit': aligned_per_sub_hit,
            'total': aligned_results['total_phonemes'],
            'deletions': aligned_results['deletions'],
            'insertions': aligned_results['insertions'],
            'y_true': y_true_aligned,
            'y_pred': y_pred_aligned,
            'y_true_sub_hit': y_true_sub_hit,
            'y_pred_sub_hit': y_pred_sub_hit,     
        }
