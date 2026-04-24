""" Data loading (sentences, audios, Lo Congres transcriptions) and dataset construction

Includes:
    - get_sentences: retrieves the graphic transcription of each audio from the tsv file provided by CommonVoice with the corpus
    - get_audio_paths: retrieves the path to each audio (separated in train, test and dev)
    - get_phonemes_from_LC: passes the graphic transcription of each audio through Lo Congres's phonetizer to get a baseline
    phonetic transcription
    - get_phonemes_from_json: retrieves Lo Congres's transcriptions, stored in json files (to avoid going through Lo Congres
    every time we need to access the transcriptions)
    - get_ref: separates Lo Congres's transcriptions by variant
    - closest: gets the closest transcription to each of the model's prediction
    - make_dataset: creates dataset using audio files and their transcriptions
"""

import csv
import requests
import json
import os
from datasets import Dataset, Audio
from pathlib import Path
from nltk import edit_distance

base_path = Path("..")
cv_path = base_path / "data.common_voice_ov/"
audio_path = base_path / "data/common_voice_oc/clips/"


# ============================================= GET SENTENCES AND AUDIO PATH FROM TSV ==============================================

def get_sentences_from_tsv(name):
    """ Retrieves sentences from tsv file provided by CommonVoice
    
    Args:
        - name (str): specifies train, test or dev. The tsv already contains a train - test - dev separation, get_sentences
        retrieves only the sentences in the group specified via the name argument
    
    Returns:
        sentences (list of str): a list of sentences (not yet transcribed phonetically)
    """

    sentences = []

    if type(name) is str:
        tsv_path = cv_path / f"{name}.tsv"

        with open (tsv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter = "\t")

            for row in reader:
                sentences.append(row["sentence"])

    elif type(name) is list:
        for n in name:
            tsv_path = cv_path / f"{n}.tsv"

            with open (tsv_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file, delimiter = "\t")

                for row in reader:
                    sentences.append(row["sentence"])

    return sentences



def get_audio_paths(name):
    """ Retrieves the path of audio files
    
    Args:
        - name (str): specifies train, test or dev
        
    Returns:
        - audio (list): a list of paths to the audio files
    """

    tsv_path = cv_path / f"{name}.tsv"
    audio = []

    with open (tsv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter = "\t")

        for row in reader:
            audio_path = os.path.join(base_path, row["path"])
            audio.append((audio_path))
    
    return audio



# ================================================= GET PHONETIC TRANSCRIPTIONS =====================================================

def get_phonemes_from_LC(sentences, name, key):
    """ Gets phonetic transcriptions through Lo Congres's phonetizer and stores them in json files
    
    Args:
        - sentences (list of str): list of sentences (graphic transcriptions)
        - name (str): specifies train, test or dev, used to name the output json file
        - key (str): API key used to access Lo Congres's API
    
    Returns:
        - results (list): a list of lists of phonetic transcriptions - for each sentence, Lo Congres returns a phonetic
        transcription in each of the 6 variants of occitan, so each element of the list returned by this function is a list
        of 6 phonetic transcriptions
    """

    API_URL = "http://api.locongres.org/fonoc.php?"
    output = f"phonemes_{name}.json"
    results = []

    for sentence in sentences:
        response = requests.get(API_URL, params={"key": key, "content": sentence})

        data = response.json()
        variantes_phrase = []

        for i in range(6):
            phonemes = data["query"][i]["pronunciation"]
            phonemes = phonemes.replace("'", "")
            phonemes = phonemes.replace("/", "")
            variantes_phrase.append(phonemes)
        
        results.append(variantes_phrase)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return results



def get_phonemes_from_json(file):
    """ Retrieves Lo Congres's phonetic transcriptions from json files
    
    Args:
        - file (str): path to the json file containing the transcriptions
    
    Returns:
        - y(list): a list of lists of phonetic transcriptions - the same list as get_phonemes_from_LC"""

    y = []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            y.append(item)
    return y



# ============================================ ORGANIZE TRANSCRIPTIONS BY VARIANT ==================================================

def get_ref(y_true):
    """ Separates Lo Congres's transcriptions by variant
    
    Args:
        - y_true (list): list of lists of Lo Congres's 6 phonetic transcriptions for each audio
        
    Returns:
        - dict containing:
            - variant name (str): list of transcriptions only in the target variant
    """

    variants = ['auvergnat', 'gascon', 'limousin', 'languedocien', 'provencal', 'vivaro']

    return {
        var: [t[i] for t in y_true] for i, var in enumerate(variants)
    }


# =================================================== GET CLOSEST VARIANT ==========================================================

def closest(pred, true):
    """ Gets the Lo Congrès transcription closest to the model's prediction
    From the six transcriptions of Lo Congrès, gets the transcription closest to each model prediction

    Args:
        - pred (list): list of phonemes predicted by the model
        - true (list): list of lists of phonemes transcribed by Lo Congrès - each list contains 6 transcriptions, one in each variant of Occitan

    Returns:
        - y_test_closest (list): list of the Lo Congrès transcriptions closest to each of the model's predictions """

    y_test_closest = []

    for p, tlist in zip(pred, true):
        best_dist = 1000

        for t in tlist:
            dist = edit_distance(t, p)
            if dist < best_dist:
                best_dist = dist
                closest_transcription = t

        y_test_closest.append(closest_transcription)

    return y_test_closest


# ======================================================== FORM DATASET ============================================================

def make_dataset(X, y):
    """ Forms dataset with audios and transcriptions
    
    Args:
        - X (list): list of paths to audio files
        - y (list): list of phonetic transcriptions
        
    Returns:
        - dataset of audio files and their transcriptions
    """

    dataset = Dataset.from_dict({"audio": X, "transcriptions": y}).cast_column("audio", Audio(sampling_rate=16000))

    return dataset
