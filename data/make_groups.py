import csv
import os



# =================================== GROUP BY METADATA =========================================

def group(column, group, category, tsv_path, base_clip_path):
    """ Groups audio files by the metadata elements given in the tsv file provided by CommonVoice with the Occitan corpus
    
    Args:
        - column (str): column of the tsv file which specifies the type of metadata (ex "age")
        - group (str): value of the metadata foun in the column in the tsv file (ex "twenties")
        - category (str): name of the category assigned to the audio (ex "Age 20") 
        - tsv_path (str): path to the tsv file
        - base_clip_path (str): path to the audio files
        
    Returns:
        - audios (list): list of paths to audio files
        - sentences (list): list of the written sentences corresponding to each audio file
    """

    audios = []
    sentences = []

    counter = 0

    with open (tsv_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter = "\t")
        for row in reader:
            if group in row[column]:
                counter += 1
                audio_path = os.path.join(base_clip_path, row["path"])
                audios.append(audio_path)
                sentences.append(row["sentence"])

    print(f"There are {counter} audios in the group '{category}'.")
    return audios, sentences



# =================================== GROUP BY SPEAKER =======================================

def group_by(age, genre, tsv_path, base_clip_path):
    """ Groups audio files by age and genre of the speaker
    
    Args:
        - age (str): age group of the speaker as specified in the tsv file (ex: "twenties")
        - genre (str): genre of the speaker as specified in the tsv file (ex: "Female_feminine")
        - tsv_path (str): path to the tsv file
        - base_clip_path (str): path to the audio files
        
    Returns:
        - audios (list): list of paths to audio files
        - sentences: list of sentences corresponding to the audio files

    """

    audios = []
    sentences = []
    counter = 0

    with open(tsv_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in reader:
            if row["age"] == age and row["gender"] == genre:
                counter += 1
                audio_path = os.path.join(base_clip_path, row["path"])
                audios.append(audio_path)
                sentences.append(row["sentence"])

    category_audios = f"x_{age}_{genre}"

    print(f"Nombre d'audios dans le groupe '{category_audios}' : {counter}")
    return audios, sentences



# ======================== SPLIT TRANSCRIPTIONS BY VARIANT ===============================

def split_variantes(transcriptions):
    """ Splits Lo Congrès transcriptions by variant
    For each audio file, 6 transcriptions are given by Lo Congrès (one by variant of Occitan).
    Lo Congrès always return the transcriptions in the same order, which allows us to extract
    each variant from every list of 6 transcriptions.

    Args:
        - transcription: list of lists of transcriptions
        
    Returns:
        - dict containing:
            - var (list): list of Lo Congrès transcriptions for a given variant"""

    variantes = ['auvergnat', 'gascon', 'limousin', 'languedocien', 'provencal', 'vivaro']
    return {
        var: [tlist[i] for tlist in transcriptions]
        for i, var in enumerate(variantes)
    }
