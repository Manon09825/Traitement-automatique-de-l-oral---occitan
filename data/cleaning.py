""" Cleaning functions for phonetic transcription strings

Includes:
    - A function to clean text directly (removes punctuation, spaces and normalizes characters)
    - A function to clean lists of transcriptions, which uses the text cleaning function

Note:
    - some phonetic characters are normalized because they aren't used by Lo Congres and leaving them in the model's predictions
    can unfairly penalize it.
"""


# ====================================================== CLEAN STRING ================================================================
def clean_text(text):
    replacements = {
        ",":"",
        ".": "",
        " ": "",
        "?": "",
        "!": "",
        "β": "b",
        "ð": "d" ,
        "ɐ": "a" ,
        "ɾ": "r" ,
        "o": "ɔ" ,
        ":": "" ,
        "æ": "a" ,
        "aw": "aʊ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text



# ====================================================== CLEAN LIST ================================================================

def clean_list(y_list, list_of_lists=False):
    """ Cleans a list of phonetic transcriptions
    
    Args:
        y_list (list): list of phonetic transcriptions - can be a list of str or a list of lists of str
        list_of_lists (boolean): specifies whether the input list is a list of str or a list of lists of str

    Returns:
        clean list of transcriptions
    """
    
    if not list_of_lists:
        return [clean_text(t) for t in y_list]

    else:
        return [[clean_text(t) for t in tlist] for tlist in y_list]
