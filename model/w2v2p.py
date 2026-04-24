""" Model loading and prediction

Includes:
    - model loading from HuggingFace
    - running inference on single audios
    - running inference on dataset
    - extraction of hidden states (hidden representations of the model's predictions)

"""

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import os


# ============================================== LOADING THE MODEL ================================================

def load_model(model_name, access_token=None):
    """ Loads the model from HuggingFace
    
    Args:
        - model_name (str): name of the model to retrieve from HuggingFace
        - access_token (str): optionnal - HF token to enable higher limits and faster download
        
    Returns:
        - processor: the Wav2Vec2Processor
        - model: the Wav2Vec2ForCTC model
    
    """

    processor = Wav2Vec2Processor.from_pretrained(model_name, token=access_token)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True, token=access_token)

    return processor, model



# =============================================== PREDICTIONS =======================================================

def predict_simple(model, processor, audio_path):
    """ Gets the phonemes predicted by the model
    
    Args:
        - model: the model loaded with the load_model function
        - processor: the processor loaded with the load_model function
        - audio_path: path to the audio files
        
    Returns:
        - phonemes: the phonemes predicted by the model
    
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"{audio_path} not found")

    else:
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 16000)

    if audio.ndim == 2:
        audio = audio.mean(dim=0)

    inputs = processor(
    audio.squeeze(),
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred_ids = torch.argmax(logits, dim=-1)
    phonemes = processor.batch_decode(pred_ids)
    transcription = phonemes[0]

    return transcription



def predict(model, processor, audio_path):
    """ Gets the model's prediction and different layers of its hidden states
    
    Args:
        - model: the model loaded with the load_model function
        - processor: the processor loaded with the load_model function
        - audio_path: path to the audio files
        
    Returns:
        - dict containing, for each audio:
            - phonemes: phonemes predicted by the model
            - last_layer: last layer of the model's hidden states corresponding to each phoneme (layer 24)
            - twelfth_layer: twelfth layer of the model's hidden states corresponding to each phoneme
            - first_layer: first layer of the model's hidden states corresponding to each phoneme
    
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"{audio_path} not found")

    else:
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 16000)

    last_layer = []
    twelfth_layer = []
    first_layer = []
    predicted_phonemes = []

    results = {}

    inputs = processor(
    audio.squeeze(),
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        last_hidden_state = outputs.hidden_states[-1]
        twelfth_hidden_state = outputs.hidden_states[12]
        first_hs = outputs.hidden_states[1]

        pred_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(pred_ids)
        transcription = phonemes[0]

    results["y_pred"] = transcription

    for phrase_idx in range(len(pred_ids)):
        for frame_idx in range(len(pred_ids[phrase_idx])):
            if pred_ids[phrase_idx][frame_idx].item() != 0:
                last_layer.append(last_hidden_state[phrase_idx][frame_idx])
                twelfth_layer.append(twelfth_hidden_state[phrase_idx][frame_idx])
                first_layer.append(first_hs[phrase_idx][frame_idx])
                predicted_phonemes.append(processor.tokenizer.convert_ids_to_tokens(pred_ids[phrase_idx][frame_idx].item()))

    results["phonemes"] = predicted_phonemes
    results["last_layer"] = last_layer
    results["twelfth_layer"] = twelfth_layer
    results["first_layer"] = first_layer

    return results



def predict_from_dataset(model, processor, dataset):
    """ Gets the model's prediction and different layers of its hidden states, from a dataset
    
    Args:
        - model: the model loaded with the load_model function
        - processor: the processor loaded with the load_model function
        - dataset: dataset containing audio files and their phonetic transcription by Lo Congres
        
    Returns:
        - dict containing, for each audio:
            - phonemes: phonemes predicted by the model
            - last_layer: last layer of the model's hidden states corresponding to each phoneme (layer 24)
            - twelfth_layer: twelfth layer of the model's hidden states corresponding to each phoneme
            - first_layer: first layer of the model's hidden states corresponding to each phoneme
    
    """

    y_pred = []
    last_layer = []
    twelfth_layer = []
    first_layer = []
    predicted_phonemes = []
    results = {}

    with torch.no_grad():
        for i, phrase in enumerate(dataset["audio"]):
            audio = phrase["array"]

            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            outputs = model(**inputs)
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            twelfth_hidden_state = outputs.hidden_states[12]
            first_hs = outputs.hidden_states[1]

            pred_ids = torch.argmax(logits, dim=-1) 
            phonemes = processor.batch_decode(pred_ids)
            transcription = phonemes[0]

            y_pred.append(transcription)

            for phrase_idx in range(len(pred_ids)):
                for frame_idx in range(len(pred_ids[phrase_idx])):
                    if pred_ids[phrase_idx][frame_idx].item() != 0:
                        last_layer.append(last_hidden_state[phrase_idx][frame_idx])
                        twelfth_layer.append(twelfth_hidden_state[phrase_idx][frame_idx])
                        first_layer.append(first_hs[phrase_idx][frame_idx])
                        predicted_phonemes.append(processor.tokenizer.convert_ids_to_tokens(pred_ids[phrase_idx][frame_idx].item()))
        
        results["y_pred"] = y_pred
        results["predicted_phonemes"] = predicted_phonemes
        results["last_layer"] = last_layer
        results["twelfth_layer"] = twelfth_layer
        results["first_layer"] = first_layer
        

    return results
