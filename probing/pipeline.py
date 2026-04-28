def prepare_probing(X, y, model, processor, predict_function, cleaning_function, alignment_function, layer):
    """ Gets the data ready for probing

    Args:
        - X (list): list of paths to audio files
        - y (list): list of lists of phonetic transcriptions by Lo Congres
        - model: Wav2Vec2Phoneme, loaded with the model.load_model() function
        - processor: processor loaded with the model.load_model() function'
        - predict_function (func): function used to predict - must return a dict containing at least 'phonemes' and 'last_layer'
        - cleaning_function (func): function used to clean data - must return a list
        - alignment_function (func): function used to align lists - must return a dict containing at least 'y_true_aligned' and 'hs_aligned'
        - layer (str): layer of hidden states - must be either 'last_layer', 'twelfth_layer' or 'first_layer'

    Returns:
        - X_clean (list): clean list of hidden state vectors aligned with the list of phonemes
        - y_clean (list): clean list of Lo Congres' phonemes aligned with the hidden states

    """

    results = {}
    phonemes = []
    hidden_vec = []

    # Getting model predictions
    for i in tqdm(range(len(X))):
        path = X[i]

        pred = predict_function(model, processor, path)
        results[f"audio_{i}"] = pred


    # Counting phonemes and hidden state vectors
    for i in range(len(results)):
        if len(results[f"audio_{i}"]['phonemes']) == len(results[f'audio_{i}'][f'{layer}']):
            continue
        elif len(results[f"audio_{i}"]['phonemes']) != len(results[f'audio_{i}'][f'{layer}']):
            print(f"There are {len(results[f"audio_{i}"]['phonemes'])} phonemes and {len(results[f'audio_{i}'][f'{layer}'])} vectors.")
            break


    # Cleaning
    for i in range(len(results)):
        phon = results[f"audio_{i}"]["phonemes"]
        vec = results[f"audio_{i}"][f'{layer}']

        phonemes.extend(phon)
        hidden_vec.extend(vec)

    y_pred = cleaning_function(phonemes)

    y_gascon = [tlist[1] for tlist in y]
    y_gascon_clean = cleaning_function(y_gascon)

    y_true = []
    for word in y_gascon_clean:
        for p in word:
            y_true.append(p)


    # Aligning true, pred and hidden vectors
    aligned_dict = alignment_function(y_true, y_pred, hidden_vec)
    y_true_aligned = aligned_dict['y_true_aligned']
    hs_aligned = aligned_dict['hs_aligned']


    # Cleaning aligned lists - removing hidden states that are None (and the corresponding phonemes)
    X_clean = []
    y_clean = []

    for h, p in tqdm(zip(hs_aligned, y_true_aligned)):
        if h is not None:
            if p!= "-":
                X_clean.append(h)
                y_clean.append(p)

    if len(X_clean) == len(y_clean):
        return X_clean, y_clean

    else:
        print( f"X and y are not the same length: {len(X_clean)} (X) | {len(y_clean)} (y). Try again before probing.")
        return X_clean, y_clean



def run_probing(X_train, y_train, X_test, y_test, model, processor, predict_function, cleaning_function, alignment_function, run_svm, run_kneighbors, layer):
    """ Probing on the preprocessed data, using an SVM and a KNeighborsClassifier

    Args:
        - X_train (list): list of paths to audio files
        - y_train (list): list of lists of phonetic transcriptions by Lo Congres
        - X_test (list): list of paths to audio files
        - y_test (list): list of lists of phonetic transcriptions by Lo Congres
        - model: Wav2Vec2Phoneme, loaded with the model.load_model() function
        - processor: processor loaded with the model.load_model() function'
        - predict_function (func): function used to predict - must return a dict containing at least 'phonemes' and 'last_layer'
        - cleaning_function (func): function used to clean data - must return a list
        - alignment_function (func): function used to align lists - must return a dict containing at least 'y_true_aligned' and 'hs_aligned'
        - run_svm (func): function to train and test an SVM model - must return a classification report
        - run_kneighbors (func): function to train and test a KNeighborsClassifier - must return a classification report

    Returns:
        - svm_report: the classification report of the test of the SVM model
        - kneighbors_report: the classification report of the test of the KNeighbors model

    """

    X_train_corpus, y_train_corpus = prepare_probing(
        X_train,
        y_train,
        model,
        processor,
        predict_function,
        cleaning_function,
        alignment_function,
        layer,
    )

    X_test_corpus, y_test_corpus = prepare_probing(
        X_test,
        y_test,
        model,
        processor,
        predict_function,
        cleaning_function,
        alignment_function,
        layer,
    )

    svm_report = run_svm(
        X_train_corpus,
        y_train_corpus,
        X_test_corpus,
        y_test_corpus,
    )

    kneighbors_report = run_kneighbors(
        X_train_corpus,
        y_train_corpus,
        X_test_corpus,
        y_test_corpus,
    )

    return svm_report, kneighbors_report
