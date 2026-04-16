""" Probing using an SVM on hidden state representations """

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_svm(X_train, y_train, X_test, y_test):
    """ Trains and evaluates the SVM
    
    Args:
        - X_train (list), y_train (list): training data - list of hidden states and list of corresponding phonemes
        - X_test (list), y_test (list): test data - list of hidden states and list of corresponding phonemes
        
    Returns:
        - classification report
    """

    clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return classification_report(y_test, y_pred)