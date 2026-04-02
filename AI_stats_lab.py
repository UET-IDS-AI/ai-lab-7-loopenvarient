"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    """
    Implement Naive Bayes spam classification using simple MLE.

    Use the dataset below:

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    Predict the class of:
        test_email = "win cash prize now"

    Returns
    -------
    priors : dict
    word_probs : dict
    prediction : int
    """
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # TODO: tokenize the texts

    # TODO: build vocabulary

    # TODO: compute class priors

    # TODO: compute word probabilities using simple MLE (no smoothing)

    # TODO: predict the class of test_email

    labels= np.array(labels)
    from collections import Counter

    def tokenize(text):
        return text.split()
    
    tokenized_texts = [tokenize(text) for text in texts]
    vocabulary = set(word for text in tokenized_texts for word in text)
    
    priors = {0: np.mean(labels == 0),  # P(ham)
        1: np.mean(labels == 1)}  # P(spam)
    
    word_probs = {0: {}, 1: {}} 
    for word in vocabulary:
        word_probs[0][word] = np.mean([word in text for text, label in zip(tokenized_texts, labels) if label == 0])  # P(word|ham)
        word_probs[1][word] = np.mean([word in text for text, label in zip(tokenized_texts, labels) if label == 1])   # P(word|spam)   
    
    def predict(email):
        tokens = tokenize(email)
        log_prob_ham = np.log(priors[0]) + sum(np.log(max(word_probs[0].get(word, 1e-6), 1e-10)) for word in tokens)
        log_prob_spam = np.log(priors[1]) + sum(np.log(max(word_probs[1].get(word, 1e-6), 1e-10)) for word in tokens)
        return 1 if log_prob_spam > log_prob_ham else 0
    prediction = predict(test_email)
    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    """
    Implement KNN from scratch on the Iris dataset.

    Steps:
    1. Load Iris data
    2. Split into train/test
    3. Compute Euclidean distance
    4. Predict with majority voting
    5. Return train accuracy, test accuracy, and test predictions

    Returns
    -------
    train_accuracy : float
    test_accuracy : float
    predictions : np.ndarray
    """
    # TODO: load iris dataset

    # TODO: split into train and test

    # TODO: implement Euclidean distance

    # TODO: implement prediction using k nearest neighbors

    # TODO: compute train predictions and test predictions

    # TODO: compute accuracies

    dataset= load_iris()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    def knn_predict(X_train, y_train, X_test, k):
        predictions = []
        for test_point in X_test:
            distances = np.array([euclidean_distance(test_point, train_point) for train_point in X_train])
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = y_train[nearest_indices]
            predicted_label = np.bincount(nearest_labels).argmax()
            predictions.append(predicted_label)
        return np.array(predictions)
    test_predictions = knn_predict(X_train, y_train, X_test, k)
    train_predictions = knn_predict(X_train, y_train, X_train, k)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return train_accuracy, test_accuracy, test_predictions

