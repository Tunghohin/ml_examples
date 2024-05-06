import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

if __name__ == "__main__":
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    y_labels = (y > np.median(y)).astype(int)

    features_train, labels_train, features_test, label_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)