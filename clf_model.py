import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class ClassfyLabel:
    def __init__(self, input, setting=None):
        # Save input as class variables
        self.X_train_val, self.y_train_val, self.X_test = input

        # Pre-declare the variables for future functions
        self.clf_model = None
        self.y_pred = None

        # Default SVC hyper-parameters: non-linear
        self.default_setting = {'kernel': 'rbf', 'gamma': 0.7}

        # Classification model
        if not setting:
            setting = self.default_setting
        self.clf_model = SVC(**setting, random_state=42)

    # Validate by 5-fold cross-validation
    def valdiation(self):
        model = self.clf_model

        # Declare KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # 5-fold cross-validation
        cv_results = cross_val_score(model, self.X_train_val, self.y_train_val, cv=kf, scoring='accuracy')

        # Accuracy of each fold
        for i, accuracy in enumerate(cv_results, start=1):
            print(f'Fold {i}: Accuracy = {accuracy:.4f}')

        # Result of whole cross-validation
        mean_accuracy = cv_results.mean()
        print(f'Mean Accuracy: {mean_accuracy:.4f}')

        return cv_results, mean_accuracy

    # Fit
    def fit(self) -> None:
        model = self.clf_model

        # Fit model
        model.fit(self.X_train_val, self.y_train_val)

        self.clf_model = model
        return
    
    # Predict
    def predict(self, X) -> np.array:
        model = self.clf_model
        
        # Predict pattern labels
        y_pred = model.predict(self.X_test)

        return y_pred



    