import pandas as pd
import numpy as np

class PredictUsage:
    def __init__(self) -> None:
        pass

    def fit(self, X_train: pd.DataFrame, y_train: np.array) -> None:
        pass

    def predict(self, X_test: pd.DataFrame) -> np.array:
        pass

    # Calculate MAPE
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate RMSE
    def rmse(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate MAE
    def mae(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    # Calcuate error by all methods
    def calculate_error(self, y_true, y_pred):
        mae = self.mae(y_true, y_pred)
        rmse = self.rmse(y_true, y_pred)
        mape = self.mape(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}