import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PredictUsage:
    def __init__(self) -> None:
        self.model = None
        
        #Hyperparameters to be searched
        self.n_splits= 5
        self.n_estimators_list=[100,200,300]
        self.max_depths = [5,10,15,20,25,30]
        self.learning_rates= [0.1,0.05,0.01]

    def fit(self, X_trainVal: pd.DataFrame, y_trainVal: np.array) -> None:
        
        cv = KFold(n_splits = self.n_splits, shuffle =True, random_state = 42)
        scores = []
        rownames = []
        

        hyperparamSetting = []
        for n_estimators in self.n_estimators_list:
            for max_depth in self.max_depths:
                for learning_rate in self.learning_rates:
                    hyperparamSetting.append((n_estimators,max_depth,learning_rate))

        for tup in hyperparamSetting:
            n_estimators,max_depth,learning_rate = tup

            mean_MSE = 0
            mean_MAE = 0
            mean_R2 = 0
            mean_MAPE = 0
            for train_idx, val_dix in cv.split(X_trainVal,y_trainVal):
                X_train = X_trainVal.iloc[train_idx,:]
                X_val =X_trainVal.iloc[val_dix, : ]
                y_train = y_trainVal.iloc[train_idx]
                y_val = y_trainVal.iloc[val_dix]


                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                model.fit(X_train,y_train)

                y_val_hat = model.predict(X_val)

                mean_MSE = mean_MSE + mean_squared_error(y_val, y_val_hat)
                mean_MAE = mean_MAE + mean_absolute_error(y_val, y_val_hat)
                mean_R2 = mean_R2 + r2_score(y_val, y_val_hat)
                mape = np.abs((y_val - y_val_hat) / y_val) * 100
                mape[np.isinf(mape)] = 0  # Replace infinite values with 0
                mean_MAPE = mean_MAPE + np.mean(mape)

            scores.append((mean_MSE/self.n_splits, mean_MAE/self.n_splits, mean_R2/self.n_splits, mean_MAPE/self.n_splits))
        
        rownames.append(tup)
        colnames = ['MSE',"MAE","R2","MAPE"]

        df_summary = pd.DataFrame(scores, index=rownames, columns=colnames)
        opt_n_estimators,opt_max_depth,opt_learning_rate = df_summary['MAPE'].idxmin()
        
        #MAPE를 기준으로 optimal hyperparameter 결정
        print("opt_n_estimators,opt_max_depth,opt_learning_rate: ",df_summary['MAPE'].idxmin())
        
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=opt_n_estimators, learning_rate=opt_learning_rate, max_depth=opt_max_depth)
        
        #fitting the model with optimal hyperparameter
        self.model.fit(X_trainVal,y_trainVal)
        
    def predict(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict(X_test)

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