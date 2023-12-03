# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import xgboost as xgb

class DataProc:
    def __init__(self):
        # Load preprocessed data
        self.data = pd.read_csv('data/preprocessed_data.csv', parse_dates=['Time']).set_index('Time', drop=True)
        
        # 사용량 결측 구간 제외
        self.data = self.data.iloc[15:]
        
        #Hyperparameters
        ##how many years are augmented? This is the hyperparameter
        self.AugTimes = 4 #현재 하이퍼파라미터로는 1+4년치 생성
        self.n_features = [10,20,30,40] #n_feature range for RFE
        
        # Pre-declare the variables for future functions
        self.augmented_data = None
        self.pred_test_idx = None
        
    # Data augmentation
    def data_augmentation(self, save=False) -> pd.DataFrame:
        
        #train_test split first
        self.data_train = self.data.iloc[:365] #1년치 train으로
        self.data_test = self.data.iloc[365:] #6개월치 test으로
        print("self.data_train.shape: ",self.data_train.shape)
        print("self.data_test.shape: ",self.data_test.shape)
        
        #augmentation for traninig data
        Augmentations = [self.data_train]
        for yearPush in range(1,self.AugTimes+1):
            #default: categoricalRandInt=False
            Augmentations.append(self.augmentation(self.data_train,yearPush,categoricalRandInt=False))

        data_train_augmented = pd.concat(Augmentations, axis=0)
        self.augmented_data = data_train_augmented
        print("data_train_augmented.shape: ",data_train_augmented.shape)
        
        #RFE
        SelctedFeaturesSrt = self.RFE_featureSelection(data_train_augmented)
        print("SelctedFeaturesSrt: ",SelctedFeaturesSrt)
        
        #X_train, X_test, y_train, y_test
        X_trainVal = data_train_augmented.drop(["AveragePower"], axis=1)
        X_trainVal = X_trainVal[SelctedFeaturesSrt]
        y_trainVal = data_train_augmented["AveragePower"]
        
        X_test = self.data_test.drop(["AveragePower"], axis=1)
        X_test = X_test[SelctedFeaturesSrt]
        y_test = self.data_test["AveragePower"]
        
        # Save augmented data as csv
        if save:
            with open('data/augmented_data.pickle', 'wb') as file:
                data = X_trainVal, X_test, y_trainVal, y_test
                pickle.dump(data, file)

        return X_trainVal, X_test, y_trainVal, y_test

    # RFE Feature Selection
    def RFE_featureSelection(self,data_train_augmented):
        
        # Assuming 'model' is your pre-defined estimator
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=5)
        
        # and 'X_train', 'y_train' are your training data and labels
        X_train = data_train_augmented.drop(["AveragePower"], axis=1)
        y_train = data_train_augmented["AveragePower"]

        best_score = np.inf  # For MSE, lower is better, so start with infinity
        best_rfe = None
        
        # Define the range for n_features_to_select
        for n_features in self.n_features:
            rfe = RFE(estimator=model, n_features_to_select=n_features)
            rfe.fit(X_train, y_train)

            # Using negative MSE because cross_val_score returns higher values as better and we want lower MSE
            score = -cross_val_score(rfe, X_train, y_train, cv=2, scoring='neg_mean_squared_error').mean()
            
            if score < best_score:
                best_score = score
                optimal_features = n_features
                best_rfe = rfe
        
        SelectedFeatures = best_rfe.ranking_==1
        SelctedFeaturesSrt = list(X_train.columns[SelectedFeatures])

        return SelctedFeaturesSrt
    
    def augmentation(self,df,yearPush,categoricalRandInt=True):
        
        augmented_df = df.loc['2022-3':'2023-02']

        # Change the index by adding 1 year to each date
        print(augmented_df.index)
        augmented_df.index = augmented_df.index + pd.DateOffset(years=yearPush)

        # Now, we will inject noise into the 'augmented_df' based on the mean and standard deviation from the 'df'
        numerical_columns = ["AveragePower","rn","ss","icsr","dsnw","ws","hm","dc10Tca","dc10LmcsCa","vs","ts","wd_x","wd_y","power_yesterday","power_ema4","power_ema7","power_ema14"]

        # Initialize a DataFrame to hold the noise for 'augmented_df'
        noise_df = pd.DataFrame(0,index=augmented_df.index,columns= augmented_df.columns)

        # Generate noise for each numerical feature based on the original 'df'
        for column in numerical_columns:
            # Calculate the mean and standard deviation of the column from the original 'df'
            mean = df[column].mean()
            std_dev = df[column].std()

            # Generate noise with mean = 0 and std = std_dev of the feature for the length of the df
            noise = np.random.normal(0, std_dev*0.2, size=augmented_df[column].shape)

            # Assign the noise to the noise DataFrame
            noise_df[column] = noise

        # Add the generated noise to the 'augmented_df'
        augmented_df = augmented_df + noise_df
        
        #Clipping target to remove minus value 
        augmented_df["AveragePower"] = augmented_df["AveragePower"].clip(lower=0)


        #Augmentation for cloud form

        cloudForm_columns_to_modify = ["ct_Ci","ct_Cc","ct_Cs","ct_Ac","ct_As","ct_Ns","ct_Sc","ct_St","ct_Cu","ct_Cb"]

        
        # Apply the modification to the selected columns
        for column in cloudForm_columns_to_modify:
            # Generate random integers in the range [-1, 1] for each entry
            
            #Noise injection on categorical feature 1)Random Int 2)Float from normal dist
            
            #1)Random Int
            if categoricalRandInt:
                noise = np.random.randint(-1, 1, augmented_df[column].shape)
            
            #2)Float from normal dist
            else:
                std_dev = df[column].std()
                noise = np.random.normal(0, std_dev*0.2, size=augmented_df[column].shape)
            # Add the noise to the column
            augmented_df[column] += noise

            # Replace negative values with 0
            augmented_df[column] = augmented_df[column].clip(lower=0)
        
            
        return augmented_df
    
    # Visualization
    def view_figure(self, data: pd.DataFrame, figure_type, save=False):

        # Show augmented data
        if figure_type == 0:
            plt.figure(figsize=(30, 8))
            data.plot(y='AveragePower', xlabel='Time', ylabel='Usage', title='Augmented Electricity Usage', legend=False)

            # Set the x-axis scale at 1-month intervals
            start_date = data.index[0]
            end_date = data.index[-1]
            plt.xticks(pd.date_range(start=start_date, end=end_date, freq='3MS'), rotation=45, ha='right', fontsize=8)

            # Save figure as png file
            if save:
                plt.savefig(f'results/augmented_data.png', dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.clf()

        # Show pattern cluster
        elif figure_type == 1:
            # 데이터 프레임에서 각 label에 해당하는 데이터를 분리
            unique_labels = data['label'].unique()

            # 각 label에 대해 따로 scatter plot 그리기
            for label in unique_labels:
                subset = data[data['label'] == label]
                plt.scatter(x=subset['x'], y=subset['y'], label=f'Cluster {label}')

            # 그래프 세팅 및 표시
            plt.title('True Electricity Usage Patterns')
            plt.legend()

            # Save figure as png file
            if save:
                plt.savefig('results/pattern_cluster.png', dpi=300, bbox_inches='tight')
            else:
                plt.show()

            plt.clf()
        
        # Show prediction result
        elif figure_type == 2:
            plt.figure(figsize=(20, 8))
            dateIndex = np.array(data.index)
            true = np.array(data.true)
            pred = np.array(data.pred)
            plt.plot(dateIndex, true, label='true')
            plt.plot(dateIndex, pred, label='pred')
            plt.xlabel('Time')
            plt.ylabel('Usage')
            plt.title('Electricity Usage Prediction')
            plt.legend()

            # Set the x-axis scale at 1-month intervals
            start_date = data.index[0]
            end_date = data.index[-1]
            plt.xticks(pd.date_range(start=start_date, end=end_date, freq='MS'), rotation=45, ha='right')

            # Save figure as png file
            if save:
                plt.savefig(f'results/prediction_result.png', dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.clf()

    # Classify data by label
    def clf_by_label(self, y, X, labels: pd.DataFrame):
        data_with_labels = pd.concat([labels, y, X], axis=1)
        data_by_label = []
        for label in range(int(labels.max())+1):
            data = data_with_labels[data_with_labels.label == label]
            y = data.iloc[:, 1]
            X = data.iloc[:, 2:]
            data_by_label.append((X, y))
        
        return data_by_label



