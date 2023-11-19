import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

class DataProc:
    def __init__(self):
        # Load preprocessed data
        self.data = pd.read_csv('BA/preprocessed_data.csv').set_index('Time', drop=True)
        
        # Pre-declare the variables for future functions
        self.augmented_data = None
        self.pred_test_idx = None
        
    # Data augmentation
    def data_augmentation(self, save=False) -> pd.DataFrame:
        # Augment data
        augmented_data = self.data
        self.augmented_data = augmented_data
        # 여기에 데이터 증강 구현

        # Save test data start point
        self.pred_test_idx = 0 # 테스트 데이터 시작 인덱스 저장

        # Save augmented data as csv
        if save:
            augmented_data.to_csv('BA/results/augmented_data.csv')

        return augmented_data

    # Data split
    def data_split(self, data: pd.DataFrame, phase, test_size=0.2):
        X, y = data.iloc[:, 1:], data.iloc[:, 0]

        # Phase: Split for electricity prediction
        if phase == 'pred':
            X_train, X_test = X.iloc[:self.pred_test_idx], X.iloc[self.pred_test_idx:]
            y_train, y_test = y.iloc[:self.pred_test_idx], y.iloc[self.pred_test_idx:]
            return X_train, X_test, y_train, y_test
        
        # Phase: Split for electricity prediction
        elif phase == 'clf':
            return train_test_split(X, y, test_size=test_size, random_state=42)
        
    # Visualization
    def view_figure(self, data: pd.DataFrame, figure_type, save=False):

        # Show augmented data
        if figure_type == 0:
            plt.figure(figsize=(10, 6))
            data.plot(y='AveragePower', xlabel='Time', ylabel='Usage', title='전기 사용량', legend=False)

            # Set the x-axis scale at 1-month intervals
            start_date = data.index[0]
            end_date = data.index[-1]
            plt.xticks(pd.date_range(start=start_date, end=end_date, freq='MS'), rotation=45, ha='right')

            # Save figure as png file
            if save:
                plt.savefig(f'results/augmented_data.png', dpi=300, bbox_inches='tight')

            plt.show()

        # Show pattern cluster
        elif figure_type == 1:
            sns.scatterplot(x='x', y='y', hue='label', data=data, palette='Set2', legend='full')
            plt.title('True Electricity Usage Patterns')
            plt.legend()

            # Save figure as png file
            if save:
                plt.savefig('results/pattern_cluster.png', dpi=300, bbox_inches='tight')

            plt.show()
        
        # Show prediction result
        elif figure_type == 2:
            plt.figure(figsize=(10, 6))
            plt.xlabel('Time')
            plt.ylabel('Usage')
            plt.title('Electricity Usage Prediction')
            plt.legend()
            plt.plot(data.index, data.true, label='true')
            plt.plot(data.index, data.pred, label='pred')

            # Set the x-axis scale at 1-month intervals
            start_date = data.index[0]
            end_date = data.index[-1]
            plt.xticks(pd.date_range(start=start_date, end=end_date, freq='MS'), rotation=45, ha='right')

            # Save figure as png file
            if save:
                plt.savefig(f'results/prediction_result.png', dpi=300, bbox_inches='tight')

            plt.show()

    # Classify data by label
    def clf_by_label(self, y, X, labels: pd.DataFrame):
        data_with_labels = pd.concat([labels, y, X], axis=1)
        data_by_label = []
        for label in range(labels.max()):
            data = data_with_labels[data_with_labels.label == label]
            y = data.iloc[:, 1]
            X = data.iloc[:, 2:]
            data_by_label.append((X, y))
        
        return data_by_label



