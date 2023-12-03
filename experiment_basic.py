import pandas as pd
import pickle

from dataproc import DataProc
from pred_model import PredictUsage

# Declare instance for data processing
dp = DataProc()

# Data augmentation: first run
# X_train, X_test, y_train, y_test = dp.data_augmentation(save=False)
# dp.view_figure(dp.augmented_data, figure_type=0, save=True) 

# Data augmentation: using existing file
with open('data/augmented_data.pickle', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)

# Train prediction model
model = PredictUsage()
model.fit(X_train, y_train)

# Predict eletricity usage
y_pred = model.predict(X_test)

# Make DataFrame
pred_result = pd.DataFrame({'true': y_test, 'pred': y_pred}, index=X_test.index)

# Calculate error of each label
error = model.calculate_error(y_test, y_pred)
print(error)

# Visualize prediction result
dp.view_figure(pred_result, figure_type=2, save=True)