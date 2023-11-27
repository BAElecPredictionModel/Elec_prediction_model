import pandas as pd
import numpy as np

from dataproc import DataProc
from cluster_model import ClusterPattern
from clf_model import ClassfyLabel
from pred_model import PredictUsage

# Declare instance for data processing
dp = DataProc()
dp.data

#################################################################################################

# Dataset that Data augmentation is applied
X_train, X_test, y_train, y_test = dp.data_augmentation(save=False)

dp.view_figure(dp.augmented_data, figure_type=0, save=True) 


#################################################################################################

# Declare instance for clustering
cp = ClusterPattern(X_train)
patterns = cp.data
patterns

# Dimension reduction
data_arr = cp.dim_reduction('tsne')
data_arr

# Clustering electricity usage patterns
pattern_labels = cp.clustering(data_arr, eps=10)
pattern_labels

# Visualize pattern clusters (arr_umap can be used instead)
df_pattern = pd.DataFrame(cp.arr_tsne, columns=['x', 'y'])
df_pattern['label'] = pattern_labels
dp.view_figure(df_pattern, figure_type=1, save=True) 

# Classify patterns(train data) by label
trainset_by_label = dp.clf_by_label(y_train, X_train, pattern_labels)

# Train each label's prediction model
models = []
for data in trainset_by_label:
    y_train_, X_train_  = data[1], data[0]
    model = PredictUsage()
    model.fit(X_train_, y_train_)
    models.append(model)

# Declare instance for pattern classfication
train_patterns = patterns
train_pattern_labels = pattern_labels
test_patterns = X_test.iloc[:, 1:]
input = (train_patterns, train_pattern_labels, test_patterns)
cl = ClassfyLabel(input)

# Validate using 5-fold cross-validation
fold_accuracy, mean_accuracy = cl.valdiation()
print("Accuracy: " + str(mean_accuracy))

# Predict pattern labels
cl.fit()
pred_labels = cl.predict()
pred_labels

# Classify patterns(train data) by label
testset_by_label = dp.clf_by_label(y_test, X_test, pred_labels)

# Predict eletricity usage
result_by_label = []
error_by_label = []
for label in range(cp.num_of_labels):
    # Get train data and model of corresponding label
    X_test_, y_test_ = testset_by_label[label]
    model = models[label]

    # Predict
    y_pred_ = model.predict(X_test_)

    # Make DataFrame
    result_by_label = pd.DataFrame({'true': y_test_, 'pred': y_pred_}, index=X_test_.index)
    result_by_label.append(result_by_label)

    # Calculate error of each label
    error = model.error(y_test_, y_pred_)
    error_by_label.append(error)

# Merge prediction results
pred_result = pd.DataFrame()
for result in result_by_label:
    pred_result = pd.concat([pred_result, result])
pred_result.sort_index()
pred_result

# Calcuate error of entire prediction
error = model.error(pred_result.true, pred_result.pred)
error

# Visualize prediction result
dp.view_figure(pred_result, figure_type=2, save=True)

