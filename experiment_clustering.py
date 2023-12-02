import pandas as pd
import pickle

from dataproc import DataProc
from cluster_model import ClusterPattern
from clf_model import ClassfyLabel
from pred_model import PredictUsage

# Declare instance for data processing
dp = DataProc()
dp.data

# Dataset that Data augmentation is applied
# X_train, X_test, y_train, y_test = dp.data_augmentation(save=True)
with open('data/augmented_data.pickle', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)
# dp.view_figure(dp.augmented_data, figure_type=0, save=True) 
# dp.view_figure(dp.augmented_data, figure_type=0, save=False) 

# Declare instance for clustering
cp = ClusterPattern(X_train)
patterns = cp.data
patterns

# Dimension reduction
data_arr = cp.dim_reduction('tsne')
data_arr

# Clustering electricity usage patterns
pattern_labels = cp.clustering(data_arr, eps=5)
with open('results/pattern_labels.pickle', 'wb') as file:
    pickle.dump(pattern_labels, file)
print(pattern_labels.value_counts())

# Visualize pattern clusters (arr_umap can be used instead)
df_pattern = pd.DataFrame(cp.arr_tsne, columns=['x', 'y'])
df_pattern['label'] = list(pattern_labels.label)
dp.view_figure(df_pattern, figure_type=1, save=True) 
dp.view_figure(df_pattern, figure_type=1, save=False) 

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
test_patterns = X_test
input = (train_patterns, train_pattern_labels, test_patterns)
cl = ClassfyLabel(input)

# Validate using 5-fold cross-validation
fold_accuracy, mean_accuracy = cl.valdiation()

# Predict pattern labels
cl.fit()
pred_labels = cl.predict()
pred_labels = pd.DataFrame({'label':pred_labels}, index=X_test.index)
pred_labels

# Classify patterns(train data) by label
testset_by_label = dp.clf_by_label(y_test, X_test, pred_labels)

# Predict eletricity usage
result_by_label = []
error_by_label = []
for label in pred_labels.label.unique():
    # Get train data and model of corresponding label
    X_test_, y_test_ = testset_by_label[label]
    model = models[label]
    
    # If data is empty
    if len(X_test_) == 0:
        result_by_label.append("-")
        error_by_label.append("-")
        continue

    # Predict
    y_pred_ = model.predict(X_test_)

    # Make DataFrame
    result = pd.DataFrame({'true': y_test_, 'pred': y_pred_}, index=X_test_.index)
    result_by_label.append(result)

    # Calculate error of each label
    error = model.calculate_error(y_test_, y_pred_)
    error_by_label.append(error)

# Merge prediction results
pred_result = pd.DataFrame()
for result in result_by_label:
    pred_result = pd.concat([pred_result, result])
pred_result.sort_index()
pred_result

# Calcuate error of entire prediction
error = model.calculate_error(pred_result.true, pred_result.pred)
print(error)

# Visualize prediction result
dp.view_figure(pred_result, figure_type=2, save=True)
dp.view_figure(pred_result, figure_type=2, save=False)


