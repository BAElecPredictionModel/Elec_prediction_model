# Electricity Usage Prediction Model
SEOULTECH Junior year  
Business Analytics Team Project: Electricity Usage Prediction Model

## Project Purpose and Description
We utilize weather data to predict optimal electricity charging demand. This approach aims to minimize losses from self-discharge by charging energy to meet the future demand, resulting in cost savings.

## How to Run an Experiment
### [Executable Files]
- preprocess.py: Data preprocessing
- experiment_basic.py: Basic model experiment
- experiment_clustering.py: Advanced model experiment with clustering

### [Execution Steps]
1. Clone the repository.
2. Navigate to the cloned folder.
3. Run Python files in the terminal:
    - Example: `python experiment_basic.py`
    - Augmented data is generated only in the first experiment; subsequently, stored files are used.

### [Result Verification]
- All results are displayed: clustering results, optimal parameters, feature importances, classification performance, errors.
- Graphs are saved: augmented data, clustering results, usage prediction results.
- Augmented data is saved for consistent experiments: 'data/augmented_data.pickle'.

## Brief Code Description
### Data Preprocessing: preprocess.py
- Overview: This file imports and preprocesses multiple Excel files. The preprocessed data is saved in a CSV file.
- Usage: Execute directly in the terminal.
- Code Description:
    1. Load data files and store them as dataframes.
    2. Vectorize time information and add it as features.
    3. Transform other features into appropriate formats.
    4. Replace missing values with the recent 3-hour average.
    5. Remove unnecessary features.
    6. Resample data at 24-hour intervals.
    7. Add recent usage and moving averages.
    8. Save the preprocessed data to a CSV file.

### Data Augmentation and Processing Module: dataproc.py
- Overview: This module provides functions for data augmentation and easy utilization in experiments.
- Usage: Import the DataProc class in the file.
- Code Description:
    - data_augmentation: Augments existing data and returns a dataframe.
    - augmentation: Utilized in data_augmentation.
    - RFE_featureSelection: Performs feature selection using RFE and returns the selected feature list.
    - view_figure: Generates and saves graphs based on the input type.
    - clf_by_label: Separates X and y by label and returns them in a list.

### Pattern Clustering: cluster_model.py
- Overview: This model clusters each time point based on the features of the input samples.
- Usage: Import the ClusterPattern class in the file.
- Code Description:
    - dim_reduction: Performs dimension reduction for pattern extraction and visualization.
    - clustering: Clusters the input samples and returns labels in dataframe format.

### Label Classification: clf_model.py
- Overview: This model predicts which cluster a given time point belongs to.
- Usage: Import the ClassifyLabel class in the file.
- Code Description:
    - validation: Conducts 5-fold cross-validation for model validation.
    - fit: Trains a classification model to predict labels for a specific time point.
    - predict: Predicts labels, saves them as a pickle file, and returns them.

### Electricity Usage Prediction Model: pred_model.py
- Overview: Train the ML model, predict on test dataset and comfirm the performance by metrics
- Usage: Import the PredictUsage class in the file.
- Code Description:
    - fit: Find the optimal hyper paramter with cross validation
    - predict: Predict with the optimal hyperparameter on test dataset
    - calculate_error: Calculates MAE, RMSE, MAPE errors and returns them in a dictionary.

### Basic Model Experiment: experiment_basic.py
- Overview: File for basic model experiments training on the entire dataset.
- Usage: Run in the terminal.
- Experiment Steps:
    1. Load data: DataProc.
    2. Data augmentation: DataProc.
        - Use stored files for consistency after the first experiment.
    3. Model training: PredictUsage.
    4. Electricity usage prediction: PredictUsage.
    - Visualize and save at intermediate steps: DataProc.

### Advanced Model Experiment: experiment_clustering.py
- Overview: File for advanced model experiments incorporating clustering and classification models.
- Usage: Run in the terminal.
- Experiment Steps:
    1. Load data: DataProc.
    2. Data augmentation: DataProc.
        - Use stored files for consistency after the first experiment.
    3. Dimension reduction: ClusterPattern.
    4. Generate labels through clustering: ClusterPattern.
    5. Train models separately for each label: PredictUsage.
    6. Validate label prediction models: ClassifyLabel.
    7. Train label prediction models: ClassifyLabel.
    8. Predict labels for test data at each time point: ClassifyLabel.
    9. Predict electricity usage using label-specific models: PredictUsage.
    - Visualize and save at intermediate steps: DataProc.

