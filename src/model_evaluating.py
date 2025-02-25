import pandas as pd
import yaml
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow

def evaluate(param_yaml_path):
    with open(param_yaml_path) as f:
        param_file = yaml.safe_load(f)

    # Taking the preocessed test data for evaluating
    processed_data_dir = Path(param_file['process']['dir'])
    
    # path to test data file
    processed_test_data_path = Path(param_file['process']['test_file'])
    test_data_path = processed_data_dir / processed_test_data_path

    # reading the file
    test = pd.read_csv(test_data_path)
    
    # defining target column
    target = [param_file['base']['target_col']]

    # the model which we trained on train data
    model_path = param_file['models']['model_dir']

    # Defining X and y from the passed data parameter in the evaluate function
    X = test.drop(target, axis = 1)
    y = test[target]

    _, X_test, _, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

     # Load the entire pipeline (preprocessor + model) from model.pkl
    with open(model_path, 'rb') as model_file_pkl:

        # Load the pipeline dictionary
        pipeline = pickle.load(model_file_pkl) 

        # Extract preprocessor
        preprocessor = pipeline['preprocessor'] 

        # Extract model
        model = pipeline['model'] 

    # Transform the evaluation data using the loaded preprocessor
    X_test_transformed = preprocessor.transform(X_test)

    # Make predictions using the model from the pipeline on TRANSFORMED test data
    y_pred = model.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # writing metrics into this file, definig the path
    metrics_output_path = 'metrics/metrics.txt'

    # Use MLflow Tracking (rest remains similar)
    # Updated run name
    with mlflow.start_run(run_name="churn_evaluation_with_preprocessing"): 
        print(f"MLflow Run ID: {mlflow.active_run().info.run_uuid}")

        mlflow.log_param("model_path", model_path)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        print(f"Evaluation metrics logged to MLflow. Metrics saved to {metrics_output_path}")

    # writing metrics to file (same)
    with open(metrics_output_path, 'w') as metrics_file:
        metrics_file.write(f"Accuracy: {accuracy:.4f}\n")
        metrics_file.write(f"Precision: {precision:.4f}\n")
        metrics_file.write(f"Recall: {recall:.4f}\n")
        metrics_file.write(f"f1-score: {f1:.4f}")


if __name__ == "__main__":

    param_yaml_path = 'params.yaml'
    evaluate(param_yaml_path)