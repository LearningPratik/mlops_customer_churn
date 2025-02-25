import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')


# function for training
def training(param_yaml_path):

    # reading from the params.yaml file for that loading it 
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)
    
    # take processed train data from processed directory
    # directory where processed data is saved
    processed_data_dir = Path(params_yaml['process']['dir'])

    # processed training data
    train_file_path = Path(params_yaml['process']['train_file'])

    # path to training data
    train_data_path = processed_data_dir / train_file_path

    
    # setting random state for reproducibility
    random_state = params_yaml['base']['random_state']

    # defining target column, took value from params yaml file
    target = [params_yaml['base']['target_col']]
    
    # reading data
    df = pd.read_csv(train_data_path)
    train = df.rename(columns = str.lower)

    X = train.drop(target, axis=1)
    y = train[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)


    # number estimators parameters for Random Forest
    n_est = 5


    # scikit-learn pipeline
    # Creating a column transformer for preprocessing
    # created list of columns with different number of value_counts()

    preprocessor = ColumnTransformer(
        transformers = [

            ('cat_2', OneHotEncoder(drop = 'first', sparse_output=False), ['gender', 'partner', 'dependents', 'phoneservice', 'paperlessbilling']),
            ('cat_3', OneHotEncoder(drop = 'first', sparse_output=False), ['multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']),
            ('cat_4', OneHotEncoder(drop = 'first', sparse_output=False), ['contract', 'paymentmethod']),
            ('charges', StandardScaler(), ['monthlycharges', 'totalcharges']),
            
        ], remainder = 'passthrough', force_int_remainder_cols = False
        
    )

    # fit and transform the training data
    preprocessor.fit_transform(X_train)

    # Fit and transform the training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Apply same transformation to test data
    X_test_transformed = preprocessor.transform(X_test) 

    # Train Logistic Regression model on TRANSFORMED training data
    model = RandomForestClassifier(n_estimators = n_est)
    model.fit(X_train_transformed, y_train)

    # Make predictions on TRANSFORMED test set
    y_pred_train = model.predict(X_test_transformed)
    accuracy_train = accuracy_score(y_test, y_pred_train)
    classification_report_train = classification_report(y_test, y_pred_train)
    
    
    # Updated run name
    with mlflow.start_run(run_name = "churn_training_with_preprocessing"): 
        print(f"MLflow Run ID: {mlflow.active_run().info.run_uuid}")

        # Log parameters (including preprocessing details)
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("preprocessor_type", "ColumnTransformer")
        mlflow.log_param("encoder", 'OneHotEncoder')
        mlflow.log_param("scaler1", "StandardScaler")
        mlflow.log_param('n_est', 'n_est')

        # Log metrics (same as before)
        mlflow.log_metric("training_accuracy", accuracy_train)
        mlflow.log_param("training_classification_report", classification_report_train)

        # Log the model as MLflow sklearn model
        mlflow.sklearn.log_model(

            # Log the trained RandomForest model
            sk_model = model, 

            # Artifact path name changed
            artifact_path = "churn_rf_pipeline", 
        )
        
        # pkl file name and path
        model_output_path = 'models/rf_model_1.pkl'

        # Save the *entire pipeline* (preprocessor + model) locally as pickle
        pipeline = {'preprocessor': preprocessor, 'model': model} # Create a pipeline dictionary
        with open(model_output_path, 'wb') as model_file_pkl:
            pickle.dump(pipeline, model_file_pkl) # Save the pipeline dictionary

        print(f"Training metrics and model pipeline logged to MLflow. Model pipeline saved to {model_output_path}")


if __name__ == "__main__":

    params_file = 'params.yaml'
    training(params_file)