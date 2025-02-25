import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import (
    LabelEncoder
)

# defining a process function, doing some basic processing for the start
def process(data_path):
    
    # reading the file
    df = pd.read_csv(data_path)

    # skipping first 2 columns, which refers to customerID
    new_df = df.iloc[:, 2:]

    # lowering the column name (not necessary)
    new_df = new_df.rename(columns = str.lower)

    # this column is in object, this is why NaN values are presented as object/string
    # converting column to numeric
    new_df['totalcharges'] = pd.to_numeric(new_df['totalcharges'], errors = 'coerce')

    # droping the NaN values as they are only 8
    new_df = new_df.dropna()
    return new_df

# encoding the dependent column, used labelencoder
def labelencoder(df):

    le = LabelEncoder()
    df['churn'] = le.fit_transform(df['churn'])
    return df


if __name__ == '__main__':

    # path to my params.yaml file
    param_yaml_path = 'params.yaml'

    # opening it with yaml library
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    # Using Path module to specify the path of the specific files
    # data directory --> where the train and test file is
    train_data_path = Path(params_yaml['data_source']['train_data_path'])
    test_data_path = Path(params_yaml['data_source']['test_data_path'])

    # use encode and label_encode function to transform train.csv
    processed_train_data = process(data_path = train_data_path)
    encoded_train_data = labelencoder(processed_train_data)
    
    # save this transformed file to new data/processed directory
    processed_data_dir = Path(params_yaml['process']['dir'])

    # make directory --> processed in data folder if not exist 
    processed_data_dir.mkdir(parents = True, exist_ok = True)
    
    # where to save the processed train file
    processed_train_file_path = Path(params_yaml['process']['train_file'])
    processed_train_data_file = processed_data_dir / processed_train_file_path

    # to_csv the returned dataframe from the function
    encoded_train_data.to_csv(processed_train_data_file, index = False)
    
    # same process for test.csv
    test_data_path = Path(params_yaml['data_source']['test_data_path'])
    processed_test_data = process(data_path = test_data_path)
    encoded_test_data = labelencoder(processed_test_data)

    processed_test_file_path = Path(params_yaml['process']['test_file'])
    processed_test_data_file = processed_data_dir / processed_test_file_path
    encoded_test_data.to_csv(processed_test_data_file, index = False)