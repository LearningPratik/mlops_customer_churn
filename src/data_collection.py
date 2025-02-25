import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

import kagglehub

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

data = pd.read_csv(f'{path}/churn.csv')

train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 1)

data_path = Path('data/raw/')
data_path.mkdir(parents=True, exist_ok=True)

train_data.to_csv(Path(data_path, 'train.csv'))
test_data.to_csv(Path(data_path, 'test.csv'))