import os
import pandas as pd

dataset_folder = 'E:\\daima\\bioinformatics\\TF\\dataset\\'
for folder_name in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder_name)
    if os.path.isdir(folder_path):
        train_file = os.path.join(folder_path, 'train.data')
        test_file = os.path.join(folder_path, 'test.data')
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            df = pd.read_csv(train_file, sep=' ', header=None, names=['chr_id', 'sequence', 'label'])
            train_csv_file = os.path.join(folder_path, f'train.csv')
            df.to_csv(train_csv_file, index=False)
            df = pd.read_csv(test_file, sep=' ', header=None, names=['chr_id', 'sequence', 'label'])
            test_csv_file = os.path.join(folder_path, f'test.csv')
            df.to_csv(test_csv_file, index=False)