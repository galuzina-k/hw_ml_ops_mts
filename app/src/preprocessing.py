# Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 

# Define column types
target_col = 'binary_target'
categorical_cols = ['частота_пополнения']
continuous_cols = ['сумма', 'секретный_скор', "pack_freq", 'частота', 'доход']
drop_cols = ['client_id', 'mrg_',
            'регион', 'использование', 'on_net',
            'зона_1', 'зона_2', 'pack',
            'сегмент_arpu', 'объем_данных', 'продукт_1', 'продукт_2']

def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=drop_cols)
    print('Data imported...')

    return input_df


# Main preprocessing function
def run_preproc(input_df):

    """All the preprocessing steps are integrated inside the pickle file 
    and are executed while calling predict. No need in this function.
    """

    # Return resulting dataset
    return input_df