import joblib

import pandas as pd
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Make prediction
def make_pred(df, path_to_file):

    print('Importing pretrained model...')
    # Load the saved model
    model = joblib.load('models/xgboost_pipeline.pkl')

    # Make predictions
    predictions = model.predict_proba(df)

    # Define optimal threshold
    model_th = 0.6

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (predictions[:, 1] > model_th) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return submission