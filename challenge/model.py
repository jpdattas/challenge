import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from joblib import dump, load
import os

# The reasons of the chosen model are descripted in the Jupiter Notebook

class DelayModel:
    TOP_10_FEATURES = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
    THRESHOLD_IN_MINUTES = 15

    def __init__(self):
        self._model = LogisticRegression(class_weight='balanced')

    # This function is used to assist in the creations of delay col
    @staticmethod
    def _get_min_diff(row):
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        
        # One-hot encoding for relevant features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)
        
        # I had to add this part to preprocess data for predictions
        for col in self.TOP_10_FEATURES:
            if col not in features.columns:
                features[col] = False
        
        features = features[self.TOP_10_FEATURES]
        
        if target_column:
            # Create 'delay' col only if needed
            # This way you can preprocess data just to predict without having to pass Fecha-O/Fecha-I cols
            if target_column == 'delay':
                data['min_diff'] = data.apply(self._get_min_diff, axis = 1)
                data['delay'] = np.where(data['min_diff'] > self.THRESHOLD_IN_MINUTES, 1, 0)
            target = data[[target_column]]  # Double brackets to keep it as DataFrame
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self._model.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        return self._model.predict(features).tolist()

    # Adedd these 2 functions to save and load the fitted model
    # Mostly for the API

    def save(self, filename: str):
        # Save just the model part of the instance, not the whole instance
        dump(self._model, filename)

    def load_or_fit(self, filename: str):
        # Check if the file exists
        if os.path.exists(filename):
            # If file exists, load the model from the file
            self._model = load(filename)
            
        else:
            # If it doesn't exist, train it using the provided data and save it
            # Then save it
            data = pd.read_csv('./data/data.csv')
            features, target = self.preprocess(data, target_column='delay')
            self.fit(features, target)
            self.save(filename)