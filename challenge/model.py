import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from datetime import datetime

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

    @staticmethod
    def _get_min_diff(row):
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data['min_diff'] = data.apply(self._get_min_diff, axis = 1)
        data['delay'] = np.where(data['min_diff'] > self.THRESHOLD_IN_MINUTES, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        features = features[self.TOP_10_FEATURES]
        
        if target_column:
            target = data[[target_column]]  # Double brackets to keep it as DataFrame
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self._model.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        return self._model.predict(features).tolist()
