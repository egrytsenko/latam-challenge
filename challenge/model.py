"""
Challenge: Flights Delays - ML Model to predict the probability of delay for a flight
           taking off or landing at SCL airport.

Description: For this challenge I would choose the model 6.b.i (XGBoost with Feature Importance
and Balance)! So, let's transcribe it to challenge/model.py as well and test it!

Author: "Eugenio Grytsenko" <yevgry@gmail.com>
"""

import xgboost as xgb
from typing import Tuple, Union, List

import pandas as pd
import numpy as np

from datetime import datetime


def get_min_diff(target_data):
    date_o = datetime.strptime(target_data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    date_i = datetime.strptime(target_data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((date_o - date_i).total_seconds())/60
    return min_diff


class DelayModel:

    def __init__(self):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self.FEATURES_COLS = [
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

    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        features = features[self.FEATURES_COLS]
        if target_column:
            target = pd.DataFrame(data[target_column])
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        n_y0 = len(target[target['delay'] == 0])
        n_y1 = len(target[target['delay'] == 1])
        scale = n_y0/n_y1
        self._model.set_params(scale_pos_weight=scale)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        predictions = self._model.predict(features)
        return [int(pred) for pred in predictions]
