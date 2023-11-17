"""
Challenge: Flights Delays - ML Model to predict the probability of delay for a flight
           taking off or landing at SCL airport.

Description: For this challenge I would choose the model 6.b.i (XGBoost with Feature Importance
and Balance)! So, let's transcribe it to challenge/model.py as well and test it!

Model's code improvements: changed the implementation of `get_min_diff` method from applying to
                           simplifying it by converting with pd.to_datetime the entire column
                           into a datetime format, and then the subtraction operation is
                           performed directly on the columns:
                                - Old `get_min_diff` performance: 4 passed in 10.80s
                                - New `get_min_diff` performance: 4 passed in 1.91s

Author: "Eugenio Grytsenko" <yevgry@gmail.com>
"""

import xgboost as xgb
from typing import Tuple, Union, List

import pandas as pd
import numpy as np


def get_min_diff(target_data):
    date_o = pd.to_datetime(target_data['Fecha-O'])
    date_i = pd.to_datetime(target_data['Fecha-I'])
    min_diff = (date_o - date_i).dt.total_seconds() / 60
    return min_diff


class DelayModel:

    def __init__(self):
        """
        Create the ML model and define top 10 features.

        Args: None

        Returns: None
        """
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self.top_10_features = [
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
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # data['min_diff'] = data.apply(get_min_diff, axis=1)
        data['min_diff'] = get_min_diff(data)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        features = features[self.top_10_features]
        if target_column:
            target = pd.DataFrame(data[target_column])
            return features, target
        else:
            # Last case (test_model_predict) isn't calling `fit` call
            # after `preprocess`, so we do force it here:
            self.fit(features, pd.DataFrame(data['delay']))
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = len(target[target['delay'] == 0])
        n_y1 = len(target[target['delay'] == 1])
        scale = n_y0/n_y1
        self._model.set_params(scale_pos_weight=scale)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        predictions = self._model.predict(features)
        return [int(pred) for pred in predictions]
