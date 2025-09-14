"""RandomForest trading model implementation.

Provides a concrete classification/regression model using scikit-learn's
RandomForest, built on top of the TradingModel base class.
"""

from typing import Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from loguru import logger

from src.models.base_model import TradingModel


class RandomForestTradingModel(TradingModel):
    """RandomForest-based trading model for classification or regression."""

    def __init__(
        self,
        model_type: str = "classification",
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        super().__init__(
            model_name="RandomForest",
            model_type=model_type,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
        )
        self._rf_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _create_model(self):
        if self.model_type == "classification":
            return RandomForestClassifier(**self._rf_kwargs)
        else:
            return RandomForestRegressor(**self._rf_kwargs)

    def _validate_input(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if y is not None:
            if not isinstance(y, (pd.Series, pd.DataFrame)):
                raise ValueError("y must be a pandas Series or DataFrame")
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of rows")
        if X.isna().all(axis=1).any():
            logger.warning("Some rows in X contain all NaNs; model may underperform. Consider preprocessing.")
