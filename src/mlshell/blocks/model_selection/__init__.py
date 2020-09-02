"""
The :mod:`mlshell.model_selection` contains hyper-parameters tuning utils.
"""

from .prediction import PredictionTransformer, ThresholdClassifier, MockClassifier, MockRegressor
from .search import Optimizer, RandomizedSearchOptimizer, MockOptimizer
from .validation import cross_val_predict, Validator
from .resolve import Resolver

__all__ = ['PredictionTransformer', 'ThresholdClassifier',
           'MockClassifier', 'MockRegressor',
           'Optimizer', 'RandomizedSearchOptimizer', 'MockOptimizer',
           'Validator', 'cross_val_predict', 'Resolver']
