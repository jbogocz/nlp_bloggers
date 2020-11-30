# Operating system
import os
# Regular expression
import re
# Dataframe
import pandas as pd
# Natural language
import nltk
from nltk.corpus import stopwords
# Custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
# Cross-validation
from sklearn.model_selection import train_test_split, GridSearchCV
# Preprocessing
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
# Pipeline
from sklearn.pipeline import Pipeline
# Feature selection
from sklearn.feature_extraction.text import TfidfVectorizer
# Multilabel
from sklearn.multiclass import OneVsRestClassifier
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
# Metrics
from sklearn.metrics import (make_scorer, average_precision_score, f1_score,
                             accuracy_score, recall_score)
# Save model
from joblib import dump

# ------------------------------------------------------------------------------

# Load blogger data
def data_info(path):
    """Load dataset and print basic info.
    """
    # Getting the Data
    df = pd.read_csv(path, compression='zip')
    print(f'File loaded successfully')

    # Print No of rows & columns
    print(f'No of rows: {df.shape[0]}\nNo of columns: {df.shape[1]}')

    # Check the missing data
    print(f'Missing cells: {df.isnull().sum().sum()}')

    return df.iloc[:1000,:]

# ------------------------------------------------------------------------------
# Custom Transformers

class RemoveNonalpha(BaseEstimator, TransformerMixin):
    """Remove all non-alphabet characters from text
    Attributes:
        X (pd.Series): Column with text
    Methods:
        fit(X): Pass
        transform (X): Remove all non-alphabet chars from X
    """

    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series):
        pattern = '[^a-z]+'
        # Remove all non-alphabet chars from string
        X = X.apply(lambda x: re.sub(pattern, ' ', x.lower().strip()))
        return X
