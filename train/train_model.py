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
