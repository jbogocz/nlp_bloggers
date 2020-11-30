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


class StopWords(BaseEstimator, TransformerMixin):
    """Remove all english stop words from text
    Attributes:
        X (pd.Series): Column with text
    Methods:
        fit(X): Pass
        transform (X): Remove stop words from X
    """

    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series):
        # Remove all stop words, punctuation
        stop_words = set(stopwords.words('english'))
        X = X.apply(lambda x: ' '.join(
            [words for words in x.split() if words not in stop_words]
        ))
        return X


class Stemming(BaseEstimator, TransformerMixin):
    """Reduce inflection in words to their root forms
    Attributes:
        X (pd.Series): Column with text
    Methods:
        fit (X): Pass
        transform (X): Convert all words from X to the root form 
    """

    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series):
        # Lemmatization/Stemming
        porter_stemmer = nltk.PorterStemmer()
        X = X.apply(lambda x: ' '.join(
            [porter_stemmer.stem(words) for words in x.split()]
        ))
        return X


# ------------------------------------------------------------------------------


class Prepare_X_Y():
    """Prepare X & Y from dataset
    Attributes:
        df (pd.DataFrame): with text, gender, age, sign columns
    Methods:
        get_X (df): Retrun X (text)
        binarize_y (df): Convert gender, age, sign to the one array
                         & encode multiple labels per instance
    """

    def __init__(self, df):
        self.df = df

    def get_X(self):
        # Prepare X (text)
        X = self.df['text']
        return X

    def binarize_y(self):
        # Prepare y (labels) data
        y = self.df.apply(
            lambda x: [x['gender'], str(x['age']), x['sign']], axis=1)
        binarizer = MultiLabelBinarizer()
        y = binarizer.fit_transform(y)
        return y



# ------------------------------------------------------------------------------
# Full Pipeline Classification

# Create scoring for models
# Accuracy
accuracy_score = make_scorer(accuracy_score, greater_is_better=True)
# F1
f1_score = make_scorer(f1_score, average='weighted', greater_is_better=True)

# All scores together
scoring = {
    "acc": accuracy_score,
    "f1": f1_score
}


pipeline = Pipeline([
    ('remove', RemoveNonalpha()),
    ('stopwords', StopWords()),
    ('stem', Stemming()),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('scaler', 'passthrough'),
    ('decomposer', 'passthrough'),
    ('classifier', 'passthrough')
])


# Based on the blogger notebook select only xgb
param_grid = {
    'scaler': ['passthrough', #StandardScaler()
        ],
    'decomposer': ['passthrough', #TruncatedSVD()
        ],
    'classifier': [
                # OneVsRestClassifier(LogisticRegression()),
                # OneVsRestClassifier(RandomForestClassifier()),
                # OneVsRestClassifier(SVC()),
                OneVsRestClassifier(xgb.XGBClassifier())]
}


class ClassificationModel():
    def __init__(self, pipeline, scoring, parameters):
        self.pipeline = pipeline
        self.scoring = scoring
        self.parameters = parameters

    def grid_search(self):
        # Run GridSearchCV
        search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.parameters,
            scoring=self.scoring,
            refit="f1",
            n_jobs=-1,
            cv=5)
        return search

    def model_fit(self, X_train, y_train):
        search = self.grid_search()
        # Fit model
        print('Training in progress.. please wait.')
        model_fit = search.fit(X_train, y_train)
        # Print best parameters
        print(f'Best model parameters:\n{model_fit.best_params_}')
        return model_fit

