import sys
import subprocess

import importlib

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def tokenize(text):
    '''Tokenize text
    
    Tokenize text with normalization and lemmatization
    
    Parameters
    ----------
    text: str
        Text for tokenization
    
    Return
    ------
    list
        Normalized tokens after lemmatization
    '''
    # Tokenize the text
    tokens = word_tokenize(text)
    
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    # Lemmatize tokens with normalization
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class MyLogisticRegression(LogisticRegression):
    '''Implementation of Logistic Regression with one-class training support
    
    Implement additional checking of unique classes count.
    If label contains one class value, save the class and always predict its values
    '''
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):
        self._single_class_label = None
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                         random_state=random_state, solver=solver, max_iter=max_iter,
                         multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                         l1_ratio=l1_ratio)

    @staticmethod
    def _has_only_one_class(y):
        return len(np.unique(y)) == 1

    def _fitted_on_single_class(self):
        return self._single_class_label is not None

    def fit(self, X, y=None):
        if self._has_only_one_class(y):
            self._single_class_label = y[0]
            self.classes_ = np.unique(y)
        else:
            super().fit(X, y)
        return self

    def predict(self, X):
        if self._fitted_on_single_class():
            return np.full(X.shape[0], self._single_class_label)
        else:
            return super().predict(X)
        
        
def word_cloud(df, top=None, title="", 
               width=400, height=400, 
               column="index", counts="count"): 
    '''Build data for wordcloud plots
    
    Generate data for Vega to render wordcloud plots
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataset with words and its frequencies, 
        sorted by frequency (in descending order)
        
    top: int
        How many top frequent words to display
    
    title: str
        Plot title
        
    width: int
        Plot width (in pixels)
        
    height: int
        Plot height (in pixels)
        
    column: str
        Column name with words
        
    counts: str
        Column name with word frequencies
        
    Return
    ------
    dict:
        Data for rendering using Vega
    '''
    
    if top:
        df = df.iloc[:top].copy()
        
    data= [dict(name="dataset", values=df.to_dict(orient="records"))]
    wordcloud = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": width,
        "height": height,
        "padding": 0,
        "title": dict(text=title, fontSize=18),
        "data": data
    }
    
    scale = dict(
        name="color",
        type="ordinal",
        range=["cadetblue", "royalblue", "steelblue", "navy", "teal"]
    )
    
    mark = {
        "type":"text",
        "from":dict(data="dataset"),
        "encode":dict(
            enter=dict(
                text=dict(field=column),
                align=dict(value="center"),
                baseline=dict(value="alphabetic"),
                fill=dict(scale="color", field=column),
                tooltip=dict(signal="datum.count + ' occurences'")
            )
        ),
        "transform": [{
            "type": "wordcloud",
            "text": dict(field=column),
            "size": [width, height],
            "font": "Helvetica Neue, Arial",
            "fontSize": dict(field="datum.{}".format(counts)),
            "fontSizeRange": [10, 60],
            "padding": 2,        
        }]
    }
    
    wordcloud["scales"] = [scale]
    wordcloud["marks"] = [mark]
    
    return wordcloud 


def get_words_count(df, genre, top):
    '''Calculate word counts from text
    
    Get top frequent words with counts for specific genre
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with column 'message' where texts are stored
    genre: 
        Genre for word counts calculation 
    top:
        Number of top frequent words to calculate
        
    Return
    ------
    pandas.DataFrame
        Dataframe with words and frequencies from the provided messages
    '''
    
    genre_df = df.loc[df['genre'] == genre]
    vec = CountVectorizer(max_features=top, stop_words='english', token_pattern=r'\b[A-Za-z]{3,}\b')
    bow = vec.fit_transform(genre_df['message'])
    
    words_sum = bow.sum(axis=0)
    words_freq = [(word, words_sum[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_count_df = pd.DataFrame(words_freq, columns=['index', 'count'])
    
    return words_count_df

