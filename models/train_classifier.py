import subprocess
import sys
import nltk
import warnings

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', package])
    
install('scikit-learn')
nltk.download('stopwords')
warnings.filterwarnings("ignore")

import pickle
import joblib
from sqlalchemy import create_engine

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

RANDOM_STATE=42

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterData', engine)
    
    related_map = {0: 0, 1: 1, 2: 0}
    df['related'] = df['related'].map(related_map).astype('int8')
    
    X = df.loc[:, ['message']]
    Y = df.loc[:, 'related':'direct_report']
    
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():    
    text_col = 'message'
    text_pipe = Pipeline([
        ('tfidf', TfidfVectorizer())
    ])

    preproc_pipe = ColumnTransformer([
        ('text', text_pipe, text_col)
    ], n_jobs=-1)

    pipe = Pipeline([
        ('preprocess', preproc_pipe),
        ('clf', MultiOutputClassifier(LinearSVC(random_state=RANDOM_STATE)))
    ])
    
    param_grid = {
        'clf__estimator__C': [0.1], #np.logspace(-2, 0, 3),
        'clf__estimator__loss': ['squared_hinge'], #['hinge', 'squared_hinge'], 
        'clf__estimator__multi_class': ['crammer_singer'], 
        'clf__estimator__class_weight': [None], #[None, 'balanced'],

        'preprocess__text__tfidf__analyzer': ['word'],
        'preprocess__text__tfidf__ngram_range': [(1, 1)], 
        'preprocess__text__tfidf__max_features': [20000], 
        'preprocess__text__tfidf__tokenizer': [tokenize],
        'preprocess__text__tfidf__token_pattern': ['(?u)\\b\\w\\w+\\b'], 
        'preprocess__text__tfidf__stop_words': [stopwords.words()], 
        'preprocess__text__tfidf__min_df': [1], 
        'preprocess__text__tfidf__max_df': [0.7],
        'preprocess__text__tfidf__lowercase': [False],
        'preprocess__text__tfidf__binary': [False], 
        'preprocess__text__tfidf__use_idf': [True],
        'preprocess__text__tfidf__smooth_idf': [True],
        'preprocess__text__tfidf__sublinear_tf': [True],
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(pipe, param_grid, scoring='f1_samples',
                               cv=cv, verbose=0, n_jobs=-1)

    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test, Y_pred_test, zero_division=0))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        joblib.dump(model, f)
        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()