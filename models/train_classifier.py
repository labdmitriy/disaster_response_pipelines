import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath('..'))
from utils.utils import MyLogisticRegression, tokenize

import joblib
from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

RANDOM_STATE=42


def load_data(database_filepath):
    '''Load data from database
    
    Load dataset from database file and split data to features and targets
    
    Parameters
    ----------
    database_filepath: str
        File path to database
    
    Return
    ------
    X: pandas.DataFrame
        Features data
    Y: pandas.DataFrame
        Targets data
    category_names: list
        Target names
    '''
    # Create connection with database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Read dataset from database table
    df = pd.read_sql_table('DisasterData', engine)
    
    # Split data to features and targets
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':'direct_report']
    
    # Save target names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def build_model():
    '''Create model
    
    Find best model with hyperparameter searching 
    
    Return
    ----------
    sklearn.model_selection.GridSearchCV
        Best found model
    '''
    # Create pipeline for data preprocessing and model training
    pipe = Pipeline([
        ('vec', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(MyLogisticRegression(random_state=RANDOM_STATE), n_jobs=1))
    ])
    
    # Create parameters grid for hyperparameter searching
    # Most of the parameter values were found at earlier separate stage,
    # to reduce the time for additional tuning of the model
    param_grid = {
        'clf__estimator': [MyLogisticRegression(random_state=RANDOM_STATE, solver='liblinear')],
        'clf__estimator__C': np.logspace(-1, 1, 3),
        'clf__estimator__class_weight': [None],

        'vec__analyzer': ['word'],
        'vec__ngram_range': [(1, 1)], 
        'vec__max_features': [None], 
        'vec__tokenizer': [tokenize],
        'vec__token_pattern': ['(?u)\\b\\w\\w+\\b'], 
        'vec__stop_words': [None], 
        'vec__min_df': [1], 
        'vec__max_df': [0.6],
        'vec__lowercase': [False],
        'vec__binary': [True], 
        'vec__use_idf': [False],
        'vec__smooth_idf': [False],
        'vec__sublinear_tf': [False],
    }

    # Defing cross-validation strategy
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # Define grid search process with cross-validation
    grid_search = GridSearchCV(pipe, param_grid, scoring='f1_samples',
                               cv=cv, verbose=1, n_jobs=1)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate model training results
    
    Calculate performance metrics for each label and all data
    
    Parameters
    ----------
    model: sklearn.model_selection.GridSearchCV
        Model for performance evaluation
    X_test: pandas.DataFrame
        Test features data
    Y_test: pandas.DataFrame
        Test targets data
    category_names:
        Target names
    '''   
    # Predict targets on test data
    Y_pred_test = model.predict(X_test)
    
    # Build classification report for precision/recall/f1 metrics
    print(classification_report(Y_test, Y_pred_test,
                                target_names=category_names))


def save_model(model, model_filepath):
    '''Save model
    
    Save model to pickle file
    
    Parameters
    ----------
    model: sklearn.model_selection.GridSearchCV
        Model to save 
    model_filepath: str
        Path to pickle file for model serialization
    '''
    with open(model_filepath, 'wb') as f:
        joblib.dump(model, f)

        
def main():
    # If number of command-line arguments is correct
    if len(sys.argv) == 3:
        # Read all arguments 
        database_filepath, model_filepath = sys.argv[1:]
        
        # Load and split data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
        
        # Build best model after hyperparameter searching
        print('Building model...')
        model = build_model()
        
        # Train the best found model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Check performance of the model using test dataset
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # Save model to pickle format
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