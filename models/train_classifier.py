import sys
import pickle
from sqlalchemy import create_engine

import pandas as pd
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE=42

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterData', engine)
    
    related_map = {0: 0, 1: 1, 2: 0}
    df['related'] = df['related'].map(related_map).astype('int8')
    
    X = df.loc[:, 'id':'genre']
    Y = df.loc[:, 'related':'direct_report']
    Y = Y.drop('child_alone', axis=1)
    
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(clean_text)
    return tokens

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
        ('clf', MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)))
    ])
    
    return pipe

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred_test = model.predict(X_test)

    average_modes = ['micro', 'macro', 'samples', 'weighted']

    print('Accuracy: ', accuracy_score(Y_test, Y_pred_test), '\n')
    
    for average in average_modes:
        print(average)
        print('F1 score: ', f1_score(Y_test, Y_pred_test, average=average, zero_division=0))
        print('Precision score: ', precision_score(Y_test, Y_pred_test, average=average, zero_division=0))
        print('Recall score: ', recall_score(Y_test, Y_pred_test, average=average, zero_division=0), '\n')
        
    for i, col_name in enumerate(category_names):
        print(col_name)
        print(classification_report(Y_test.iloc[:, i], Y_pred_test[:, i], zero_division=0))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
        
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