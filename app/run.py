import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from utils.utils import install, MyLogisticRegression
    
import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import altair as alt


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterData', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    alt.renderers.set_embed_options(actions=False)
    alt.data_transformers.disable_max_rows()
    
    genre_classes_df = pd.melt(df,
                               id_vars=['id', 'genre', 'message'], 
                               var_name='class_name', 
                               value_name='has_class')#.query('has_class == 1').copy()
    genre_classes_df['class_name'] = (genre_classes_df['class_name'].str.replace('_', ' ')
                                                                    .str.capitalize())
    genre_classes_df['genre'] = genre_classes_df['genre'].str.capitalize()
    genre_classes_df['message_len'] = genre_classes_df['message'].str.len()
    
    classes_df = genre_classes_df.query('has_class == 1').copy()
    genres_df = genre_classes_df.drop_duplicates('id')
    
    class_selection = alt.selection_multi(fields=['class_name'])
    class_color = alt.condition(class_selection,
                                alt.value('steelblue'),
                                alt.value('lightgray'))

    class_counts_chart = alt.Chart(classes_df).mark_bar().encode(
        x=alt.X('class_count:Q', title='Number Of Occurences'),
        y=alt.Y('class_name:N', sort='-x', title='Class Name'),
        tooltip='class_count:Q',
    ).transform_aggregate(
        class_count='count()',
        groupby=['class_name']
    ).properties(
        width=400
    )

    genre_counts_chart = alt.Chart(genres_df).mark_bar().encode(
        x=alt.X('genre_count:Q', title='Number Of Occurences'),
        y=alt.Y('genre:N', sort='-x', title='Genre Name'),
        tooltip='genre_count:Q',
    ).transform_aggregate(
        genre_count='count()',
        groupby=['genre']
    ).properties(
        width=400
    )

    message_len_chart = alt.Chart(genres_df).mark_bar().encode(
        x=alt.X('mean_message_len:Q', title='Mean Message Length'),
        y=alt.Y('genre:N', sort='-x', title='Genre Name'),
        tooltip=alt.Tooltip('mean_message_len:Q', format='.0f')
    ).transform_aggregate(
        mean_message_len='mean(message_len)',
        groupby=['genre']
    ).properties(
        width=400
    )

    chart = (class_counts_chart).configure_axis(
        labelFontSize=14,
        titleFontSize=14
    )
    chart_json = chart.to_json()
    
    chart_2 = (genre_counts_chart & message_len_chart).configure_axis(
        labelFontSize=14,
        titleFontSize=14
    )
    chart_2_json = chart_2.to_json()
    
    # render web page with plotly graphs
    return render_template('master.html', viz=chart_json, viz_2=chart_2_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()