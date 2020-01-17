import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from utils.utils import MyLogisticRegression, word_cloud, get_words_count, tokenize
    
import json
import plotly
import numpy as np
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
import joblib
from sqlalchemy import create_engine

import altair as alt


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterData', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Disable button with saving options
    alt.renderers.set_embed_options(actions=False)
    # Disample max rows limit
    alt.data_transformers.disable_max_rows()
    
    # Prepare data for classes occurences bar plot
    df['message_len'] = df['message'].str.len()
    classes_df = pd.melt(df, id_vars=['id','genre', 'message', 'message_len'], 
                         var_name='class_name', value_name='has_class')
    classes_df = classes_df.query('has_class == 1').copy()
    classes_df['class_name'] = (classes_df['class_name'].str.replace('_', ' ')
                                                                     .str.capitalize())
    classes_df['genre'] = classes_df['genre'].str.capitalize()

    classes_count_df = classes_df['class_name'].value_counts().reset_index()
    classes_count_df.columns = ['class_name', 'class_count']
    
    # Prepare data for genre counts and mean message lengths by genre plots
    genre_stats_df = df.groupby('genre').agg({'id': 'count', 'message_len': 'mean'}).reset_index()
    genre_stats_df['genre'] = genre_stats_df['genre'].str.capitalize()
    genre_stats_df.columns = ['genre', 'genre_count', 'mean_message_len']
    
    # Create chart for classes occurences
    class_counts_chart = alt.Chart(
        classes_count_df,
        title='Number Of Class Occurences'
    ).mark_bar().encode(
        x=alt.X('class_count:Q', title='Number Of Occurences'),
        y=alt.Y('class_name:N', sort='-x', title='Class Name'),
        tooltip='class_count:Q',
    ).configure_title(
        fontSize=18
    ).properties(
        width=400
    )
    
    # Create genre counts chart
    genre_counts_chart = alt.Chart(
        genre_stats_df,
        title='Number Of Class Occurences By Genre'
    ).mark_bar().encode(
        x=alt.X('genre_count:Q', title='Number Of Occurences'),
        y=alt.Y('genre:N', sort='-x', title='Genre Name'),
        tooltip='genre_count:Q',
    ).properties(
        width=400
    )

    # Create mean message length by genre chart
    message_len_chart = alt.Chart(
        genre_stats_df, 
        title='Mean Message Length By Genre'
    ).mark_bar().encode(
        x=alt.X('mean_message_len:Q', title='Mean Message Length'),
        y=alt.Y('genre:N', sort='-x', title='Genre Name'),
        tooltip=alt.Tooltip('mean_message_len:Q', format='.0f')
    ).properties(
        width=400
    )
    
    # Prepare class occurencies charts for plotting
    chart = (class_counts_chart).configure_axis(
        labelFontSize=14,
        titleFontSize=14
    ).configure_title(
        fontSize=18
    )
    chart_json = chart.to_json()
    
    # Prepare genre charts for plotting
    chart_2 = (genre_counts_chart & message_len_chart).configure_axis(
        labelFontSize=14,
        titleFontSize=14
    ).configure_title(
        fontSize=18
    )
    chart_2_json = chart_2.to_json()
    
    # Build worldcloud plots for different genres
    top = 100
    genres_list = df['genre'].unique().tolist()
    word_clouds_list = []

    for genre in genres_list:
        words_count_df = get_words_count(df, genre, top)
        word_clouds_list.append(word_cloud(words_count_df, top=top, 
                                           title=f'Most Popular Words For Genre "{genre.capitalize()}"', 
                                           width=500, height=100))
        
    wc_1_json = json.dumps(word_clouds_list[0])
    wc_2_json = json.dumps(word_clouds_list[1])
    wc_3_json = json.dumps(word_clouds_list[2])
    
    # Render web page with Altair graphs
    return render_template('master.html', viz=chart_json, viz_2=chart_2_json,
                           wc_1=wc_1_json, wc_2=wc_2_json, wc_3=wc_3_json)


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