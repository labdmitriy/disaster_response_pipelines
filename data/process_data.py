import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    df =  messages_df.merge(categories_df, how='inner', on='id')
    
    return df

def clean_data(df):
    # parse categories column
    parsed_categories = df['categories'].str.split(';|-')
    parsed_categories_df = parsed_categories.str[1::2].apply(pd.Series)
    parsed_categories_df = parsed_categories_df.astype('int8')
    parsed_categories_df.columns = parsed_categories.str[0::2].iloc[0]
#     parsed_categories_df = parsed_categories_df.drop('child_alone', axis=1)
    
    # drop original categories column
    df = df.drop('categories', axis=1)
    
    # add parsed categories data
    df = pd.concat([df, parsed_categories_df], axis=1)

    return df

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterData', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()