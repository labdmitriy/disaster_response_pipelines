import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    '''Load data from files
    
    Load and merge datasets from specified files
    
    Parameters
    ----------
    messages_filepath: str
        File path for messages data
    categories_filepath: str
        File path for categories data
            
    Return
    -------
    pandas.DataFrame
        Merged dataset with messages and categories info
    '''
    # Load datasets
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df =  messages_df.merge(categories_df, how='inner', on='id')
    
    return df


def clean_data(df):
    '''Clean dataset
    
    Clean, parse and select required data
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataset for cleaning
        
    Return
    ------
    pandas.DataFrame
        Clean dataset
    '''
    # Drop duplicated records
    df = df.drop_duplicates()
    
    # Parse categories column
    parsed_categories = df['categories'].str.split(';|-')
    parsed_categories_df = parsed_categories.str[1::2].apply(pd.Series)
    parsed_categories_df = parsed_categories_df.astype('int8')
    parsed_categories_df.columns = parsed_categories.str[0::2].iloc[0]
    
    # Drop original categories column
    df = df.drop('categories', axis=1)
    
    # Add parsed categories data
    df = pd.concat([df, parsed_categories_df], axis=1)
    
    # Remap incorrect label values for "related" class
    related_map = {0: 0, 1: 1, 2: 0}
    df['related'] = df['related'].map(related_map).astype('int8')

    return df


def save_data(df, database_filename):
    '''Save dataset to database table
    
    Save data to SQLite database. 
    If table exists, it will be replaced with new data.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataset to save
    database_filename: str
        File path to database 
    '''
    # Create connection with database
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save dataset to database table
    df.to_sql('DisasterData', engine, if_exists='replace', index=False)

    
def main():
    # If number of command-line arguments is correct
    if len(sys.argv) == 4:
        # Read all arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        # Load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save data
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