# Disaster Response Pipeline Project
## Description
This dataset contains set of messages related to disaster response, covering multiple languages, suitable for text categorization and related natural language processing tasks.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

## Results  
Multilabel classification model was built and used at the backend of the dashboard, where summary information about data is displayed and new text can be entered to be classified.  

![Index Page](images/index_page.png?raw=true "Index Page")
![Go Page](images/go_page.PNG?raw=true "Go Page")

## Deployment Instructions
1. Run the following command to install the latest versions of Scikit-Learn and Altair packages (**required step**):  
    `pip install -U scikit-learn altair`  
    
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app:  
    `python run.py`

4. Go to http://0.0.0.0:3001/

## File structure
- app/
    - Files for frontend: Flask application (run.py), HTML templates (templates/)
- data/
    - Data files (disaster_messages.csv, disaster_categories.csv), database (DisasterResponse.db) and data processing code (process_data.py)
- images/
    - Screenshosts of the dashboard
- models/
    - Saved model (classifier.pkl) and code for model training (train_classifier.py)
- notebooks/
    - Jupyter notebooks with intermediate preparation steps (ETL Pipeline Preparation.ipynb, ML Pipeline Preparation.ipynb) and full EDA, pipeline and visualization
- orig_data/
    - Original data provided by Figure Eight project
- utils/
    - Package with helper functions 
- LICENSE.md
    - License file
- README.md
    - File with repository description
- .gitignore
    - File with git-ignored files/directories 

## License
This project is licensed under the terms of the MIT license

## Acknowledgements
Thanks to [Figure Eight](https://www.figure-eight.com) project for providing original dataset with [Multilingual Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data).
