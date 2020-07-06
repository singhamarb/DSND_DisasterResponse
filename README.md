# Disaster Response Pipeline Project
### By Amar Singh (Date: 7/7/2020)

## Table of Content:
 - Requirements
 - Instructions
 - Project Overview
 - Licensing, Authors, Acknowledgements


## Requirements 

 - Python 3.5 or higher
 - Kindly see the Requirement.txt file for the required libraries

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Overview:

In this course, I've learned and built on my data engineering skills to expand my opportunities and potential as a data scientist. In this project, i have applied these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data. 


## Project Components
There are three components you'll need to complete for this project.

#### 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

 - Loads the messages and categories datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database

#### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
#### 3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you

## Licensing, Authors, Acknowledgements:

Credit is given to Figure8 for the dataset provided for this project and the udacity also.
