# test_titanic_preprocessing.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from titanic_preprocessing import TitanicPreprocess
import pytest
import pandas as pd
import argparse


@pytest.fixture
def sample_data():
    # Create a small DataFrame as a sample for testing
    data = pd.DataFrame({
        'Pclass': [1, 3, 2],
        'Age': [22, None, 35],
        'SibSp': [1, 0, 1],
        'Parch': [0, 0, 0],
        'Fare': [7.25, 8.05, 26.55],
        'Cabin': [None, 'C123', 'E46'],
        'Name': ['Braund, Mr. Owen Harris', 'Allen, Mr. William Henry', 'Hewlett, Mrs. (Mary D Kingcome)'],
        'Sex': ['male', 'male', 'female']
    })
    return data

def test_impute_missing_values_titanic(sample_data):
    # Get the absolute path of the config file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/titanic_config.json'))

    # Create a mock args structure with config path
    args = argparse.Namespace(config=config_path)

    # Initialize TitanicPreprocess with the mocked args
    tp = TitanicPreprocess(args=args)

    # Run the imputation function
    result = tp.impute_missing_values_titanic(sample_data)

    # Add assertions based on the expected output
    assert result['Age'].isnull().sum() == 0  
    assert result['Cabin'].isnull().sum() == 0  

def test_generate_features(sample_data):
    # Get the absolute path of the config file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/titanic_config.json'))

    # Create a mock args structure with config path
    args = argparse.Namespace(config=config_path)

    # Initialize TitanicPreprocess with the mocked args
    tp = TitanicPreprocess(args=args)

    # Run the feature generation function
    result = tp.generate_features(sample_data)

    # Add assertions based on the expected output
    assert 'Title' in result.columns
    assert 'FamilySize' in result.columns