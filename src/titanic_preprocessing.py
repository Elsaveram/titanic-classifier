import pandas as pd
import numpy as np
import json
import joblib

class TitanicPreprocess:
    def __init__(self, args):
        self.args = args
        
        with open(self.args.config, 'rb') as file:
            self.config = json.load(file)
        
        # Load scaler from config
        self.scaler_path = self.config['scaler']['scaler_path']
        self.scaler = joblib.load(self.scaler_path)

        # Load mappings from config
        self.mappings = self.config['mappings']
        
    def impute_age(self, data, median_age_Pclass1, median_age_Pclass2, median_age_Pclass3):
        """
        Function to impute missing 'Age' values based on 'Pclass'.

        This function imputes missing 'Age' values using the median age
        for each 'Pclass' (passenger class):
        - Pclass 1: Uses median_age_Pclass1
        - Pclass 2: Uses median_age_Pclass2
        - Pclass 3: Uses median_age_Pclass3
        If the 'Age' is not missing, it returns the original value.

        Parameters:
        data (Series): A row of the Titanic dataset with columns 'Age' and 'Pclass'.

        Returns:
        float: The imputed or original 'Age' value.
        """        
        Age = data['Age']
        Pclass = data['Pclass']
        
        if pd.isnull(Age):
            if Pclass == 1:
                return median_age_Pclass1
            elif Pclass == 2:
                return median_age_Pclass2
            else:
                return median_age_Pclass3
        else:
            return Age
        
    def impute_missing_values_titanic(self, data):
        """
        Imputes missing values in 'Age', 'Cabin', and other columns.
        
        - 'Age' is filled using median values by 'Pclass'.
        - 'Cabin' is filled with 'U' for unknown.
        - Any remaining missing values are dropped.

        Parameters:
        data (DataFrame): Titanic dataset.

        Returns:
        DataFrame: Dataset with missing values handled.
        """
        # Calculate median ages by Pclass
        median_age_Pclass1 = data[data['Pclass']==1]['Age'].median()
        median_age_Pclass2 = data[data['Pclass']==2]['Age'].median()
        median_age_Pclass3 = data[data['Pclass']==3]['Age'].median()
                
        data['Age'] = data.apply(lambda row: self.impute_age(row, median_age_Pclass1, median_age_Pclass2, median_age_Pclass3), axis=1)
        data['Cabin'] = data['Cabin'].fillna('U')
        data.dropna(inplace=True)

        return data
    
    def generate_features(self, data):
        """
        Function to generate additional features for the Titanic dataset.
        
        This function creates new features such as 'Title', 'FamilySize', and 'Deck' by transforming 
        existing columns. It extracts 'Title' from 'Name', computes 'FamilySize' using 'SibSp' and 
        'Parch', and extracts the deck level from 'Cabin'. Additionally, unnecessary columns are 
        dropped after feature extraction.
        
        Parameters:
        data (DataFrame): The Titanic dataset.
        
        Returns:
        DataFrame: The modified dataset with newly generated features and redundant columns removed.
        """
        
        # Extract 'Title' from 'Name' column
        data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

        # Create 'FamilySize' from 'SibSp' + 'Parch' + 1 (including the individual)
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

        # Extract the first letter from 'Cabin' column to create 'Deck' feature
        # If 'Cabin' is missing, the deck is assigned as 'U' for unknown
        data['Deck'] = data['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'U')

        # Drop original columns
        data.drop(['SibSp','Parch','Cabin','Name'], axis=1, inplace=True)

        return data
    
    def ordinal_encoding(self, data):
        """
        Encodes 'Title' and 'Deck' columns using predefined mappings and drops the originals.

        Parameters:
        data : pandas.DataFrame
            Input DataFrame with 'Title' and 'Deck' columns.

        Returns:
        pandas.DataFrame
            DataFrame with encoded 'Title' and 'Deck' columns as '_encoded'.
        """

        ordinal_columns = ['Title', 'Deck']

        for column in ordinal_columns:
            data[column + '_encoded'] = data[column].map(lambda x: self.mappings[column].get(x, 0))
            data.drop(column, axis=1, inplace=True)

        return data
    
    def log_transform_fare(self, data):
        """
        Log-transform the 'Fare' column and drop the original.
        
        Parameters:
        data (DataFrame): Titanic dataset.

        Returns:
        DataFrame: Dataset with 'Fare_log' and without 'Fare'.
        """
        data['Fare_log'] = np.log1p(data['Fare'])
        data.drop(['Fare'], axis=1, inplace=True)
        return data
    
    def scale_continuous_features(self, data):
        """
        Scales the 'Age' and 'Fare_log' columns using a pre-fitted scaler.

        Parameters:
        data : pandas.DataFrame
            Input data containing the columns to scale.

        Returns:
        pandas.DataFrame
            DataFrame with scaled 'Age' and 'Fare_log' columns.
        """
        columns_to_scale = self.config['scaler']['columns_to_scale']
        data[columns_to_scale] = self.scaler.transform(data[columns_to_scale])

        return data
    
    def preprocess_data(self, data):
        """
        Preprocesses the Titanic dataset by handling missing values, 
        feature engineering, encoding, log transforming, and scaling features.
        """
        # Drop features that were not used in the final version of the model
        data.drop(['Embarked', 'Ticket'], axis=1, inplace=True)

        # Handle missing values
        data = self.impute_missing_values_titanic(data)

        # Feature engineering
        data = self.generate_features(data)

        # One-hot encoding for categorical variables
        data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

        # Ordinal encoding for ordinal features
        data = self.ordinal_encoding(data)

        # Log transformation for skewed continuous variables
        data = self.log_transform_fare(data)
        
        # Scale continuous features for normalization
        data = self.scale_continuous_features(data)

        return data