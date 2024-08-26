import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


class TitanicForecast:
    def __init__(self, args):
        self.args = args
        
        with open(self.args.config, 'rb') as file:
            self.config = json.load(file)
        
        # Loading model 
        self.model_path = self.config['model']['best_model_path']
        with open(self.model_path, 'rb') as file:
            self.optimized_model = pickle.load(file)
        
        # Model features and best cutoff
        self.model_features = self.config['features']['model_features']
        self.threshold = self.config['model']['best_cutoff_threshold']

    
    def generate_predictions(self, data):
        """
        Generates predictions using a pre-trained Random Forest model and appends the results to a DataFrame.

        This function loads the pre-trained model, extracts the relevant model features from the input data,
        applies a custom probability threshold for classification, and generates predictions. The results are
        stored in a DataFrame along with 'PassengerId' and the current run date.

        If this function were running in a production environment (e.g., connected to a table in S3 or a database), 
        the results would need to be appended to an existing table rather than overwriting it. In such a scenario, 
        the function could be adapted to upload the results to S3 or a database by connecting to the appropriate 
        data storage service and appending the new predictions.

        Parameters:
        -----------
        data : pandas.DataFrame
            The input DataFrame containing the features used for making predictions.
        
        Returns:
        --------
        pandas.DataFrame
            A DataFrame with three columns:
            - 'PassengerId': The ID of each passenger.
            - 'Prediction': The predicted survival outcome based on the custom threshold.
            - 'RunDate': The date the pipeline was run.
        """
        
        passenger_ids = data['PassengerId']

        # Get the predicted probabilities
        y_pred_prob = self.optimized_model.predict_proba(data[self.model_features])[:, 1]  

        # Apply the custom threshold
        y_pred_custom = (y_pred_prob >= self.threshold).astype(int)

        # Get the current date when the pipeline is run
        current_date = datetime.now().strftime('%Y-%m-%d')

        #Create a DataFrame for the output
        output_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Prediction': y_pred_custom,
        'RunDate': current_date
        })

        return output_df
