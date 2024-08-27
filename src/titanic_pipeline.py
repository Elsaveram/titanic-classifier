import argparse
import json
import pandas as pd
import numpy as np
from titanic_preprocessing import TitanicPreprocess
from titanic_forecast import TitanicForecast


class TitanicPipeline:
    def __init__(self, args):
        self.args = args
        
        with open(args.config, 'rb') as file:
            self.config_file = json.load(file)
        
        self.test_data = None
        self.processed_data = None
            
    def load_tables(self):
        self.test_data = pd.read_csv(self.config_file['data']['test_data_path'])
        return self.test_data
        
    def main(self):
        # Load test data
        titanic_data = self.load_tables()
        print('Loaded data')
        
        # Run preprocessing
        tp =TitanicPreprocess(self.args)
        print('Running TitanicPreprocess')
        self.preprocessed_data = tp.preprocess_data(titanic_data)
        print(self.preprocessed_data)
        self.preprocessed_data.to_csv(self.config_file['data']['processed_data_path'], index=False)
        print(f'Preprocessed data saved to {self.config_file["data"]["processed_data_path"]}')

        #Forecast
        tf =TitanicForecast(self.args)
        print('Running TitanicForecast')
        self.output_table = tf.generate_predictions(self.preprocessed_data)
        self.output_table.to_csv(self.config_file['data']['output_data_path'], index=False)
        print(f'Output table saved to {self.config_file["data"]["output_data_path"]}')
        print(self.output_table)
        
#---------------------------------------------------------------------------------------------        
# MAIN SECTION 
#---------------------------------------------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file") 
    args = parser.parse_args()
    
    titanic = TitanicPipeline(args)
    titanic.main()