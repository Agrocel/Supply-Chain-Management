import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import os
import json
from datetime import datetime

# Adjust path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Logging.logger import get_logger
from Source.Database.db_con import get_engine

# ----------------------------Configuration of files--------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class DataLoader:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.logger = get_logger("Load_data_set_files")
        self.engine = get_engine()
        self.required_columns = [
            'Billing_Date', 'Sold_To_Party_Name', 'Invoice_Value', 'Plant_Code',
            'Mat_Desc', 'Inv_Qty', 'Inv_Qty_UOM'
        ]

    def _load_config(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _fetch_data(self):
        """Fetches raw data from the database."""
        self.logger.info("Loading Data from database ...........")
        try:
            # Using the centralized engine from db_con
            df = pd.read_sql("SELECT * FROM raw_data", con=self.engine)
            self.logger.info("raw_data loaded from database\n")
            self.logger.info(f"Head of raw_data:\n{df.head()}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from database: {e}", exc_info=True)
            raise

    def _validate_and_filter_columns(self, df):
        """Selects required columns and checks for missing ones."""
        self.logger.info("Validating and selecting required columns...")
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            self.logger.warning(f"Missing columns in the dataset: {missing}")
            raise ValueError(f'Missing required columns: {missing}')
        
        df_selected = df[self.required_columns].copy()
        self.logger.info("Columns validated and selected.\n")
        return df_selected

    def _process_dates(self, df):
        """Converts Billing_Date to datetime and handles errors."""
        self.logger.info("Processing date format...")
        initial_count = df.shape[0]
        
        df = df.copy()
        df['Billing_Date'] = pd.to_datetime(df['Billing_Date'], errors='coerce', dayfirst=True)
        
        df_clean = df.dropna(subset=['Billing_Date'])
        final_count = df_clean.shape[0]
        
        self.logger.info(f'Number of rows in data: {initial_count}')
        self.logger.info(f'Number of rows dropped due to invalid dates: {initial_count - final_count}\n')
        
        return df_clean

    def _save_data(self, df):
        """Saves the processed dataframe to CSV."""
        output_path = os.path.join(PROJECT_ROOT, 'Data', 'Processed', 'data_load_raw_data.csv')
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Data saved successfully to {output_path}\n")
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}", exc_info=True)
            raise

    def load_raw_data(self):
        """
        Orchestrates the loading, cleaning, and saving of raw data.
        Returns:
            pandas.DataFrame: The processed dataframe.
        """
        try:
            df = self._fetch_data()
            df = self._validate_and_filter_columns(df)
            df = self._process_dates(df)
            
            self.logger.info(f"Load Data Set Completed. Final shape: {df.shape}\n")
            
            self._save_data(df)
            return df
            
        except Exception as e:
            self.logger.error(f"Error in load_raw_data process: {e}", exc_info=True)
            raise ValueError("Error cleaning and loading raw data")

