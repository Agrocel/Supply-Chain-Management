import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os 
import re
import sys
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet.serialize import model_to_json
import json

# Adjust path to import custom modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..', '..')))
from Logging.logger import get_logger

class ProphetTrainer:
    def __init__(self, state_name, config_path, model_params=None):
        self.state = state_name
        self.config = self._load_config(config_path)
        self.logger = get_logger(f"Prophet_{state_name}")
        self.model_params = model_params or self._get_default_params()
        self.model = None

    def _load_config(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _get_default_params(self):
        return {
            'changepoint_prior_scale': 0.1,
            'seasonality_mode': 'multiplicative',
            'seasonality_prior_scale': 10.0,
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'interval_width': 0.80,
        }

    def prepare_data(self, data):
        """Prepares data for Prophet by renaming columns."""
        try:
            self.logger.info(f"Preparing Data for Prophet ({self.state})....")
            prophet_data = data[['Date', 'QTY_MT']].rename(columns={'Date': 'ds', 'QTY_MT': 'y'}).copy()
            return prophet_data
        except Exception as e:
            self.logger.error(f"Error preparing data for {self.state}: {e}", exc_info=True)
            raise ValueError(f"Error in Data Preparation Part for {self.state}")

    def train(self, prophet_data):
        """Trains the Prophet model."""
        try:
            self.logger.info("Fitting Data to the Model......")
            self.model = Prophet(**self.model_params)
            self.model.fit(prophet_data)
            self.logger.info("Model Training Complete\n")
        except Exception as e:
            self.logger.error(f"Error training model for {self.state}: {e}", exc_info=True)
            raise ValueError(f"Error in Model training Part for {self.state}")

    def evaluate(self, prophet_data):
        """Generates forecasts and calculates evaluation metrics."""
        try:
            self.logger.info("Forecasting test and Future Data .........")
            future = self.model.make_future_dataframe(periods=3, freq='MS')
            forecast_future = self.model.predict(future)
            prophet_data_pred = self.model.predict(prophet_data)
            self.logger.info("Model Forecasting Complete\n")

            self.logger.info("Calculating MAE, RMSE and Accuracy for Test....")
            MAE = mean_absolute_error(prophet_data['y'], prophet_data_pred['yhat'])
            RMSE = np.sqrt(mean_squared_error(prophet_data['y'], prophet_data_pred['yhat']))
            accuracy = r2_score(prophet_data['y'], prophet_data_pred['yhat'])

            self.logger.info(f"MAE: {MAE:.2f}, RMSE: {RMSE:.2f}, R2: {accuracy:.2f}\n")
            
            return forecast_future, prophet_data_pred, MAE, RMSE, accuracy

        except Exception as e:
            self.logger.error(f"Error during model prediction for {self.state}: {e}", exc_info=True)
            raise ValueError(f"Error in Model prediction Part for {self.state}")

    def save_artifacts(self, prophet_data_pred, MAE, RMSE, accuracy):
        """Saves model, report, and forecast data."""
        try:
            self.logger.info("Saving Model......")
            
            # Save Model
            model_path_key = f'Prophet_model_{self.state}'
            if model_path_key in self.config:
                with open(self.config[model_path_key], "w") as fout:
                    fout.write(model_to_json(self.model))
                self.logger.info("Model Saved")
            else:
                self.logger.warning(f"Config key '{model_path_key}' not found. Model not saved.")

            # Save Metrics Report
            report_key = f'model_evaluation_{self.state}'
            if report_key in self.config:
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_model_performance = pd.DataFrame({
                    'Date': [timestamp],
                    'Model': [f'Prophet_{self.state}'],
                    'MAE': [MAE],
                    'RMSE': [RMSE]
                })
                header = not pd.io.common.file_exists(self.config[report_key])
                save_model_performance.to_csv(self.config[report_key], mode='a', index=False, header=header)
                self.logger.info('Model Report Generated\n')

            # Save Forecast
            forecast_key = f'model_forecast_{self.state}'
            if forecast_key in self.config:
                save_forecast = pd.DataFrame({
                    "Date_forecast": prophet_data_pred['ds'],
                    "Forecast": prophet_data_pred['yhat'],
                    "Lower Bound": prophet_data_pred['yhat_lower'],
                    "Upper Bound": prophet_data_pred['yhat_upper'],
                    "Accuracy": accuracy
                })
                save_forecast.to_csv(self.config[forecast_key], index=False)
                self.logger.info("Forecast Data Saved\n")

        except Exception as e:
            self.logger.error(f"Error saving files for {self.state}: {e}", exc_info=True)
            raise ValueError(f"Error Occured while Saving File for {self.state}")

    def run(self, data):
        """Orchestrates the training pipeline."""
        prophet_data = self.prepare_data(data)
        self.train(prophet_data)
        forecast_future, prophet_data_pred, MAE, RMSE, accuracy = self.evaluate(prophet_data)
        self.save_artifacts(prophet_data_pred, MAE, RMSE, accuracy)
        
        return prophet_data, forecast_future, prophet_data_pred


