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
from Source.Utils.helpers import load_data
from Logging.logger import get_logger
from prophet.serialize import model_to_json
import json



# information from the Config file
# information from the Config file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, '..', 'config.json'), "r") as f:
    config = json.load(f)


def train_model_prophet(data, State, model_params=None):
    """
    Train and evaluate a Prophet model on time-series sales data.

    This function prepares time-series data for the Prophet model, 
    trains the model, evaluates its performance (MAE, RMSE, R2),
    saves the model and forecast results, and returns the key outputs.

    Steps:
        1. Prepares data for Prophet by renaming columns ('Date' → 'ds', 'QTY_MT' → 'y').
        2. Trains a Prophet model with specified hyperparameters.
        3. Generates predictions on historical and future data (3 months ahead).
        4. Calculates evaluation metrics: MAE, RMSE, and R2.
        5. Saves:
            - The trained Prophet model in JSON format.
            - Model performance metrics in a CSV file (append mode).
            - Forecast values (with confidence intervals) in a CSV file.
        6. Returns Prophet-prepared data, forecast for future, and fitted predictions.

    Args:
        data (pd.DataFrame): Input dataset with at least 'Date' and 'QTY_MT' columns.
        State (str): Name of the state, used for saving model and report files and logging.
        model_params (dict): Dictionary of Prophet hyperparameters.

    Returns:
        tuple:
            prophet_data (pd.DataFrame): Data formatted for Prophet ('ds', 'y').
            forecast_future (pd.DataFrame): Forecasted values for future periods.
            prophet_data_pred (pd.DataFrame): Model predictions on historical data.

    Raises:
        ValueError: If an error occurs during model training or saving.
    """

    # Logger with dynamic state name
    logger = get_logger(f"Prophet_{State}")

    # ----------------------------------Prepare data for Prophet---------------------------------#

    try:
    
        logger.info(f"Preparing Data for Prophet ({State})....")
        prophet_data = data[['Date', 'QTY_MT']].rename(columns = {'Date':'ds', 'QTY_MT':'y'}).copy()    

    except Exception as e:
        logger.error(f"An error occurred during data preparation for {State}: {e}", exc_info=True)
        raise ValueError(f"Error in Data Preparation Part for {State}")


    #------------------------------------Prophet Model-------------------------------------------# 
    try:

        logger.info("Fitting Data to the Model......")

        # Default parameters if none provided
        if model_params is None:
             model_params = {
                'changepoint_prior_scale': 0.1,
                'seasonality_mode': 'multiplicative',
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'interval_width': 0.80,
            }

        model = Prophet(**model_params)
        model.fit(prophet_data)
        logger.info("Model Training Complete\n")

    except Exception as e:
        logger.error(f"An error occurred during model training for {State}: {e}", exc_info=True)
        raise ValueError(f"Error in Model training Part for {State}")


    # ------------------------------------Model Evaluation--------------------------------------------#
    try:
        
        logger.info("Forecasting test and Future Data .........")
        future = model.make_future_dataframe(periods=3, freq='MS')
        forecast_future = model.predict(future)
        prophet_data_pred = model.predict(prophet_data)
        logger.info("Model Forecasting Complete\n")


        logger.info("Calculating MAE, RMSE and Accuracy for Test....")
        MAE = mean_absolute_error(prophet_data['y'], prophet_data_pred['yhat'])
        RMSE = np.sqrt(mean_squared_error(prophet_data['y'], prophet_data_pred['yhat']))
        accuracy = r2_score(prophet_data['y'], prophet_data_pred['yhat'])
        
        logger.info("MAE, RMSE and Accuracy calculated\n")
        logger.info(f'Value of MAE for Prophet is {MAE:.2f}')
        logger.info(f'Value of RMSE for Prophet is {RMSE:.2f}')
        logger.info(f'Value of R2 Score (Accuracy) for Prophet is {accuracy:.2f}')

    except Exception as e:
        logger.error(f"An error occurred during model prediction for {State}: {e}", exc_info=True)
        raise ValueError(f"Error in Model prediction Part for {State}")
        

        #------------------------------------ Saving Model and file-------------------------------------#

    try:
        logger.info("Saving Model......")
        model_path_key = f'Prophet_model_{State}'
        if model_path_key in config:
            with open(config[model_path_key],"w") as fout:
                fout.write(model_to_json(model))
            logger.info("Model Saved")
        else:
             logger.warning(f"Config key '{model_path_key}' not found. Model not saved.")

        # Save MAE and RMSE
        logger.info('Creating Report for Prophet Model.....')

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_key = f'model_evaluation_{State}'

        if report_key in config:
            save_model_performance = pd.DataFrame({
                'Date' : [timestamp],
                'Model': [f'Prophet_{State}'],
                'MAE':[MAE],
                'RMSE':[RMSE]
            })
            save_model_performance.to_csv(config[report_key], mode = 'a', index = False, header=not pd.io.common.file_exists(config[report_key]))
            logger.info('Model Report Genearated\n')
        else:
            logger.warning(f"Config key '{report_key}' not found. Evaluation report not saved.")


        # File for Forecast Value
        logger.info("Saving Forecast Data to CSV File......")
        forecast_key = f'model_forecast_{State}'

        if forecast_key in config:
            save_model_performace = pd.DataFrame({
                "Date_forecast": prophet_data_pred['ds'],
                "Forecast": prophet_data_pred['yhat'],
                "Lower Bound": prophet_data_pred['yhat_lower'],
                "Upper Bound": prophet_data_pred['yhat_upper'],
                "Accuracy" : accuracy
            })
            save_model_performace.to_csv(config[forecast_key], index=False)
        else:
            logger.warning(f"Config key '{forecast_key}' not found. Forecast data not saved.")
    

    except Exception as e:
        logger.error(f"Error Occured while Saving File for {State} :{e}", exc_info=True)
        raise ValueError(f"Error Occured while Saving File for {State}")

    return prophet_data, forecast_future, prophet_data_pred
