import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os 
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Source.Utils.helpers import load_data
from Logging.logger import get_logger
from prophet.serialize import model_to_json
import json



# information from the Config file
with open(r'Z:\Supply-Chain_management(SCM)\Source\config.json',"r") as f:
    config = json.load(f)


# Adding logger
logger = get_logger("Prophet-Model-Training")

def train_model_prophet(data,State):
    """
    Train and evaluate a Prophet model on time-series sales data.

    This function prepares time-series data for the Prophet model, 
    trains the model, evaluates its performance (MAE, RMSE), 
    saves the model and forecast results, and returns the key outputs.

    Steps:
        1. Prepares data for Prophet by renaming columns ('Date' → 'ds', 'QTY_MT' → 'y').
        2. Splits the data into train/test sets (80/20, no shuffle).
        3. Trains a Prophet model with specified hyperparameters.
        4. Generates predictions on historical, test, and future data (4 months ahead).
        5. Calculates evaluation metrics: MAE and RMSE.
        6. Saves:
            - The trained Prophet model in JSON format.
            - Model performance metrics in a CSV file (append mode).
            - Forecast values (with confidence intervals) in a CSV file.
        7. Returns Prophet-prepared data, forecast for future, and fitted predictions.

    Args:
        data (pd.DataFrame): Input dataset with at least 'Date' and 'QTY_MT' columns.
        State (str): Name of the state, used for saving model and report files.

    Returns:
        tuple:
            prophet_data (pd.DataFrame): Data formatted for Prophet ('ds', 'y').
            forecast_future (pd.DataFrame): Forecasted values for future periods.
            prophet_data_pred (pd.DataFrame): Model predictions on historical data.

    Raises:
        ValueError: If an error occurs during model training or saving.
    """
     
    # ----------------------------------Prepare data for Prophet---------------------------------#

    try:
    
        logger.info("Preparing Data for Prophet....")
        prophet_data = data[['Date', 'QTY_MT']].rename(columns = {'Date':'ds', 'QTY_MT':'y'}).copy()    
        #train, test = train_test_split(prophet_data, test_size=0.2, shuffle=False)
        #logger.info(f"Data Prepared for Prophet : {train.shape}, {test.shape}\n")

    except Exception as e:
        logger.error(f"An error occurred during data preparation: {e}", exc_info=True)
        raise ValueError("Error in Data Preparation Part")


    #------------------------------------Prophet Model-------------------------------------------# 
    try:

        logger.info("Fitting Data to the Model......")
        model = Prophet(
            changepoint_prior_scale=0.1,   # Flexibility of trend changes
            seasonality_mode='multiplicative',    # 'additive' or 'multiplicative'
            seasonality_prior_scale=10.0,
            # holidays_prior_scale=10.0,
            yearly_seasonality=True,      # Can be True / False / number of Fourier terms
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,            # Width of uncertainty interval
            # uncertainty_samples=1000 
        )            
        model.fit(prophet_data)
        logger.info("Model Training Complete\n")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}", exc_info=True)
        raise ValueError("Error in Model training Part") 


    # ------------------------------------Model Evaluation--------------------------------------------#
    try:
        
        logger.info("Forecasting test and Future Data .........")
        future = model.make_future_dataframe(periods=4, freq='M')
        forecast_future = model.predict(future)
        prophet_data_pred = model.predict(prophet_data)
        # test_pred = model.predict(test)
        logger.info("Model Forecasting Complete\n")


        logger.info("Calculating MAE and RMSE For Test....")
        MAE = mean_absolute_error(prophet_data['y'], prophet_data_pred['yhat'])
        RMSE = np.sqrt(mean_squared_error(prophet_data['y'], prophet_data_pred['yhat']))
        
        logger.info("MAE and RMSE calculated\n")
        logger.info(f'Value of MAE for Prophet is {MAE:.2f}')
        logger.info(f'Value of RMSE for Prophet is {RMSE:.2f}')

    except Exception as e:
        logger.error(f"An error occurred during model prediction: {e}", exc_info=True)
        exit()
        

        #------------------------------------ Saving Model and file-------------------------------------#

    try:
        logger.info("Saving Model......")
        with open(config[rf'Prophet_model_{State}'],"w") as fout:
            fout.write(model_to_json(model))    
        logger.info("Model Saved")

        # Save MAE and RMSE
        logger.info('Creating Report for Prophet Model.....')

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_model_performance = pd.DataFrame({
            'Date' : [timestamp],
            'Model': ['Prophet'],
            'MAE':[MAE],
            'RMSE':[RMSE]
        })
        save_model_performance.to_csv(config[f'model_evaluation_{State}'], mode = 'a', index = False, header=not pd.io.common.file_exists(config[f'model_evaluation_{State}']))
        logger.info('Model Report Genearated\n')


        # File for Forecast Value
        logger.info("Saving Forecast Data to Excel File......")
        save_model_performace = pd.DataFrame({
            'Date' : prophet_data_pred['ds'],
            'Forecast': prophet_data_pred['yhat'],
            "Lower_Interval": prophet_data_pred['yhat_lower'],
            "Upper_Interval": prophet_data_pred['yhat_upper']
        })
        save_model_performace.to_csv(config[rf'model_forecast_{State}'], index=False)
    

    except Exception as e:
        logger.error(f"Error Occured while Saving File :{e}", exc_info=True)
        raise ValueError(f"Error Occured while Saving File")

    return prophet_data, forecast_future, prophet_data_pred


