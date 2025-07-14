import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os 
import re
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Source.Utils.helpers import load_data
from Logging.logger import get_logger


# Adding logger
logger = get_logger("Prophet-Model-Training")

# Load Dataset
logger.info('Loading DataSet')
try:
    data = load_data() # Assuming load_data needs a file path argument
    logger.info(f'Dataset loaded with {len(data)} rows')
except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}", exc_info=True)
    exit()
def train_prophet():
    # Prepare the data for Prophet
    logger.info("Preparing Data for Prophet")
    prophet_data = data[['Date', 'QTY_MT']].rename(columns = {'Date':'ds', 'QTY_MT':'y'})    #  change Name  according to the model



    # Train test data seperation for model
    test_size = 5
    try:   
        if len(prophet_data) < test_size + 2:
            raise ValueError(
                f"Dataset is too small. It has {len(prophet_data)} rows., but need atlest"
                f"{test_size + 2} rows for testing"
            )
        train = prophet_data.iloc[:-test_size]
        test = prophet_data['ds'].iloc[-test_size:]

        logger.info(f'Train-test split completed, Training on {len(train)} rows and testing on {len(test)} rows')
    except (ValueError, IndexError) as e:
        logger.error("An error occured during train test spilt:{e}",exc_info=True)
        exit()

    try:
        # Fit Data to the Model
        logger.info("Fitting Data to the Model......")
        model = Prophet(
        growth='linear',                 # 'linear' or 'logistic'
        changepoint_range=0.5,        # % of data to check for changepoints
        changepoint_prior_scale=0.05,   # Flexibility of trend changes
        seasonality_mode='additive',    # 'additive' or 'multiplicative'
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        yearly_seasonality='auto',      # Can be True / False / number of Fourier terms
        weekly_seasonality='auto',
        daily_seasonality='auto',
        interval_width=0.80,            # Width of uncertainty interval
        uncertainty_samples=1000        # For prediction intervals
        )                                     
        model.fit(train)
        logger.info("Model Training Complete")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}", exc_info=True)
        exit()


    try:
        # Model Evalution
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)
        forecast_train = model.predict(train['ds'])
        logger.info("Model Evaluation Complete")
    except Exception as e:
        logger.error(f"An error occurred during model prediction: {e}", exc_info=True)
        exit()

    try:
        MAE_train = mean_absolute_error(train['y'], forecast_train['yhat'])
        MAE = mean_absolute_error(train['y'], forecast['yhat'])
        logger.info(f"Train data MAE is {MAE_train}")
        logger.info(f"Forecast MAE is: {MAE}")
    except Exception as e:
        logger.error(f"An error occurred during MAE calculation: {e}", exc_info=True)
