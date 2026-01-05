import json
import pandas as pd 
import numpy as np 
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from Logging.logger import get_logger 
from Source.Data.load_data import load_raw_data
from Source.Data.clean_data import Clean_raw_data
from Source.Models.train_prophet import train_model_prophet
from Source.Evalution.evalution import interactive_evalution


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(BASE_DIR, 'Source', 'config.json'), "r") as f:
    config = json.load(f)
logger = get_logger("Main-Logger")

# Configuration for Prophet Model for each state
# This allows for specific tuning per state if needed.
STATE_MODEL_CONFIGS = {
    'Gujarat': {
        'changepoint_prior_scale': 0.1,
        'seasonality_mode': 'multiplicative',
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'interval_width': 0.80,
    },
    'Maharashtra': {
        'changepoint_prior_scale': 0.1,
        'seasonality_mode': 'multiplicative',
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'interval_width': 0.80,
    },
    'Chattisgarh': {
        'changepoint_prior_scale': 0.1,
        'seasonality_mode': 'multiplicative',
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'interval_width': 0.80,
    },
    'TamilNadu': {
        'changepoint_prior_scale': 0.1,
        'seasonality_mode': 'multiplicative',
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'interval_width': 0.80,
    }
}

def process_state(state_name, data, model_params):
    """
    Train model and evaluate for a specific state.
    """
    logger.info(f"{state_name} Started......")
    try:
        prophet_data, forecast_future, prophet_data_pred = train_model_prophet(data, state_name, model_params)
        interactive_evalution(prophet_data, forecast_future, prophet_data_pred, state_name)
        logger.info(f"{state_name} Completed ")
    except Exception as e:
        logger.error(f"Error processing {state_name}: {e}", exc_info=True)


if  __name__ == "__main__":
    logger.info("Main logger Started.")
    data = load_raw_data()
    logger.info("data Loading Completed..")

    logger.info("Cleaing data...........")
    # clean_data returns: data_GJ, data_CG, data_MH, data_TN, data
    data_GJ, data_CG, data_MH, data_TN, data_all = Clean_raw_data(data)
    logger.info("Cleanig Completed.")
    
    # Process Gujarat
    process_state('Gujarat', data_GJ, STATE_MODEL_CONFIGS['Gujarat'])

    # Process Maharashtra
    process_state('Maharashtra', data_MH, STATE_MODEL_CONFIGS['Maharashtra'])

    # Process Chattisgarh
    process_state('Chattisgarh', data_CG, STATE_MODEL_CONFIGS['Chattisgarh'])

    # Process TamilNadu
    process_state('TamilNadu', data_TN, STATE_MODEL_CONFIGS['TamilNadu'])
