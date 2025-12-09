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
from Source.Models.prophet_gj import train_prophet_model_gj
from Source.Models.prophet_maha import train_prophet_model_maha
from Source.Models.prophet_cg import train_prophet_model_cg
from Source.Models.prophet_tn import train_prophet_model_tn
from Source.Evalution.evalution import interactive_evalution


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_existing = os.path.join(BASE_DIR, 'Data', 'Processed', 'data_load_raw_data.csv')
data_new = os.path.join(BASE_DIR, 'Data', 'Processed', 'data-25-Aug.csv')

with open(os.path.join(BASE_DIR, 'Source', 'config.json'), "r") as f:
    config = json.load(f)
logger = get_logger("Main-Logger")



def gujarat(data_GJ):
    prophet_data, forecast_future, prophet_data_pred = train_prophet_model_gj(data_GJ)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, 'Gujarat')

def maharashtra(data_MH):
    prophet_data, forecast_future, prophet_data_pred =train_prophet_model_maha(data_MH)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, 'Maharashtra')

def chattisgarh(data_CG):
    prophet_data, forecast_future, prophet_data_pred =train_prophet_model_cg(data_CG)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, 'Chattisgarh')

def TamilNadu(data_TN):
    prophet_data, forecast_future, prophet_data_pred =train_prophet_model_tn(data_TN)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, 'TamilNadu')


if  __name__ == "__main__":
    logger.info("Main logger Started.")
    data = load_raw_data()
    logger.info("data Loading Completed..")

    logger.info("Cleaing data...........")
    data_GJ, data_CG, data_MH, data_TN, data = Clean_raw_data(data)
    logger.info("Cleanig Completed.")
    
    logger.info("Gujarat Started......")
    gujarat(data_GJ)
    logger.info("Gujarat Completed ")

    logger.info("Maharashtra Started......")
    maharashtra(data_MH)
    logger.info("Maharashtra Completed ")

    logger.info("Chattisgarh Started......")
    chattisgarh(data_CG)
    logger.info("Chattisgarh Completed ")

    logger.info("TamilNadu Started......")
    TamilNadu(data_TN)
    logger.info("TamilNadu Completed ")