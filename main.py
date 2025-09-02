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


data_existing = r"Z:\Supply-Chain_management(SCM)\Data\Processed\data-25-Aug.csv"
data_new = r'Z:\Supply-Chain_management(SCM)\Data\Raw\Sales\Sales-25-July.xlsx'


with open('Z:\\Supply-Chain_management(SCM)\\Source\\config.json', "r") as f:
    config = json.load(f)
logger = get_logger("Main-Logger")



def gujarat(data_GJ, state):
    prophet_data, forecast_future, prophet_data_pred = train_model_prophet(data_GJ, state)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, state)

def maharashtra(data_MH, state):
    prophet_data, forecast_future, prophet_data_pred =train_model_prophet(data_MH, state)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, state)

def chattisgarh(data_CG, state):
    prophet_data, forecast_future, prophet_data_pred =train_model_prophet(data_CG, state)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, state)

def TamilNadu(data_TN, state):
    prophet_data, forecast_future, prophet_data_pred =train_model_prophet(data_TN, state)
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, state)


if  __name__ == "__main__":
    logger.info("Main logger Started.")
    data = load_raw_data(data_existing, data_new)
    logger.info("data Loading Completed..")

    logger.info("Cleaing data...........")
    data_GJ, data_CG, data_MH, data_TN, data = Clean_raw_data(data)
    logger.info("Cleanig Completed.")

    logger.info("Gujarat Started......")
    gujarat(data_GJ, 'Gujarat')
    logger.info("Gujarat Completed ")

    logger.info("Maharashtra Started......")
    maharashtra(data_MH, 'Maharashtra')
    logger.info("Maharashtra Completed ")

    logger.info("Chattisgarh Started......")
    chattisgarh(data_CG, 'Chattisgarh')
    logger.info("Chattisgarh Completed ")

    logger.info("TamilNadu Started......")
    TamilNadu(data_TN, 'TamilNadu')
    logger.info("TamilNadu Completed ")