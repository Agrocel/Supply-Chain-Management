import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Source.Data.load_data import load_raw_data
from Logging.logger import get_logger
import json
from Source.Data.clean_data import Clean_raw_data
from Source.Models.train_prophet import train_model_prophet
from Source.Evalution.evalution import interactive_evalution






BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_existing = os.path.join(BASE_DIR, 'Data', 'Raw', 'Sales', 'Sales-20-25(Jan-June).csv')
data_new = os.path.join(BASE_DIR, 'Data', 'Raw', 'Sales', 'Sales-25-July.xlsx')


# For Configuration of file
with open(os.path.join(BASE_DIR, 'Source', 'config.json'), "r") as f:
    config = json.load(f)
logger = get_logger("test_Clean_data")


if __name__ == "__main__":
    logger.info("Loading Raw data\n")
    data = load_raw_data(data_existing, data_new)
    logger.info(f"Number of Rows {data.head()}")


    logger.info("Loading Clean Data\n")
    data_GJ, data_CG, data_MH, data_TN, data = Clean_raw_data(data)
    prophet_data, forecast_future, prophet_data_pred =train_model_prophet(data_GJ, 'Gujarat')
    interactive_evalution(prophet_data,forecast_future,prophet_data_pred, "Gujarat")
    
