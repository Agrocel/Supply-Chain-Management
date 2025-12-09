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
from Source.Models.prophet_maha import train_prophet_model_maha
from Source.Models.prophet_cg import train_prophet_model_cg
from Source.Evalution.evalution import interactive_evalution
# THis is new model okay 




# Config File 
Base_dir = os.path.dirname(os.path.abspath(__file__))
# data_existing = os.path.join(Base_dir,"Data\Raw\Sales\data_load_raw_data.csv")
# data_new = os.path.join(Base_dir,"Data\Raw\Sales\Sales-25-Aug.xlsx")


# For Configuration of file
with open(os.path.join(Base_dir,"..","Source","config.json"), "r") as f:
    config = json.load(f)
logger = get_logger("test_Clean_data")



if __name__ == "__main__":

    data_25 = load_raw_data()
    data_gj ,data_cg,data_mh,data_tn,data = Clean_raw_data(data_25)

    prophet_cg_data,forecast_future,prophet_cg_data_pred = train_prophet_model_cg(data_cg)  
    interactive_evalution(prophet_cg_data,forecast_future,prophet_cg_data_pred, 'Chattisgarh_test')
    



    