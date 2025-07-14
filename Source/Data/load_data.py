import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
from Logging import logger
from Source.Utils.helpers import load_data
import json

# For Configuration of file
with open('config.json',"r") as f:
    config = json.load(f)


# Logger
logger = logger("Load_data_set_files")

def load_raw_data(Existing,new):

    # Load raw data
    logger.info("Loading Dataset.........")
    data_existing= load_data(Existing)
    data_new = load_data(new)
    logger.info("Dataset completly loaded \n")

    try:
        logger.info("Formating Column in order in both df\n")
        required_columns = [
            'Billing Date', 'Sold-To-Party Name', 'Basic Value', 'Invoice Value',
            'Taxable Value', 'Agent Name', 'Plant Code', 'Mat. Code',
            'Mat. Desc.', 'Inv Qty.', 'Inv Qty UOM.']


        data_existing = data_existing[required_columns]
        data_new = data_new[required_columns]

        logger.info("Staring Concating two df's........")
        data_25 = pd.concat([data_existing, data_new])
        logger.info("successfully Concated df's\n")

        missing = [col for col in required_columns if col not in data_25.columns]

        if missing:
            logger.warning(f"Missing columns in the dataset: {missing}")
            raise ValueError(f'Missing required columns :{missing}')
        else:
            data_25 = data_25[required_columns]
        logger.info("Columns are already Present in the Data.\n")

    except Exception as e:
        logger.error(f"Error in Foramting the Columns in df's :{e}")
        ValueError("Error while foramting Column in df's{e}")
