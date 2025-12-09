import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Logging.logger import get_logger
from Source.Utils.helpers import load_data
import json
from sqlalchemy import create_engine
from datetime import datetime



# ----------------------------Configuration of files--------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'Source', 'config.json'), "r") as f:
    config = json.load(f)


# ----------------------------Logger------------------------------------------------------
logger = get_logger("Load_data_set_files")



def load_raw_data():
    """
    Loads and concatenates two raw Excel datasets, ensuring consistent column formatting.

    This function performs the following:
    - Loads two Excel datasets using a custom `load_data()` function.
    - Validates and reorders columns to match a required structure.
    - Concatenates both datasets into a single DataFrame.
    - Logs key operations and checks for missing required columns.

    Parameters
    ----------
    Existing : str
        File path or identifier for the existing dataset (Excel file).
    new : str
        File path or identifier for the new dataset (Excel file).

    Returns
    -------
    pandas.DataFrame
        A single DataFrame containing the combined data from both input files,
        with validated and ordered columns.

    Raises
    ------
    ValueError
        If any required columns are missing in the concatenated DataFrame.

    Logging
    -------
    Logs detailed progress at each step, including loading, formatting,
    and concatenation of datasets. Errors and missing columns are also logged.
    """


    #-------------------- Load Existing and raw data--------------------#
    logger.info("Lodaing Data from database ...........\n")
    engine = create_engine(f"mysql+pymysql://{config['username']}@{config['host']}/{config['database']}")
    df = pd.read_sql("SELECT * FROM raw_data",con=engine)
    logger.info("raw_data loaded from database\n")


    # logger.info("Loading Dataset.........")
    # data_existing= load_data(Existing)
    # data_new = load_data(new)
    # logger.info("Dataset completly loaded \n")

    try:
        # logger.info("Formating Column in order in both df\n")
        # logger.info(f"head of existing data {data_existing.head()}")
        # logger.info(f"Info of existing data {data_existing.info()}")
        logger.info(f"Head of raw_data{df.head()}")
        required_columns = [
            'Billing_Date', 'Sold_To_Party_Name', 'Invoice_Value', 'Plant_Code',
            'Mat_Desc','Inv_Qty','Inv_Qty_UOM']

        df = df[required_columns]
        # data_existing = data_existing[required_columns]
        # data_new = data_new[required_columns]


        # ----------------------------Datetime Foramt for Both DFs-----------------------------------------#
        # data_existing['Billing Date'] = pd.to_datetime(data_existing['Billing Date'], errors='coerce',dayfirst= True)
        # data_new['Billing Date'] = pd.to_datetime(data_new['Billing Date'], errors='coerce', dayfirst = True)
        # data_existing['Billing Date'] = data_existing['Billing Date'].dt.date
        # data_new['Billing Date'] = data_new['Billing Date'].dt.date
        df['Billing_Date'] = pd.to_datetime(df['Billing_Date'], errors='coerce', dayfirst = True)


        #----------------------------Combining Both Data---------------------------------------------------#

        # logger.info("Staring Concating two df's........")
        # logger.info(f"Before Concat:{data_new.shape},{data_existing.shape}")
        # data_25 = pd.concat([data_existing, data_new], ignore_index=True)
        # logger.info("successfully Concated df's\n")

        initial = df.shape[0]
        logger.info(f'Number of row in data:{initial}\n')
        data_25 = df.dropna(subset = ['Billing_Date'])
        final = df.shape[0]
        logger.info(f'Number of rows Affected {final-initial}\n')
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            logger.warning(f"Missing columns in the dataset: {missing}")
            raise ValueError(f'Missing required columns :{missing}')
        else:
            df = df[required_columns]
        logger.info("Columns are already Present in the Data.\n")
        logger.info(f"Load Data Set Completed\n{df.shape}")
        

        #-----------------------------------------Saving File----------------------------------------------#
        data_25.to_csv(os.path.join(PROJECT_ROOT, 'Data', 'Processed', 'data_load_raw_data.csv'), index=False)
        
    except Exception as e:
        logger.error(f"Error in Foramting the Columns in df's :{e}",exc_info=True)
        raise ValueError(f"Error while foramting Column in df's")
    
        
    
    return data_25
