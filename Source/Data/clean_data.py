import pandas as pd 
import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Source.Utils.helpers import load_data
from Logging.logger import get_logger
import json

# information from the Config file
with open('config.json',"r") as f:
    config = json.load(f)

# Addes New Log file for Scm-data-Clean 
logger = get_logger("SCM-Data-Clean")

# Data Required for the File to map 
state_district_df = load_data(config[r'state_district'])


def Clean(data):
    """
    Cleans and filters raw SCM data for Mahalaabh-related analysis.

    This function performs the following operations:
    - Cleans and preprocesses the input DataFrame
    - Filters only Mahalaabh-related product entries
    - Splits the cleaned Mahalaabh data into subsets based on major states:
      Gujarat, Maharashtra, Tamil Nadu, and Chhattisgarh

    Parameters:
    ----------
    data : pandas.DataFrame
        The raw input data containing columns like 'Mat. Desc.', 'District', etc.

    Returns:
    -------
    data_gj : pandas.DataFrame
        Filtered Mahalaabh data for Gujarat.

    data_mh : pandas.DataFrame
        Filtered Mahalaabh data for Maharashtra.

    data_tn : pandas.DataFrame
        Filtered Mahalaabh data for Tamil Nadu.

    data_cg : pandas.DataFrame
        Filtered Mahalaabh data for Chhattisgarh.

    data : pandas.DataFrame
        Fully cleaned and transformed input data.

    data_mahalaabh : pandas.DataFrame
        Filtered data containing only Mahalaabh-related products.
    """


    logger.info("Checking if columns are same as required......")
    required_columns = [
        'Billing Date', 'Sold-To-Party Name', 'Basic Value', 'Invoice Value',
        'Taxable Value', 'Agent Name', 'Plant Code', 'Mat. Code',
        'Mat. Desc.', 'Inv Qty.', 'Inv Qty UOM.']

    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        logger.warning(f"Missing columns in the dataset: {missing}")
        raise ValueError(f'Missing required columns :{missing}')
    else:
        data = data[required_columns]
    logger.info("Columns are already Present in the Data.\n")


    # Map Product and Clean Mat. Desc. Column
    try:
        product_patterns = {                            
            "Dripsafe": r"^dripsafe.*",
            "Herbovita": r"^herbovita.*",
            "Mahakite": r".*mahakite.*",
            "Mahalaabh Gr.": r"^mahalaabh.*",
            "Mahalaabh": r"^potassium.*",
            "Neem oil ": r"^neem oil.*",
            "Neem cake": r".*cake.*",
            "Quickact ": r".*quickact.*",
            "Nutrubhumi": r"^nutribhumi.*",
            "K-Immune": r".*immune.*",
            "Butrbloom Super": r".*butrabloom.*",
            "L.G.O.": r".*l.g.o.*",
            "Gibber power": r".*gibber.*",
        }

        def map_product(mat_desc):
            if pd.isna(mat_desc):
                return mat_desc
            mat_desc = mat_desc.strip()
            for product, pattern in product_patterns.items():
                if re.search(pattern, mat_desc, re.IGNORECASE):
                    return product
            return "value not Found in product_pattern"

        logger.info("Creating Clean Product Column.......")
        data['Product'] = data['Mat. Desc.'].apply(map_product)
        logger.info("Sucessfully Created Product Column.\n")

        logger.info("Dropping NA value in Product Columns.......")
        initial_count = data.shape[0]
        data = data.dropna(subset=["Product"])
        final_count = data.shape[0]
        rows_removed = initial_count - final_count
        logger.info(f"Number of rows removed: {rows_removed}\n")

    except Exception as e:
        logger.error(f"Error occured while mapping product :{e}")



    # Create Season, Month and FY columns(mostly Date columns)
    try:
        data['Billing Date'] = pd.to_datetime(data['Billing Date'], format='%Y-%m-%d', errors='coerce')

        month_mapping = {
            "January": "Jan", "February": "Feb", "March": "March",
            "April": "April", "May": "May", "June": "June",
            "July": "July", "August": "Aug", "September": "Sep",
            "October": "Oct", "November": "Nov", "December": "Dec"
        }

        logger.info("Creating Month Column......")                                    
        data['Month'] = data['Billing Date'].dt.strftime("%B").map(month_mapping) 
        logger.info("successfully Created Month Column.\n")

        logger.info("Creating Season Column.......")
        data["Season"] = data["Month"].apply(
            lambda x: "Kharif" if x in ["April", "May", "June", "July", "Aug", "Sep"] else "Rabi"
        )
        logger.info('Successfully Created Season column.\n')

        def get_financial_year(date):               # Extract Financial year
            year = date.year
            month = date.month

            if month < 4:                           # If Jan, Feb, Mar, it's part of pervious FY
                fy_start = year - 1
                fy_end = year
            else:                                   # If Apr-Dec its part of current FY
                fy_start = year
                fy_end = year + 1

            return f"{str(fy_start)[-2:]}-{str(fy_end)[-2:]}"       # Format as 'YY-YY'

        logger.info("Creating FY Column..........")
        data['FY'] = data['Billing Date'].apply(get_financial_year)
        logger.info("Successfully Created FY Column\n")

        data['Year'] = data['Billing Date'].dt.year
        data['Num_Month'] = data['Billing Date'].dt.month

    except Exception as e:
        logger.error(f"Error occured while Creating Season,Month or FY Columns:{e}")



    try:
        plant_state_mapping = {
        "CAP1": "Andhra Pradesh",
        "CCH1": "Chhattisgarh",
        "CGJ1": "Gujarat",
        "CHP1": "Himachal Pradesh",
        "CHR1": "Haryana",
        "CHR2": "Haryana",
        "CKA1": "Karnataka",
        "CMH1": "Maharashtra",
        "CMH2": "Maharashtra",
        "CMP1": "Madhya Pradesh",
        "CPB1": "Punjab",
        "CRJ1": "Rajasthan",
        "CTN1": "Tamil Nadu",
        "CWB1": "West Bengal",
        "DH01": "Gujarat"
        }

        logger.info("Creating State Columns.......")
        data['State'] = data['Plant Code'].map(plant_state_mapping)
        logger.info("Successfully Created State Columns\n")

    except Exception as e:
        logger.error(f"Error occured while Creating State column:{e}")


    # This bock of code Creates New Columns, filter data(mahalaabh).
    Deaslership_to_district = state_district_df.set_index('Dealership Name')['District'].to_dict()
    data['District'] = data['Sold-To-Party Name'].map(Deaslership_to_district)
    logger.info(f"Number of rows where district is not found:{data['District'].isna().sum()}")
    data.rename(columns = {"Billing Date":"Date"}, inplace =True)


    logger.info("Filtering Mahalaabh and Gr. from the data.....")
    data_mahalaabh = data[data['Product'].isin(['Mahalaabh Gr.', 'Mahalaabh'])].copy()
    logger.info(f"Number of rows after filtering:{data_mahalaabh.shape}\n")
    
    if data_mahalaabh.empty:
        logger.warning("No mahalaabh record found after filtering")

    logger.info('Converting Kg to MT.....')
    data_mahalaabh['UOM'] = 'MT'
    data_mahalaabh['QTY_MT'] = data_mahalaabh['Inv Qty.'] / 1000
    logger.info("Sucessfully Converted Kg to MT\n")


    logger.info("Required Columns.......")
    required_col = ['Date','District','Product','QTY_MT','UOM','Season',"State",'FY','Month','Invoice Value','Num_Month','Year']
    missing_col = [col for col in required_col if col not in data_mahalaabh.columns]
    if missing_col:
        logger.error(f'Missing Columns(columns are not matching):{missing_col}\n')
    
    data_mahalaabh = data_mahalaabh[required_col]

    data_GJ = data_mahalaabh[data_mahalaabh['State'] == 'Gujarat']
    data_MH = data_mahalaabh[data_mahalaabh['State'] == 'Maharashtra']
    data_CG = data_mahalaabh[data_mahalaabh['State'] == 'Chhattisgarh']
    data_TN = data_mahalaabh[data_mahalaabh['State'] == 'Tamil Nadu']
    logger.info("Sucessfully Created MH,TN,CG,GJ")
    logger.info("---- Step Completed ----\n")

    return data_GJ,data_CG,data_MH,data_TN,data_mahalaabh,data

