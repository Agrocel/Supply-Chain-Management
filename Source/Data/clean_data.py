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
from datetime import datetime 


# ----------------------------Config File------------------------------#
with open(r'Z:\\Supply-Chain_management(SCM)\\Source\\config.json',"r") as f:
    config = json.load(f)


# ----------------------------Log File---------------------------------#
logger = get_logger("SCM-Data-Clean")


# ----------------------------State File Config Path ------------------# 
state_district_df = load_data(config[r'state_district'])



def Clean_raw_data(data):
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

    """
    

    # ---------------------PRODUCT COLUMN--------------------------------------#
    try:
        product_patterns = {                            
            "Dripsafe": r"^dripsafe.*",
            "Herbovita": r"^herbovita.*",
            "Mahakite": r".*mahakite.*",
            "Boost-1kg": r"Potassium Schoenite (Boost-1kg )",
            "mahalaabh bulk" : r"Potassium Schoenite(Potassium Schoenite)",
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



    #-------------------------Create Season, Month and FY columns(mostly Date columns)--------------------------#
    try:
        # Billing Date Column
        logger.info("Converting Billing Date to datetime format.......")
        # data.loc[: ,'Billing Date'] = pd.to_datetime(data['Billing Date'], format='%d-%m-%Y', errors='coerce')
        # data.loc[:,'Billing Date'] = pd.to_datetime(data['Billing Date'],format='%d-%m-%Y', errors='coerce')
        # data.loc[: ,'Billing Date'] = data['Billing Date'].dt.date()


        invalid_data = data['Billing Date'].isna().sum()
        if invalid_data > 0:
            logger.warning(f"{invalid_data} invalid billing date row dropped due to conversion failure")
            data = data.dropna(subset=['Billing Date'])


        # Month Column
        month_mapping = {
            "January": "Jan", "February": "Feb", "March": "March",
            "April": "April", "May": "May", "June": "June",
            "July": "July", "August": "Aug", "September": "Sep",
            "October": "Oct", "November": "Nov", "December": "Dec"
        }
        
        logger.info("Creating Month Column......")    
        data = data.copy()                                
        data.loc[:,'Month'] = data['Billing Date'].dt.strftime("%B").map(month_mapping)
        logger.info("successfully Created Month Column.\n")
        


        # Season Column
        logger.info("Creating Season Column.......")
        data = data.copy() 
        data.loc[:,"Season"] = data["Month"].apply(
            lambda x: "Kharif" if x in ["April", "May", "June", "July", "Aug", "Sep"] else "Rabi"
        )
        logger.info('Successfully Created Season column.\n')

        # F-Y Column
        def get_financial_year(date):               

            if pd.isna(date):
                return np.nan
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
        data = data.copy() 
        data.loc[:,'FY'] = data['Billing Date'].apply(get_financial_year)
        logger.info("Successfully Created FY Column\n")

        # Month and Year Column
        data['Num_Month'] = data['Billing Date'].dt.month
        data['Year'] = data['Billing Date'].dt.year

        # Billing Date to Date
        data = data.rename(columns = {"Billing Date":"Date"})

    except Exception as e:
        logger.error(f"Error occured while Creating Season,Month or FY Columns:{e}",exc_info=True)
        raise ValueError(f"Error Occured while Creating Season,Month or FY Columns")



    #-----------------------------------Plant Code Columns------------------------------------------------#
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
        data = data.copy() 
        data.loc[:,'State'] = data['Plant Code'].map(plant_state_mapping)
        logger.info("Successfully Created State Columns\n")

    except Exception as e:
        logger.error(f"Error occured while Creating State column:{e}")



    # --------------------------------------District Column ----------------------------------------------#
    try: 
        logger.info("Mapping Dealership to District Column....... ")
        Dealership_to_district = state_district_df.set_index('Dealership Name')['District'].to_dict()
        data = data.copy() 
        data.loc[:,'District'] = data['Sold-To-Party Name'].map(Dealership_to_district)
        logger.info(f"Number of rows where district is not found:{data['District'].isna().sum()}")
    except Exception as e:
        logger.error(f"Error occured while Creating District column:{e}")
        raise ValueError(f"Error Occured while Creating District column")
    

    # ----------------------------------------Filtering Mahalaabh----------------------------------------#
    try:
        logger.info("Filtering Mahalaabh and Gr. from the data.....")
        data = data[data['Product'].isin(['Mahalaabh Gr.', 'Mahalaabh'])].copy()
        logger.info(f"Number of rows after filtering:{data.shape}\n")
        
        if data.empty:
            logger.warning("No mahalaabh record found after filtering")
        


    except Exception as e:
        logger.error(f"Error occured while Filtering Mahalaabh:{e}")
        raise ValueError(f"Error Occured while Filtering Mahalaabh")


    # ---------------------------------------------QTY_MT--------------------------------------#
    try:
        logger.info('Converting Kg to MT.....')
        data['UOM'] = 'MT'
        data.loc[:, 'Inv Qty.'] = pd.to_numeric(data['Inv Qty.'], errors='coerce')
        data.loc[:, 'QTY_MT'] = data['Inv Qty.'] / 1000
        logger.info("Sucessfully Converted Kg to MT\n")


    except Exception as e:
        logger.error(f"Error occured while converting Kg to MT:{e}")
        raise ValueError(f"Error Occured while converting Kg to MT:{e}")


    # -------------------------------------------Monthly------------------------------------------------#
    try:
        data = data.copy()
        data['Invoice Value'] = data.to_numeric(data['Invoice Value'], errors='coerce')
        data = data.groupby([pd.Grouper(key="Date", freq='MS'), "State"]).agg({  
            'Product': 'first',
            'Invoice Value': 'sum',
            'QTY_MT': 'sum',
            'UOM': 'first',
            'Season': 'first',
            'FY': 'first',
            'Month': 'first',
            'Num_Month': 'first',
            'Year': 'first',
        }).reset_index() 
    except Exception as e:
        logger.error(f"Error occured while creating monthly data:{e}")
        raise ValueError(f"Error Occured while creating monthly data")


    # -------------------------------------------Columns Format--------------------------------# 
    try:

        logger.info("Required Columns.......")
        required_col = ['Date','Product','QTY_MT','UOM','Season',"State",'FY','Month','Invoice Value','Num_Month','Year']
        missing_col = [col for col in required_col if col not in data.columns]
        if missing_col:
            logger.error(f'Missing Columns(columns are not matching):{missing_col}\n')
        data = data[required_col]
        
        data_GJ = data[data['State'] == 'Gujarat'].copy()
        data_MH = data[data['State'] == 'Maharashtra'].copy()
        data_CG = data[data['State'] == 'Chhattisgarh'].copy()
        data_TN = data[data['State'] == 'Tamil Nadu'].copy()

        logger.info("Sucessfully Created MH,TN,CG,GJ varaible")
        logger.info("---- Step Completed ----\n")

    except Exception as e:
        logger.error(f"Error occured while creating state wise dataframes:{e}")
        raise ValueError(f"Error Occured while creating state wise dataframes")
    


    # -----------------------------------Creating CSV file--------------------------------------------#
    try:
        logger.info("Creating CSV for data.......")
        data_GJ.to_csv(config[r"data_GJ"], index=False)
        data_MH.to_csv(config[r"data_MH"], index=False)
        data_CG.to_csv(config[r"data_CG"], index=False)
        data_TN.to_csv(config[r"data_TN"], index=False)
        data.to_csv(config[r"data"], index=False)
        logger.info('Successfully created CSV files.\n')

    except Exception as e:
        logger.error(f"Error occurred while writing CSV files: {e}")
        raise ValueError("Error while writing CSV files")


    return data_GJ, data_CG, data_MH, data_TN, data
