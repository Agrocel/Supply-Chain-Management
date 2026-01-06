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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Source.Database.db_con import get_engine

# ----------------------------Config File------------------------------#
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'Source', 'config.json'), "r") as f:
    config = json.load(f)


# ----------------------------Log File---------------------------------#
logger = get_logger("SCM-Data-Clean")


# ----------------------------State File Config Path ------------------# 
state_district_df = load_data(config[r'state_district'])

class DataCleaner:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.logger = get_logger("SCM-Data-Clean")
        self.state_district_df = load_data(config[r'state_district'])
        self.engine = get_engine()

        self.product_patterns = {                            
            "Dripsafe": r"^dripsafe.*",
            "Herbovita": r"^herbovita.*",
            "Mahakite": r".*mahakite.*",
            "Mahalaabh Gr.": r"(?i)^mahalaabh.*",
            "Mahalaabh": r"(?i)^potassium.*",
            "Neem oil ": r"^neem oil.*",
            "Neem cake": r".*cake.*",
            "Quickact ": r".*quickact.*",
            "Nutrubhumi": r"^nutribhumi.*",
            "K-Immune": r".*immune.*",
            "Butrbloom Super": r".*butrabloom.*",
            "L.G.O.": r".*l.g.o.*",
            "Gibber power": r".*gibber.*",
        }

        self.month_mapping = {
            "January": "Jan", "February": "Feb", "March": "March",
            "April": "April", "May": "May", "June": "June",
            "July": "July", "August": "Aug", "September": "Sep",
            "October": "Oct", "November": "Nov", "December": "Dec"
        }

        self.plant_state_mapping = {
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

        self.required_col = ['Date','Product','QTY_MT','UOM','Season',"State",'FY','Month','Invoice_Value','Num_Month','Year']


    def _load_config(self, path):
        with open(path, "r") as f:
            return json.load(f)


    def _map_product_name(self, mat_desc):
        """Helper Method to map product name based on the mat_desc column."""
        if pd.isna(mat_desc):
            return mat_desc
        mat_desc = mat_desc.strip()
        for product, pattern in self.product_patterns.items():
            if re.search(pattern, mat_desc, re.IGNORECASE):
                return product
        return None   

    def _get_financial_year(self,date):               
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

        return f"{str(fy_start)[-2:]}-{str(fy_end)[-2:]}" 


    def _clean_product_column(self, df):
        self.logger.info("Mapping Product Column.......")
        df_clean = df[~df['Mat_Desc'].isin(['Potassium Schoenite (Boost-1kg )','Potassium Schoenite(Potassium Schoenite)'])]        
        
        self.logger.info("Creating Clean Product Column.......")
        df_clean['Product'] = df_clean['Mat_Desc'].apply(self._map_product_name)
        self.logger.info("Sucessfully Created Product Column.\n")

        self.logger.info("Dropping NA value in Product Columns.......")
        initial_count = df_clean.shape[0]
        df_clean = df_clean.dropna(subset=["Product"])
        final_count = df_clean.shape[0]
        rows_removed = initial_count - final_count
        self.logger.info(f"Number of rows removed: {rows_removed}\n")

        return df_clean

    def _clean_date_features(self, df):

        self.logger.info("Cleaning Date Column.......")

        invalid_data = df['Billing_Date'].isna().sum()
        if invalid_data > 0:
            self.logger.warning(f"{invalid_data} invalid billing date row dropped due to conversion failure")
            df = df.dropna(subset=['Billing_Date'])

        self.logger.info("Creating Month Column......")    
        df = df.copy()                                
        df.loc[:,'Month'] = df['Billing_Date'].dt.strftime("%B").map(self.month_mapping)
        self.logger.info("successfully Created Month Column.\n")

                # Season Column
        self.logger.info("Creating Season Column.......")
        df = df.copy() 
        df.loc[:,"Season"] = df["Month"].apply(
            lambda x: "Kharif" if x in ["April", "May", "June", "July", "Aug", "Sep"] else "Rabi"
        )
        self.logger.info('Successfully Created Season column.\n')

        self.logger.info("Creating FY Column..........")
        df = df.copy() 
        df.loc[:,'FY'] = df['Billing_Date'].apply(self._get_financial_year)
        self.logger.info("Successfully Created FY Column\n")

        # Month and Year Column
        df['Num_Month'] = df['Billing_Date'].dt.month
        df['Year'] = df['Billing_Date'].dt.year

        # Billing Date to Date
        df = df.rename(columns = {"Billing_Date":"Date"})

        return df

    def _plant_state_mapping(self,df):

        self.logger.info("Creating State Columns.......")
        df = df.copy() 
        df.loc[:,'State'] = df['Plant_Code'].map(self.plant_state_mapping)
        self.logger.info("Successfully Created State Columns\n")

        return df
    
    def _district_mapping(self,df):
        self.logger.info("Mapping Dealership to District Column....... ")
        Dealership_to_district = self.state_district_df.set_index('Dealership Name')['District'].to_dict()
        df = df.copy() 
        df.loc[:,'District'] = df['Sold_To_Party_Name'].map(Dealership_to_district)
        self.logger.info(f"Number of rows where district is not found:{df['District'].isna().sum()}")

        return df

    def _filtering_mahalaabh(self,df):

        self.logger.info(f"Filtering Mahalaabh and Gr. from the data.....{df.shape}")
        df = df[df['Product'].isin(['Mahalaabh Gr.', 'Mahalaabh'])].copy()
        self.logger.info(f"Number of rows after filtering:{df.shape}\n")
        
        if df.empty:
            self.logger.warning("No mahalaabh record found after filtering")

        return df

    
    def _clean_quantity(self,df):
        self.logger.info('Converting Kg to MT.....')
        df['UOM'] = 'MT'
        df.loc[:, 'Inv_Qty'] = pd.to_numeric(df['Inv_Qty'], errors='coerce')
        df.loc[:, 'QTY_MT'] = df['Inv_Qty'] / 1000
        self.logger.info("Sucessfully Converted Kg to MT\n")

        return df

    def _monthly_aggregation(self,df):
        self.logger.info("Monthly Aggregation....")
        df = df.copy()
        self.logger.info(f"data type of all columns:\n{df.dtypes}")
        df.loc[:, 'Invoice_Value'] = pd.to_numeric(df['Invoice_Value'], errors='coerce')
        df = df.groupby([pd.Grouper(key="Date", freq='MS'), "State"]).agg({  
            'Product': 'first',
            'Invoice_Value': 'sum',
            'QTY_MT': 'sum',
            'UOM': 'first',
            'Season': 'first',
            'FY': 'first',
            'Month': 'first',
            'Num_Month': 'first',
            'Year': 'first',
        }).reset_index()
        
        self.logger.info("Monthly Aggregation Completed\n")
        return df
    
    def _column_formatting(self,df):
        self.logger.info("Column Formatting....")
        self.logger.info("Required Columns.......")

        missing_col = [col for col in self.required_col if col not in df.columns]
        if missing_col:
            self.logger.error(f'Missing Columns(columns are not matching):{missing_col}\n')
        df = df[self.required_col]
        
        df_GJ = df[df['State'] == 'Gujarat'].copy()
        df_MH = df[df['State'] == 'Maharashtra'].copy()
        df_CG = df[df['State'] == 'Chhattisgarh'].copy()
        df_TN = df[df['State'] == 'Tamil Nadu'].copy()

        self.logger.info("Sucessfully Created MH,TN,CG,GJ varaible")
        self.logger.info("---- Step Completed ----\n")

        return df_GJ,df_MH,df_CG,df_TN,df

    def _saving_file(self,df_GJ,df_MH,df_CG,df_TN,df):
        self.logger.info("Creating CSV for data.......")
        df_GJ.to_csv(self.config[r"data_GJ"], index=False)
        df_MH.to_csv(self.config[r"data_MH"], index=False)
        df_CG.to_csv(self.config[r"data_CG"], index=False)
        df_TN.to_csv(self.config[r"data_TN"], index=False)
        df.to_csv(self.config[r"data"], index=False)
        self.logger.info('Successfully created CSV files.\n')

        self.logger.info("Clean data -> clean table")
        df.to_sql(name = self.config[r"clean_data_path"], con = self.engine, if_exists = 'replace', index = False)

    def process_all(self,raw_data):

        df = raw_data.copy()
        df = self._clean_product_column(df)
        df = self._clean_date_features(df)
        df = self._plant_state_mapping(df)
        df = self._district_mapping(df)
        df = self._filtering_mahalaabh(df)
        df = self._clean_quantity(df)
        df = self._monthly_aggregation(df)
        df_GJ,df_MH,df_CG,df_TN,df = self._column_formatting(df)
        self._saving_file(df_GJ,df_MH,df_CG,df_TN,df)
        
        return df_GJ,df_MH,df_CG,df_TN,df



# def Clean_raw_data(data):
#     """
#     Cleans and filters raw SCM data for Mahalaabh-related analysis.

#     This function performs the following operations:
#     - Cleans and preprocesses the input DataFrame
#     - Filters only Mahalaabh-related product entries
#     - Splits the cleaned Mahalaabh data into subsets based on major states:
#       Gujarat, Maharashtra, Tamil Nadu, and Chhattisgarh

#     Parameters:
#     ----------
#     data : pandas.DataFrame
#         The raw input data containing columns like 'Mat. Desc.', 'District', etc.

#     Returns:
#     -------
#     data_gj : pandas.DataFrame 
#         Filtered Mahalaabh data for Gujarat.
#     data_mh : pandas.DataFrame
#         Filtered Mahalaabh data for Maharashtra.

#     data_tn : pandas.DataFrame
#         Filtered Mahalaabh data for Tamil Nadu.

#     data_cg : pandas.DataFrame
#         Filtered Mahalaabh data for Chhattisgarh.

#     data : pandas.DataFrame
#         Fully cleaned and transformed input data.

#     """
    

#     # ---------------------PRODUCT COLUMN--------------------------------------#
#     try:
#         logger.info("Mapping Product Column.......")
#         data = data[~data['Mat_Desc'].isin(['Potassium Schoenite (Boost-1kg )','Potassium Schoenite(Potassium Schoenite)'])]

#         product_patterns = {                            
#             "Dripsafe": r"^dripsafe.*",
#             "Herbovita": r"^herbovita.*",
#             "Mahakite": r".*mahakite.*",
#             "Mahalaabh Gr.": r"(?i)^mahalaabh.*",
#             "Mahalaabh": r"(?i)^potassium.*",
#             "Neem oil ": r"^neem oil.*",
#             "Neem cake": r".*cake.*",
#             "Quickact ": r".*quickact.*",
#             "Nutrubhumi": r"^nutribhumi.*",
#             "K-Immune": r".*immune.*",
#             "Butrbloom Super": r".*butrabloom.*",
#             "L.G.O.": r".*l.g.o.*",
#             "Gibber power": r".*gibber.*",
#         }

#         def map_product(mat_desc):
#             if pd.isna(mat_desc):
#                 return mat_desc
#             mat_desc = mat_desc.strip()
#             for product, pattern in product_patterns.items():
#                 if re.search(pattern, mat_desc, re.IGNORECASE):
#                     return product
        
        
#         logger.info("Creating Clean Product Column.......")
#         data['Product'] = data['Mat_Desc'].apply(map_product)
#         logger.info("Sucessfully Created Product Column.\n")


#         logger.info("Dropping NA value in Product Columns.......")
#         initial_count = data.shape[0]
#         data = data.dropna(subset=["Product"])
#         final_count = data.shape[0]
#         rows_removed = initial_count - final_count
#         logger.info(f"Number of rows removed: {rows_removed}\n")

#     except Exception as e:
#         logger.error(f"Error occured while mapping product :{e}",exc_info=True)
#         raise ValueError(f"Error Occured while mapping product")



#     #-------------------------Create Season, Month and FY columns(mostly Date columns)--------------------------#
#     try:
#         # Billing Date Column
#         logger.info("Converting Billing Date to datetime format.......")
#         # data.loc[: ,'Billing Date'] = pd.to_datetime(data['Billing Date'], format='%d-%m-%Y', errors='coerce')
#         # data.loc[:,'Billing Date'] = pd.to_datetime(data['Billing Date'],format='%d-%m-%Y', errors='coerce')
#         # data.loc[: ,'Billing Date'] = data['Billing Date'].dt.date()


#         invalid_data = data['Billing_Date'].isna().sum()
#         if invalid_data > 0:
#             logger.warning(f"{invalid_data} invalid billing date row dropped due to conversion failure")
#             data = data.dropna(subset=['Billing_Date'])


#         # Month Column
#         month_mapping = {
#             "January": "Jan", "February": "Feb", "March": "March",
#             "April": "April", "May": "May", "June": "June",
#             "July": "July", "August": "Aug", "September": "Sep",
#             "October": "Oct", "November": "Nov", "December": "Dec"
#         }
        
#         logger.info("Creating Month Column......")    
#         data = data.copy()                                
#         data.loc[:,'Month'] = data['Billing_Date'].dt.strftime("%B").map(month_mapping)
#         logger.info("successfully Created Month Column.\n")
        


#         # Season Column
#         logger.info("Creating Season Column.......")
#         data = data.copy() 
#         data.loc[:,"Season"] = data["Month"].apply(
#             lambda x: "Kharif" if x in ["April", "May", "June", "July", "Aug", "Sep"] else "Rabi"
#         )
#         logger.info('Successfully Created Season column.\n')

#         # F-Y Column
#         def get_financial_year(date):               

#             if pd.isna(date):
#                 return np.nan
#             year = date.year
#             month = date.month

#             if month < 4:                           # If Jan, Feb, Mar, it's part of pervious FY
#                 fy_start = year - 1
#                 fy_end = year
#             else:                                   # If Apr-Dec its part of current FY
#                 fy_start = year
#                 fy_end = year + 1

#             return f"{str(fy_start)[-2:]}-{str(fy_end)[-2:]}"       # Format as 'YY-YY'

#         logger.info("Creating FY Column..........")
#         data = data.copy() 
#         data.loc[:,'FY'] = data['Billing_Date'].apply(get_financial_year)
#         logger.info("Successfully Created FY Column\n")

#         # Month and Year Column
#         data['Num_Month'] = data['Billing_Date'].dt.month
#         data['Year'] = data['Billing_Date'].dt.year

#         # Billing Date to Date
#         data = data.rename(columns = {"Billing_Date":"Date"})

#     except Exception as e:
#         logger.error(f"Error occured while Creating Season,Month or FY Columns:{e}",exc_info=True)
#         raise ValueError(f"Error Occured while Creating Season,Month or FY Columns")



#     #-----------------------------------Plant Code Columns------------------------------------------------#
#     try:
#         plant_state_mapping = {
#         "CAP1": "Andhra Pradesh",
#         "CCH1": "Chhattisgarh",
#         "CGJ1": "Gujarat",
#         "CHP1": "Himachal Pradesh",
#         "CHR1": "Haryana",
#         "CHR2": "Haryana",
#         "CKA1": "Karnataka",
#         "CMH1": "Maharashtra",
#         "CMH2": "Maharashtra",
#         "CMP1": "Madhya Pradesh",
#         "CPB1": "Punjab",
#         "CRJ1": "Rajasthan",
#         "CTN1": "Tamil Nadu",
#         "CWB1": "West Bengal",
#         "DH01": "Gujarat"
#         }

#         logger.info("Creating State Columns.......")
#         data = data.copy() 
#         data.loc[:,'State'] = data['Plant_Code'].map(plant_state_mapping)
#         logger.info("Successfully Created State Columns\n")

#     except Exception as e:
#         logger.error(f"Error occured while Creating State column:{e}",exc_info=True)
#         raise ValueError(f"Error Occured while Creating State column")





#     # --------------------------------------District Column ----------------------------------------------#
#     try: 
#         logger.info("Mapping Dealership to District Column....... ")
#         Dealership_to_district = state_district_df.set_index('Dealership Name')['District'].to_dict()
#         data = data.copy() 
#         data.loc[:,'District'] = data['Sold_To_Party_Name'].map(Dealership_to_district)
#         logger.info(f"Number of rows where district is not found:{data['District'].isna().sum()}")
#     except Exception as e:
#         logger.error(f"Error occured while Creating District column:{e}", exc_info=True)
#         raise ValueError(f"Error Occured while Creating District column")
    

#     # ----------------------------------------Filtering Mahalaabh----------------------------------------#
#     try:
#         logger.info(f"Filtering Mahalaabh and Gr. from the data.....{data.shape}")
#         data = data[data['Product'].isin(['Mahalaabh Gr.', 'Mahalaabh'])].copy()
#         logger.info(f"Number of rows after filtering:{data.shape}\n")
        
#         if data.empty:
#             logger.warning("No mahalaabh record found after filtering")
        

#     except Exception as e:
#         logger.error(f"Error occured while Filtering Mahalaabh:{e}", exc_info=True)
#         raise ValueError(f"Error Occured while Filtering Mahalaabh")


#     # ---------------------------------------------QTY_MT--------------------------------------#
#     try:
#         logger.info('Converting Kg to MT.....')
#         data['UOM'] = 'MT'
#         data.loc[:, 'Inv_Qty'] = pd.to_numeric(data['Inv_Qty'], errors='coerce')
#         data.loc[:, 'QTY_MT'] = data['Inv_Qty'] / 1000
#         logger.info("Sucessfully Converted Kg to MT\n")


#     except Exception as e:
#         logger.error(f"Error occured while converting Kg to MT:{e}",exc_info=True)
#         raise ValueError("Error Occured while converting Kg to MT")


#     # -------------------------------------------Monthly------------------------------------------------#
#     try:
#         data = data.copy()
#         logger.info(f"data type of all columns:\n{data.dtypes}")
#         data.loc[:, 'Invoice_Value'] = pd.to_numeric(data['Invoice_Value'], errors='coerce')
#         data = data.groupby([pd.Grouper(key="Date", freq='MS'), "State"]).agg({  
#             'Product': 'first',
#             'Invoice_Value': 'sum',
#             'QTY_MT': 'sum',
#             'UOM': 'first',
#             'Season': 'first',
#             'FY': 'first',
#             'Month': 'first',
#             'Num_Month': 'first',
#             'Year': 'first',
#         }).reset_index() 
#     except Exception as e:
#         logger.error(f"Error occured while creating monthly data:{e}",exc_info=True)
#         raise ValueError(f"Error Occured while creating monthly data")


#     # -------------------------------------------Columns Format--------------------------------# 
#     try:

#         logger.info("Required Columns.......")
#         required_col = ['Date','Product','QTY_MT','UOM','Season',"State",'FY','Month','Invoice_Value','Num_Month','Year']
#         missing_col = [col for col in required_col if col not in data.columns]
#         if missing_col:
#             logger.error(f'Missing Columns(columns are not matching):{missing_col}\n')
#         data = data[required_col]
        
#         data_GJ = data[data['State'] == 'Gujarat'].copy()
#         data_MH = data[data['State'] == 'Maharashtra'].copy()
#         data_CG = data[data['State'] == 'Chhattisgarh'].copy()
#         data_TN = data[data['State'] == 'Tamil Nadu'].copy()

#         logger.info("Sucessfully Created MH,TN,CG,GJ varaible")
#         logger.info("---- Step Completed ----\n")

#     except Exception as e:
#         logger.error(f"Error occured while creating state wise dataframes:{e}",exc_info=True)
#         raise ValueError(f"Error Occured while creating state wise dataframes")
    


#     # -----------------------------------Creating CSV file & Database--------------------------------------------#
#     try:
#         logger.info("Creating CSV for data.......")
#         data_GJ.to_csv(config[r"data_GJ"], index=False)
#         data_MH.to_csv(config[r"data_MH"], index=False)
#         data_CG.to_csv(config[r"data_CG"], index=False)
#         data_TN.to_csv(config[r"data_TN"], index=False)
#         data.to_csv(config[r"data"], index=False)
#         logger.info('Successfully created CSV files.\n')

#         logger.info("Clean data -> clean table")
#         data.to_sql(name = config[r"clean_data_path"], con = engine, if_exists = 'replace', index = False)
        

#     except Exception as e:
#         logger.error(f"Error occurred while writing CSV files: {e}",exc_info=True)
#         raise ValueError("Error while writing CSV files")


#     return data_GJ, data_CG, data_MH, data_TN, data
