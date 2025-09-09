import os 
import sys
import pandas as pd 
from sqlalchemy import create_engine
from datetime import datetime
import shutil

#CONFIGURATION
folder_incoming = r"Z:\Supply-Chain_management(SCM)\Data\Raw\Incoming"
folder_preocessed = r"Z:\Supply-Chain_management(SCM)\Data\Raw\Processed"
db_user = r'root'
db_host = r'localhost'
db_name = r'scm'
table_name = r'raw_data'


# Connect to database
engine = create_engine(f'mysql+pymysql://{db_user}@{db_host}/{db_name}')

# Scan for Excel File 
files = [f for f in os.listdir(folder_incoming) if f.endswith(('.xlsx', '.xls',".XLSX"))]

for file in files:
    file_path = os.path.join(folder_incoming, file)


    required_columns = [
        'Billing Date', 'Sold-To-Party Name', 'Invoice Value', 'Plant Code',
        'Mat. Desc.','Inv Qty.','Inv Qty UOM.']
    
    try:
        df = pd.read_excel(file_path, sheet_name=0, usecols = required_columns)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue



    df.columns = (
    df.columns.str.strip()  # remove leading/trailing spaces
              .str.replace(r'[^0-9a-zA-Z]+', '_', regex=True) 
              .str.replace(r'_+$', '', regex=True) # replace bad chars with _
    )
    df['imported_at'] = datetime.now()
    df['source_file_name'] = file


    # DUMP to DATABASE

    try:
        df.head(0).to_sql(name=table_name , con=engine, if_exists='append', index=False)   
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print(f"IMported {file} successfully")
    except Exception as e:
        print(f"Error importing {file}: {e}")
        continue

    # Move File to Processed
    shutil.move(file_path, os.path.join(folder_preocessed, file))