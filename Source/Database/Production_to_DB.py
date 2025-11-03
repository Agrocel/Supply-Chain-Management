import pandas as pd 
import numpy as np 

import sqlalchemy 
from sqlalchemy import create_engine
from datetime import datetime
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Source.Database.db_con import get_engine

folder_incoming = r"E:\Supply-Chain_management(SCM)\Data\Raw\Production"
db_user = r'root'
db_host = r'localhost'
db_name = r'scm'
table_name = r'production_data'


# Connect to database
engine = create_engine(f'mysql+pymysql://{db_user}@{db_host}/{db_name}')

files = [f for f in os.listdir(folder_incoming) if f.endswith(('.xlsx', '.xls',".XLSX",'.csv','.CSV'))]

for file in files:
    file_path = os.path.join(folder_incoming, file)

    try:
        df = pd.read_csv(file_path)
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

    try:
        df.head(0).to_sql(name=table_name , con=engine, if_exists='append', index=False)   
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print(f"Imported {file} successfully")
    except Exception as e:
        print(f"Error importing {file}: {e}")
        continue




