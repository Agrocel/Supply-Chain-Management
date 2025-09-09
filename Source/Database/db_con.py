import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
import json

"THIS FILE IS ENGINE OF SQL"

with open('Z:\\Supply-Chain_management(SCM)\\Source\\config.json', "r") as f:
    config = json.load(f)

def get_engine():
    return create_engine(f"mysql+pymysql://{config['username']}@{config['host']}/{config['database']}")

