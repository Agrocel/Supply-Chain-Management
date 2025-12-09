import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
import json
import os

"THIS FILE IS ENGINE OF SQL"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, '..', 'config.json'), "r") as f:
    config = json.load(f)

def get_engine():
    return create_engine(f"mysql+pymysql://{config['username']}@{config['host']}/{config['database']}")

