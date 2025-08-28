import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Source.Data.load_data import load_raw_data
from Logging.logger import get_logger
import json

data_existing = r"Z:\Supply-Chain_management(SCM)\Data\Raw\Sales\Sales-20-25(Jan-June).csv"
data_new = r'Z:\Supply-Chain_management(SCM)\Data\Raw\Sales\Sales-25-July.xlsx'


# For Configuration of file
with open('Z:\\Supply-Chain_management(SCM)\\Source\\config.json', "r") as f:
    config = json.load(f)
logger = get_logger("test_load_data")


if __name__ == "__main__":
    logger.info("Loading Raw data\n")
    data = load_raw_data(data_existing, data_new)
    print(data['Billing Date'].dtype)
