import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Source.Utils.helpers import load_data
from Logging.logger import get_logger
import json
from datetime import datetime
from Source.Data import load_data


# For Configuration of file
with open('config.json',"r") as f:
    config = json.load(f)

sales_ap = r"Z:\Supply-Chain_management(SCM)\Data\Raw\Sales\Sales-25(Jan-Apr).xlsx"
sales_june = r"Z:\Supply-Chain_management(SCM)\Data\Raw\Sales\Sales-25(May-June).xlsx"

# Unit test of load_raw_data

if __name__ == "__main__":
    data = load_data.load_raw_data(sales_ap, sales_june)
    print(data.head())