import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# This module contains utility functions for the Project.



def load_data(file_data):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    file_data (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(file_data):
        raise FileNotFoundError(f"The file {file_data} does not exist.")
    
    if file_data.endswith(".csv"):
        data = pd.read_csv(file_data, parse_dates=["Billing Date"],encoding = "cp1252")
    elif file_data.endswith(".xlsx"):
        data = pd.read_excel(file_data)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    return data