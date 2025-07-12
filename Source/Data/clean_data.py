import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Source.Utils.helpers import load_data


required_col = ['Billing Date', 'Month ', 'Season', 'District ',
                'Product ', 'Inv Qty.', 'QTY_MT', 'Invoice Value']
