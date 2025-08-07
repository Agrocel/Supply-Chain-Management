import pandas as pd 
import numpy as np 
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Logging.logger import get_logger
import plotly.graph_objects as go
import json
import re
from plotly.subplots import make_subplots


# For Configuration of file
with open(r'Z:\Supply-Chain_management(SCM)\Source\config.json', "r") as f:
    config = json.load(f)

#Logging initiated
logger = get_logger("evalution-logger")

def interactive_evalution(data,forecast_future,prophet_data_pred,State):
    
    # Subplots (Graph and Table)
    fig = make_subplots(
        rows = 2, cols = 1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        row_heights=[0.75, 0.25],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=[
            f'Monthly Sales Forecast - {State} - 2025',
            'Future Prediction']
    )

    # Plot Actual Data
    fig.add_trace(
        go.Scatter(
            x = data['ds'],
            y = data['y'],
            mode = 'lines+markers',
            name = 'Actual',
            line = dict(color = 'purple', width = 2),
            marker = dict(size = 6)
        ), row = 1, col = 1)
    

    # Plot Prediction of data
    fig.add_trace(
        go.Scatter(
            x = prophet_data_pred['ds'],
            y = prophet_data_pred['yhat'],
            mode = 'lines+markers',
            name = 'Train Forecast',
            line = dict(color = 'lightblue', width = 2),
            marker = dict(size = 6)
        ), row = 1, col = 1)
    

    # Plot Future Forecast 
    fig.add_trace(
        go.Scatter(
            x = forecast_future['ds'][-4:],
            y = forecast_future['yhat'][-4:],
            mode = 'lines+markers',
            name = 'Forecast',
            line = dict(color = 'orange', width = 2)
        ), row = 1, col = 1)
    
    # PLot Forecat Table 
    fig.add_trace(
        go.Table(
            header=dict(
                values = ['Forecast Month', "Predicted QTY (MT)", "Lower Limit", "Upper Limit"],
                fill_color = 'lightblue',
                align = 'center'
            ),
            cells=dict(
                values = [
                    forecast_future['ds'][-4:].dt.strftime('%B-%Y'),
                    forecast_future['yhat'][-4:].round(2),
                    forecast_future['yhat_lower'][-4:].round(2),
                    forecast_future['yhat_upper'][-4:].round(2)
                ],
                fill_color = 'lavender',
                align = 'center'
            )
        ), row = 2, col = 1)

    # Layout And Export
    fig.update_layout(
        xaxis_title = 'Month',
        yaxis_title = "QTY (MT)",
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=60),
        height=700

    )

    fig.write_html(config[f'Graph_{State}'])
    

    
