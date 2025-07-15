import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Logging.logger import get_logger
import plotly.graph_objects as go
import json


with open("config.json", "r") as f:
    config = json.load(f)


logger = get_logger("evalution-logger")

def model_evaluation(train,test,y_true, y_pred, model_name):
    logger.info("Calculating MAE and RMSE....")

    MAE = mean_absolute_error(y_true['QTY_MT'], y_pred['yhat'])
    RMSE = np.sqrt(mean_squared_error(y_true['QTY_MT'], y_pred['yhat']))
    
    logger.info("MAE and RMSE calculated\n")
    logger.info(f'✅Value of MAE for {model_name} is {MAE:.2f}')
    logger.info(f'✅Value of RMSE for {model_name} is {RMSE:.2f}')

    return MAE, RMSE


def plot_plotly(train,test,MAE,RMSE,model_name,forecast_future,forecast_train):
    fig = go.Figure()

    # Plot train Data
    fig.add_trace(go.Scatter(
        x=train['ds'], y = train['y'],
        mode = 'lines+markers',
        name = 'Train',
        line = dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x = forecast_train['ds'], y = forecast_train['yhat'],
        mode = 'lines+markers',
        name = 'Forecast Train',
        line = dict(color='green')
    ))

    #Plot Test Data
    fig.add_trace(go.Scatter(
        x=test['ds'], y = test['y'],
        mode = 'lines+markers',
        name = 'Test',
        line = dict(color='purple')
    ))
    

    # Plot Forecast / Prediction
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', dash='dot')
    ))

    # Annotate MAE and RMSE
    fig.add_annotation(
        x=0.01, y=0.99,
        xref="paper", yref="paper",
        text=f"<b>{model_name} Evaluation</b><br>MAE: {MAE:.2f}<br>RMSE: {RMSE:.2f}",
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="lightyellow",
        font=dict(size=12)
    )

    # Layout settings
    fig.update_layout(
        title=f"{model_name} Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title="QTY (MT)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=600
    )

    fig.write_html(config['Figure'] + f"/{model_name.lower()}_actual_vs_forecast.html")

