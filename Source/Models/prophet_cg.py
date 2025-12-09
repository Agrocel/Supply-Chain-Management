import pandas as pd 
import numpy as np
import os 
import json 
from Logging.logger import get_logger
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.serialize import model_to_json


Base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(Base,"..","config.json"),"r") as f:
    config = json.load(f)

logger = get_logger("Prophet_Chattisgarh")

def train_prophet_model_cg(data):
    try:
        logger.info("Training Prophet Model for Chhattisgarh")
        prophet_cg_data = data[["Date", "QTY_MT"]].rename(columns = {"Date":"ds","QTY_MT":"y"}).copy()
    except Exception as e:
        logger.error(f"Error in Training Prophet Model for Chhattisgarh :{e}",exc_info=True)
        raise ValueError(f"Error in Training Prophet Model for Chhattisgarh")
    
    try:
        logger.info("Fitting Data to the model......")
        model = Prophet(
            changepoint_prior_scale=0.1,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=10.0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(prophet_cg_data)
        logger.info("Model Training Complete\n")
    except Exception as e:
        logger.error(f"Error in Fitting Data to the model :{e}",exc_info=True)
        raise ValueError(f"Error in Fitting Data to the model")
    
    try:
        logger.info("Forecasting Test and Future Date")
        future = model.make_future_dataframe(periods=3, freq = "MS")
        forecast_future = model.predict(future)
        prophet_cg_data_pred = model.predict(prophet_cg_data)
        logger.info("Model Forecasting Complete\n")
        
        logger.info("Calculating MAE and RMSE for Test.....")
        MAE = mean_absolute_error(prophet_cg_data['y'], prophet_cg_data_pred['yhat'])
        RMSE = np.sqrt(mean_squared_error(prophet_cg_data['y'], prophet_cg_data_pred['yhat']))
        logger.info("MAE and RMSE calculated\n")
        logger.info(f'Value of MAE for Prophet is {MAE:.2f}')
        logger.info(f'Value of RMSE for Prophet is {RMSE:.2f}')
        
    except Exception as e:
        logger.error(f"Error in Forecasting Test and Future Date :{e}",exc_info=True)
        raise ValueError(f"Error in Forecasting Test and Future Date")
    
    try:
        logger.info("Saving Model")
        with open(os.path.join(Base, "..", "..", config['Prophet_model_Chattisgarh']), 'w') as fout:
            fout.write(model_to_json(model))
        logger.info("Model Saved")

        logger.info("Creating Report for Prophet Chhattisgarh model")

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_model_error = pd.DataFrame({
            "Date" : [timestamp],
            "Model" : ['Prophet_Chattisgarh'],
            "MAE" : [MAE],
            "RMSE" : [RMSE]
        })
        save_model_error.to_csv(config[f'model_evaluation_Chattisgarh'], mode = 'a', index = False, header=not pd.io.common.file_exists(config[f'model_evaluation_Chattisgarh']))

        save_model_performance = pd.DataFrame({
            "Date_forecast": prophet_cg_data_pred['ds'],
            "Forecast": prophet_cg_data_pred['yhat'],
            "Lower Bound": prophet_cg_data_pred['yhat_lower'],
            "Upper Bound": prophet_cg_data_pred['yhat_upper']
        })
        save_model_performance.to_csv(config[f'model_forecast_Chattisgarh'])
        logger.info("Report Generated")
        
    except Exception as e:
        logger.error(f"Error in Saving Model :{e}",exc_info=True)
        raise ValueError(f"Error in Saving Model")
    
    return prophet_cg_data,forecast_future,prophet_cg_data_pred

