from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os


logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")


def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)


if __name__ == '__main__':
    OR = {
        "X1": [0,0,1,1],
        "X2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 100
    try:
        logging.info(">>>>> starting training >>>>>")
        main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<<< training done successfully<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e

