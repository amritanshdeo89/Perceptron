from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, modelName, plotName, eta, epochs):

    """
      :param data: dataset
      :param modelName: name of the model
      :param plotName: name of the plot
      :param eta: values between 0 to 1
      :param epochs: number of epochs

    """

   
    df = pd.DataFrame(data)

    print(df)

    X,y = prepare_data(df)

    
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    save_model(model, filename=modelName)
    save_plot(df,plotName,model )


if __name__ == '__main__':
     AND = {
        "X1" : [0,0,1,1],
        "X2" : [0,1,0,1],
        "y" : [0,0,0,1]
    }

     ETA = 0.3 # 0 to 1
     EPOCHS = 10

     main(data= AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)