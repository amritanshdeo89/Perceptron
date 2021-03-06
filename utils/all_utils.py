import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib ## for saving the file
from matplotlib.colors import ListedColormap
import os
import logging



def prepare_data(df):

  """it is used to separate the dependent and independent variables

  Args:
      df (pd.DataFrame): its  a dataset   

  Returns:
     tuple: it returns the tuple of dependent and independent variables
  """

  logging.info('Preparing the data by segrating the dependent and independent variables')
  X = df.drop("y", axis=1)
  y = df["y"]

  return X, y




def save_model(model, filename):

  """This saves the trainde model

  Args:
      model (python object): trained model
      filename (str): Path to save the trained model
  """

  logging.info('Saving the trained model')
  model_dir = 'models'
  os.makedirs(model_dir, exist_ok = True) ## if model directory doesnot exist
  filePath = os.path.join(model_dir, filename) ## model/filename
  joblib.dump(model, filePath)
  logging.info(f"saved the model at {filePath}")


def save_plot(df, file_name, model):

  """
    :param df : this a dataset
    :param file_name : its a path to save the file to
    :param model : trained model
  """

  def _create_base_plot(df):

    """
      :param df : this a dataset
    """
    logging.info('creating the base plot')
    df.plot(kind="scatter", x="X1", y="X2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):

    """
      :param X: input
      :param y: input
      :param classfier: model class
      :param resolution: resolution, Defaults to 0.02.
    """

    logging.info('Plotting the decision region')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
  logging.info(f'Saving the plot at {plotPath}')