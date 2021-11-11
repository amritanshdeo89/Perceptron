from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

df

X,y = prepare_data(df)

ETA = 0.3 # 0 to 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X,y)