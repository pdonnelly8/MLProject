import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'datasetWIP.csv'
diagnosis = np.array(tf.io.gfile.listdir("AudioFiles"))
print('Diagnosis:', diagnosis)
