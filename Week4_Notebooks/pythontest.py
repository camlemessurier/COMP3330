import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
data = pd.read_excel('./data/DryBeanDataset/Dry_Bean_Dataset.xlsx')
print(data.head())
