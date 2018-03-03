import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

get_ipython().magic('matplotlib inline')
np.random.seed(1)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.
#
# <img src="images/SIGNS.png" style="width:800px;height:300px;">
#
# The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples.

# In[5]:

# Example of a picture
index = 10
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:, index])))
