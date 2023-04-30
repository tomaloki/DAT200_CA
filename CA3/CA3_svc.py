# %%

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as m


# Load data
water_train = pd.read_csv('assets/train.csv', delimiter=',')

# Visualizing the data in a table
water_train

# Removing the column "Unnamed: 0", appears to be an extra index-volumn
# with no relevant information
water_train = water_train.iloc[:, 1:11]

# Looking at the descriptive statistics
descr_stats = water_train.describe()

# We can see that the scale differs for several of the features, because of the different mean
# and std's.

# Plotting the distribution of the data in a histogram to observe the distribution

# water_train.hist()
# plt.show()

# All the features are normally distributed, but because of the different scale we
# will perform standardization on the data set.

#
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=water_train, ax=ax)



# Extracting data columns used for predicting, and target variable which we
# want to predict
X_train = water_train.iloc[:, :9]
y_train = water_train.iloc[:, 9]
