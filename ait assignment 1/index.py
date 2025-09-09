"""
#program : lab1.py
#author  : Aavash Gurung
#date    : 2025-09-09
#purpose : Data analysis and reading data in Python
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

path = "."

filenames = os.listdir(path)
print("Files in directory:", filenames)

df = pd.read_csv("price.csv", index_col=False, header=None)

prices = np.array(df[0]) 
print("Original Prices:\n", prices)

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(prices), 1), prices, label="Original Prices")
plt.title("Original Price Data")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Compute mean (µ) and standard deviation (σ)
mu = np.mean(prices)
sigma = np.std(prices)
print("Mean (µ):", mu, " Standard Deviation (σ):", sigma)

# Step 8: Clean the data (clip values within µ ± σ)
cleaned_prices = []
for p in prices:
    if p > mu + sigma:
        cleaned_prices.append(mu + sigma)
    elif p < mu - sigma:
        cleaned_prices.append(mu - sigma)
    else:
        cleaned_prices.append(p)

cleaned_prices = np.array(cleaned_prices)

plt.figure(figsize=(10,5))
plt.plot(np.arange(0, len(cleaned_prices), 1), cleaned_prices, label="Cleaned Prices", color="red")
plt.title("Cleaned Price Data (within µ ± σ)")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
