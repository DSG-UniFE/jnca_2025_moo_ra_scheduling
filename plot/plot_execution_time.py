import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

matplotlib.use('TkAgg')
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load data from CSV file
usecase = "smartcity50"
file_path = "../results/usecase/" + usecase + "/f1_f2_f3/results_usecase_" + usecase + ".csv"  # Change to your file path
df = pd.read_csv(file_path)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(df["MIP_Gap"], df["Load_Time"], marker='o', label="Load Time")
plt.plot(df["MIP_Gap"], df["Creation_time"], marker='s', label="Creation Time")
plt.plot(df["MIP_Gap"], df["Execution_Time"], marker='^', label="Execution Time")

# Log scale for y-axis
plt.yscale("log")

# Labels and Title
plt.xlabel(r'\textbf{MIP Gap (between 0 and 1)}', fontsize=11)
plt.ylabel(r'\textbf{Time (log scale)}', fontsize=11)
plt.legend()
plt.grid(True, which="major", linestyle="--", linewidth=0.5)  # Only major grid lines

# Show plot
plt.savefig('execution_time_' + usecase + '.pdf', dpi=250, bbox_inches='tight')
