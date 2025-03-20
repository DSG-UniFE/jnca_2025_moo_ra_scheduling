import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

matplotlib.use('TkAgg')
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load data from CSV file
df_smartcity = pd.read_csv("../results/usecase/smartcity50/f1_f2_f3/results_usecase_smartcity50.csv")
df_ai = pd.read_csv("../results/usecase/ai50/f1_f2_f3/results_usecase_ai50.csv")
df_iiot = pd.read_csv("../results/usecase/iiot50/f1_f2_f3/results_usecase_iiot50.csv")
df_vr = pd.read_csv("../results/usecase/vr50/f1_f2_f3/results_usecase_vr50.csv")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(df_smartcity["MIP_Gap"], df_smartcity["Execution_Time"], marker='o', label="Smart City")
plt.plot(df_ai ["MIP_Gap"], df_ai ["Execution_Time"], marker='s', label="GenAI Inference")
plt.plot(df_iiot["MIP_Gap"], df_iiot["Execution_Time"], marker='^', label="Industrial IoT")
plt.plot(df_vr["MIP_Gap"], df_vr["Execution_Time"], marker='^', label="Virtual Reality")

# Log scale for y-axis
plt.yscale("log")

# Labels and Title
plt.xlabel(r'\textbf{MIP Gap (between 0 and 1)}', fontsize=11)
plt.ylabel(r'\textbf{Time (log scale)}', fontsize=11)
plt.legend()
plt.grid(True, which="major", linestyle="--", linewidth=0.5)  # Only major grid lines

# Show plot
plt.savefig('execution_time_50.pdf', dpi=250, bbox_inches='tight')

'''
# Load data from CSV file
df_smartcity = pd.read_csv("results/usecase/smartcity/f1_f2_f3/benchmark_results_usecase_smartcity.csv")
df_ai = pd.read_csv("results/usecase/ai/f1_f2_f3/benchmark_results_usecase_ai.csv")
df_iiot = pd.read_csv("results/usecase/iiot/f1_f2_f3/benchmark_results_usecase_iiot.csv")
df_vr = pd.read_csv("results/usecase/vr/f1_f2_f3/benchmark_results_usecase_vr.csv")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(df_smartcity["MIP_Gap"], df_smartcity["Mean_Time_ms"], marker='o', label="Smart City")
plt.plot(df_ai ["MIP_Gap"], df_ai ["Mean_Time_ms"], marker='s', label="GenAI Inference")
plt.plot(df_iiot["MIP_Gap"], df_iiot["Mean_Time_ms"], marker='^', label="Industrial IoT")
plt.plot(df_vr["MIP_Gap"], df_vr["Mean_Time_ms"], marker='^', label="Virtual Reality")

# Log scale for y-axis
plt.yscale("log")

# Labels and Title
plt.xlabel(r'\textbf{MIP Gap (between 0 and 1)}', fontsize=11)
plt.ylabel(r'\textbf{Time (log scale)}', fontsize=11)
plt.legend()
plt.grid(True, which="major", linestyle="--", linewidth=0.5)  # Only major grid lines

# Show plot
plt.savefig('benchmark_execution_time.pdf', dpi=250, bbox_inches='tight')


# Load data from CSV file
df_ai = pd.read_csv("results/usecase/ai/f1_f2_f3/results_usecase_ai.csv")
df_ai50 = pd.read_csv("results/usecase/ai50/f1_f2_f3/results_usecase_ai50.csv")
df_ai100 = pd.read_csv("results/usecase/ai100/f1_f2_f3/results_usecase_ai100.csv")


# Plot
plt.figure(figsize=(6, 4))
plt.plot(df_ai["MIP_Gap"], df_ai["Execution_Time"], marker='o', label="GenAI Inference")
plt.plot(df_ai50["MIP_Gap"], df_ai50["Execution_Time"], marker='s', label="GenAI Inference - 50 requests")
plt.plot(df_ai100["MIP_Gap"], df_ai100["Execution_Time"], marker='^', label="GenAI Inference - 100 requests")

# Log scale for y-axis
plt.yscale("log")

# Labels and Title
plt.xlabel(r'\textbf{MIP Gap (between 0 and 1)}', fontsize=11)
plt.ylabel(r'\textbf{Time (log scale)}', fontsize=11)
plt.legend()
plt.grid(True, which="major", linestyle="--", linewidth=0.5)  # Only major grid lines

# Show plot
plt.savefig('execution_time_ai_only.pdf', dpi=250, bbox_inches='tight')
'''
