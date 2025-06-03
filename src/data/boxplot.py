import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Set seaborn style for professional, paper-ready plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Data for Production (Patient 1)
data = {
    "Systolic": [101, 114, 116, 118, 139],
    "Diastolic": [67, 75, 77, 78, 94],
    "SpO2": [76, 98, 98, 98, 98],
    "HRV": [37, 39, 41, 44, 46],
    "CVRR": [2, 3, 4, 5, 6],
    "Resp. Rate": [7, 13, 14, 16, 35],
    "HR": [34, 67, 72, 78, 172],
    "Steps": [0, 0, 18, 136, 5290],
}

# Data for Pilot (Patient 2)
data_patient_2 = {
    "Systolic": [123, 127, 128, 133, 145],
    "Diastolic": [83, 88, 89, 90, 104],
    "SpO2": [78, 97, 98, 98, 98],
    "HRV": [37, 39, 42, 44, 46],
    "CVRR": [2, 3, 4, 5, 6],
    "Resp. Rate": [7, 13, 16, 17, 33],
    "HR": [37, 64, 78, 82, 162],
    "Steps": [0, 0, 1205, 5261, 27678],
}

# Mean and Standard Deviation data
mean_std_data = {
    "Production": {
        "Mean": [116.10, 76.65, 97.63, 41.51, 3.99, 14.58, 72.40, 141.61],
        "Std": [3.08, 2.01, 0.86, 2.87, 1.42, 1.68, 7.76, 480.70]
    },
    "Pilot": {
        "Mean": [129.59, 89.03, 97.10, 41.55, 3.97, 14.99, 74.32, 3207.51],
        "Std": [3.95, 1.51, 1.72, 2.89, 1.43, 2.42, 11.53, 4290.84]
    }
}

# Parameter labels
param_labels = list(data.keys())

# Define proper y-labels according to the requirements
y_labels = {
    "Systolic": "Systolic (mmHg)",
    "Diastolic": "Diastolic (mmHg)",
    "Resp. Rate": "Breaths/min",
    "SpO2": "SpO2 (%)",
    "HR": "Heart Rate (bpm)",
    "HRV": "RMSSD (ms)",
    "CVRR": "CVRR",
    "Steps": "Total Steps"
}

# Prepare data for seaborn by restructuring into long format
df_list = []

for param in param_labels:
    # Production data
    for value in data[param]:
        df_list.append({
            'Parameter': param,
            'Value': value,
            'Group': 'Production'
        })
    
    # Pilot data
    for value in data_patient_2[param]:
        df_list.append({
            'Parameter': param,
            'Value': value,
            'Group': 'Pilot'
        })

df = pd.DataFrame(df_list)

# Create figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()

# Color palette for a professional look - colorblind friendly
colors = sns.color_palette("colorblind")
palette = {'Production': colors[0], 'Pilot': colors[1]}

# Plot each parameter
for i, param in enumerate(param_labels):
    ax = axes[i]
    param_data = df[df['Parameter'] == param]
    
    # Apply symlog scale for Steps
    if param == "Steps":
        ax.set_yscale('symlog', linthresh=1)
    
    # Create seaborn boxplot with jittered points
    sns.boxplot(
        x='Group', 
        y='Value', 
        data=param_data,
        ax=ax,
        palette=palette,
        width=0.5,
        fliersize=0
    )
    
    # Add individual data points
    sns.stripplot(
        x='Group', 
        y='Value', 
        data=param_data,
        ax=ax,
        palette=palette,
        size=5,
        jitter=True,
        alpha=0.7,
        edgecolor='gray',
        linewidth=0.5
    )
    
    # Add mean and std error bars
    for j, group in enumerate(['Production', 'Pilot']):
        ax.errorbar(
            j, 
            mean_std_data[group]["Mean"][i], 
            yerr=mean_std_data[group]["Std"][i],
            fmt='o', 
            color='black',
            ecolor='black',
            elinewidth=1.5,
            capsize=5,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.5,
            zorder=10
        )
    
    # Customize appearance
    ax.set_title(param, fontsize=12)
    ax.set_xlabel('')
    # Use the proper y-label from our dictionary
    ax.set_ylabel(y_labels[param], fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set bottom limit to 0 for non-negative parameters
    if param in ["SpO2", "HRV", "CVRR", "Resp. Rate", "HR", "Steps"]:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

# Add a unified legend
handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['Production'], markersize=10, 
               label="Production"),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['Pilot'], markersize=10, 
               label="Pilot"),
    plt.Line2D([0], [0], marker='o', color='black', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
               label="Mean ± Std")
]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=True)

# Adjust spacing
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save as PDF in plots folder
pdf_filename = "plots/pilot_vs_production_stats"
plt.savefig(f"{pdf_filename}.pdf", format="pdf", bbox_inches="tight")
plt.savefig(f"{pdf_filename}.svg", format="svg", bbox_inches="tight")

print(f"Plot saved to {pdf_filename}")