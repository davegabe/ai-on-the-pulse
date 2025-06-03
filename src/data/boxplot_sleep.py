import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Set seaborn style for professional, paper-ready plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

# Load data from CSV files
df_patient = pd.read_csv('ATHOME/processed_0/sleep_features.csv')
df_pilot = pd.read_csv('ATHOME/processed_1/sleep_features.csv')

# Filter out sleep features with sleep duration less than 4 hours
df_patient = df_patient[df_patient['sleep_duration'] >= 240]
df_pilot = df_pilot[df_pilot['sleep_duration'] >= 240]

# Remove nan values
df_patient = df_patient.dropna()
df_pilot = df_pilot.dropna()

# Select key sleep features to analyze
sleep_features = [
    'sleep_duration',
    'deep_sleep_ratio',
    'sleep_onset',
    'wake_up'
]

# Define proper y-labels with units
y_labels = {
    'sleep_duration': 'Sleep Duration (min)',
    'deep_sleep_ratio': 'Deep Sleep Ratio',
    'sleep_onset': 'Sleep Onset (time of day)',
    'wake_up': 'Wake Up Time (time of day)'
}

# Define better formatted titles for subplots
formatted_titles = {
    'sleep_duration': 'Sleep Duration',
    'deep_sleep_ratio': 'Deep Sleep Ratio',
    'sleep_onset': 'Sleep Onset',
    'wake_up': 'Wake Up Time'
}

# Function to format time values (minutes from midnight) as HH:MM
def time_formatter(x, pos):
    # Convert minutes to hours and minutes
    hours = int(x // 60)
    minutes = int(x % 60)
    
    # Handle hours that go beyond 24 (wrap around)
    hours = hours % 24
    
    return f"{hours:02d}:{minutes:02d}"

# Prepare data for seaborn by restructuring into long format
df_list = []

# Add patient data
for feature in sleep_features:
    for value in df_patient[feature]:
        df_list.append({
            'Parameter': feature,
            'Value': value,
            'Group': 'Patient'
        })

# Add pilot data
for feature in sleep_features:
    for value in df_pilot[feature]:
        df_list.append({
            'Parameter': feature,
            'Value': value,
            'Group': 'Pilot'
        })

# Create DataFrame for plotting
df_combined = pd.DataFrame(df_list)

# Calculate mean and standard deviation for each group and parameter
mean_std_data = {}
for group in ['Patient', 'Pilot']:
    mean_std_data[group] = {"Mean": [], "Std": []}
    for feature in sleep_features:
        if group == 'Patient':
            data = df_patient[feature]
        else:
            data = df_pilot[feature]
        
        mean_std_data[group]["Mean"].append(data.mean())
        mean_std_data[group]["Std"].append(data.std())

# Create figure with subplots
# With only 4 features, we can use a 2x2 layout
num_features = len(sleep_features)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes = axes.flatten()

# Color palette for a professional look - colorblind friendly
colors = sns.color_palette("colorblind")
palette = {'Patient': colors[0], 'Pilot': colors[1]}

# Plot each parameter
for i, feature in enumerate(sleep_features):
    ax = axes[i]
    feature_data = df_combined[df_combined['Parameter'] == feature]
    
    # Create seaborn boxplot with jittered points
    sns.boxplot(
        x='Group', 
        y='Value', 
        data=feature_data,
        ax=ax,
        palette=palette,
        width=0.5,
        fliersize=0
    )
    
    # Add individual data points
    sns.stripplot(
        x='Group', 
        y='Value', 
        data=feature_data,
        ax=ax,
        palette=palette,
        size=5,
        jitter=True,
        alpha=0.7,
        edgecolor='gray',
        linewidth=0.5
    )
    
    # Add mean and std error bars
    for j, group in enumerate(['Patient', 'Pilot']):
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
    ax.set_title(formatted_titles[feature], fontsize=12)
    ax.set_xlabel('')
    # Use the proper y-label from our dictionary
    ax.set_ylabel(y_labels[feature], fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # If the feature is a time-of-day feature, set a custom formatter on the y-axis
    if feature in ["wake_up", "sleep_onset"]:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    
    # Set bottom limit to 0 for non-negative parameters (except for time features)
    if feature not in ['sleep_onset', 'wake_up']:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

# Hide empty subplots if any
for i in range(num_features, len(axes)):
    axes[i].set_visible(False)

# Add a unified legend
handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['Patient'], markersize=10, 
               label="Patient"),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['Pilot'], markersize=10, 
               label="Pilot"),
    plt.Line2D([0], [0], marker='o', color='black', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
               label="Mean ± Std")
]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=True)

# Adjust spacing
plt.tight_layout()
plt.subplots_adjust(top=0.83)

# Save as PDF in plots folder
pdf_filename = "plots/sleep_features_comparison"
plt.savefig(f"{pdf_filename}.pdf", format="pdf", bbox_inches="tight")
plt.savefig(f"{pdf_filename}.svg", format="svg", bbox_inches="tight")

print(f"Plot saved to {pdf_filename}")