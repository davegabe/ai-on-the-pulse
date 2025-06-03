from io import BytesIO
import os
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

from athome.data.ATHOME.retrieve import PATIENTS

"""

    # # 1. Pressure Section (Systolic and Diastolic)
    # plt.figure(figsize=(14, 6))
    # plt.plot(df.index, df['systolic'], label='Systolic', color=colors[0])
    # plt.plot(df.index, df['diastolic'], label='Diastolic', color=colors[1])
    # plt.title('Blood Pressure over Time', fontsize=12)
    # plt.ylabel('Blood Pressure (mmHg)', fontsize=10)
    # plt.legend(loc='upper right')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('blood_pressure.pdf')
    # plt.show()

    # # 2. Breath/Oxygen Section (Separate Subplots)
    # plt.figure(figsize=(14, 8))

    # # Subplot 1: Breath Rate
    # plt.subplot(2, 1, 1)
    # plt.plot(df.index, df['breath'], label='Breath Rate', color=colors[2])
    # plt.title('Breath Rate over Time', fontsize=12)
    # plt.ylabel('Breath Rate (bpm)', fontsize=10)
    # plt.legend(loc='upper right')

    # # Subplot 2: SpO2
    # plt.subplot(2, 1, 2)
    # plt.plot(df.index, df['spo2'], label='SpO2', color=colors[3])
    # plt.title('Oxygen Saturation (SpO2) over Time', fontsize=12)
    # plt.ylabel('SpO2 (%)', fontsize=10)
    # plt.xlabel('Timestamp', fontsize=10)
    # plt.legend(loc='upper right')

    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('breath_oxygen.pdf')
    # plt.show()

    # # 3. Heart Section (Separate Subplots)
    # plt.figure(figsize=(14, 12))

    # # Subplot 1: Pulse Rate
    # plt.subplot(3, 1, 1)
    # plt.plot(df.index, df['pulse'], label='Pulse Rate', color=colors[4])
    # plt.title('Pulse Rate over Time', fontsize=12)
    # plt.ylabel('Pulse Rate (bpm)', fontsize=10)
    # plt.legend(loc='upper right')

    # # Subplot 2: HRV
    # plt.subplot(3, 1, 2)
    # plt.plot(df.index, df['hrv'], label='Heart Rate Variability (HRV)', color=colors[5])
    # plt.title('Heart Rate Variability (HRV) over Time', fontsize=12)
    # plt.ylabel('HRV', fontsize=10)
    # plt.legend(loc='upper right')

    # # Subplot 3: CVRR
    # plt.subplot(3, 1, 3)
    # plt.plot(df.index, df['cvrr'], label='CVRR', color=colors[6])
    # plt.title('CVRR over Time', fontsize=12)
    # plt.ylabel('CVRR', fontsize=10)
    # plt.xlabel('Timestamp', fontsize=10)
    # plt.legend(loc='upper right')

    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('heart.pdf')
    # plt.show()

    # # 4. Steps Section
    # plt.figure(figsize=(14, 4))
    # plt.plot(df.index, df['steps'], label='Steps', color=colors[7])
    # plt.title('Steps over Time', fontsize=12)
    # plt.ylabel('Steps', fontsize=10)
    # plt.xlabel('Timestamp', fontsize=10)
    # plt.legend(loc='upper right')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('steps.pdf')
    # plt.show()

    # # Plotting Environmental Variables
    # fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    # # Temperature
    # df['temperature_y'].plot(ax=axes[0,0], title='Temperature (°C)')
    # axes[0,0].set_ylabel('°C')

    # # Humidity
    # df['humidity'].plot(ax=axes[0,1], title='Humidity (%)')
    # axes[0,1].set_ylabel('%')

    # # CO2
    # df['co2'].plot(ax=axes[1,0], title='CO2 Concentration (ppm)')
    # axes[1,0].set_ylabel('ppm')

    # # Pressure
    # df['pressure'].plot(ax=axes[1,1], title='Atmospheric Pressure (hPa)')
    # axes[1,1].set_ylabel('hPa')

    # # TVOC
    # df['tvoc'].plot(ax=axes[2,0], title='Total VOC (ppb)')
    # axes[2,0].set_ylabel('ppb')

    # # Luminosity
    # df['luminosity'].plot(ax=axes[2,1], title='Luminosity (lux)')
    # axes[2,1].set_ylabel('lux')

    # plt.tight_layout()
    # plt.savefig('environmental_variables.pdf')
    # plt.show()
"""

def visualize_location(df: pd.DataFrame) -> BytesIO:
    """
    Visualize the location data for a given patient.

    Args:
        df (pd.DataFrame): The data for the patient.
    
    Returns:
        BytesIO: The image buffer containing the plot.
    """
    # Set timestamp as index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    # Define activity columns
    activity_cols = ['bagno', 'stanza1', 'stanza2', 'stanza3', 'stanza4', 'outside']
    activity_cols = [col for col in activity_cols if col in df.columns]

    # Ensure that numeric columns are of correct dtype
    numeric_cols = ['systolic', 'diastolic', 'spo2', 'hrv', 'cvrr', 'breath', 'pulse', 
                    'steps', 'temperature', 'humidity', 'co2', 'pressure', 'tvoc', 'luminosity', 
                    'time', 'weekday', 'day'] + activity_cols
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Set thin lines for plotting
    plt.rc('lines', linewidth=0.5)
    
    # Plotting Activity Indicators
    activity_df = df[activity_cols]
    activity_df["N/A"] = (activity_df.sum(axis=1) == 0).astype(int)
    
    # Determine active room for each timestamp
    activity_df['room'] = activity_df.idxmax(axis=1)
    resampled = activity_df[['room']]
    
    # Convert room labels to numeric values for plotting
    room_mapping = resampled['room'].unique()
    room_mapping = {room: i for i, room in enumerate(room_mapping)}
    
    resampled['room'] = resampled['room'].map(room_mapping)
    
    # Extract date and time components
    resampled['date'] = resampled.index.date
    resampled['time'] = resampled.index.time
    
    # Pivot data for heatmap
    pivot_df = resampled.pivot(index='date', columns='time', values='room')
    
    # Define colormap and boundaries
    cmap = ListedColormap(sns.color_palette("tab10", n_colors=len(room_mapping)))
    boundaries = np.arange(-0.5, len(room_mapping) + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)
    
    # Create the plot, height is adjusted based on the number of days
    height = 0.5 + 0.3 * len(pivot_df.index.unique())
    plt.figure(figsize=(14, height))

    # Plot heatmap
    ax = sns.heatmap(pivot_df, cmap=cmap, norm=norm, cbar=False, linewidths=0)
    
    # Add horizontal white lines
    for y in range(1, pivot_df.shape[0]):
        ax.axhline(y, color='white', linewidth=2)
    
    # Adjust the rest of the plot
    plt.xticks(np.arange(0, len(pivot_df.columns), 30), pivot_df.columns[::30], rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.grid(color='lightgrey', linestyle='-', linewidth=1)
    plt.title('Room Activity over Time')
    plt.xlabel('Time of Day')
    plt.ylabel('Date')
    
    # Add legend
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor='black', label=room) for i, room in enumerate(room_mapping.keys())
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save the image to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()    
    return img_buf

if __name__ == '__main__':
    # Get script directory
    script_dir = os.path.dirname(__file__)
    os.makedirs('plots', exist_ok=True)

    # Visualize the location data for each patient
    for patient in range(len(PATIENTS)):
        # Read the CSV file
        path = os.path.join(script_dir, f'processed_{patient}/all_data.csv')
        df = pd.read_csv(path, parse_dates=['timestamp'])
        # Visualize the data
        img_buf = visualize_location(df)
        # Convert to PIL Image
        img_buf.seek(0)
        img = Image.open(img_buf)
        # Save the plot
        img.save(f'plots/room_activity_{patient}.png')