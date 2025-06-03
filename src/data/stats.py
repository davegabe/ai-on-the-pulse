import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

def main(filename: str, K: int, show_healthy_ranges: bool = True):
    """
    Perform data analysis on a CSV file, extract a contiguous K-hour block with no NaN values,
    and plot all variable groups in separate subplots with publication-quality formatting.
    
    Args:
        filename (str): Path to the CSV file to analyze
        K (int): Number of contiguous hours to plot
        show_healthy_ranges (bool): Whether to display healthy range references on plots
    """
    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # 1. Load CSV & Compute Statistics
    # -------------------------------
    df = pd.read_csv(filename, parse_dates=['timestamp'])
    
    print("Dataset Preview:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())
    
    print("\nBasic Descriptive Statistics (Numeric Columns):")
    print(df.describe())
    
    numeric_columns = df.select_dtypes(include='number').columns
    print("\nAdditional Statistics per Numeric Column:")
    for col in numeric_columns:
        median_val = df[col].median()
        mode_val = df[col].mode().tolist()
        missing_count = df[col].isnull().sum()
        print(f"\nColumn '{col}':")
        print(f"  Median: {median_val}")
        print(f"  Mode: {mode_val}")
        print(f"  Missing values: {missing_count}")
    
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    
    # -------------------------------
    # 2. Extract a Contiguous K-Hour Block with Complete Data
    # -------------------------------
    df_complete = df.dropna().sort_values('timestamp')
    
    if df_complete.empty:
        print("No complete data available to plot.")
        return
    
    # Calculate time differences between consecutive entries
    time_diffs = df_complete['timestamp'].diff().dt.total_seconds().div(60)  # in minutes
    
    # Determine the typical sampling interval (mode of time differences)
    if len(time_diffs) >= 2:
        mode_diff = time_diffs.iloc[1:].mode()
        mode_diff = mode_diff[0] if not mode_diff.empty else 5
    else:
        mode_diff = 5  # default if insufficient data
    
    # Find contiguous blocks of at least K hours starting at hour:00
    selected_group = None
    hour = 11 # Start at exact 11:00 AM
    for i, row in df_complete.iterrows():
        if row['timestamp'].hour == hour:
            # Check if a contiguous block of at least K hours is available
            block_end = row['timestamp'] + pd.Timedelta(hours=K)
            block = df_complete[(df_complete['timestamp'] >= row['timestamp']) & (df_complete['timestamp'] <= block_end)]
            if len(block) >= K * 60 / mode_diff:
                selected_group = block
                break
    
    if selected_group is None:
        print(f"No contiguous block of at least {K} hours found.")
        return
    
    # Extract first K hours from the selected group
    start_time = selected_group['timestamp'].iloc[0]
    end_time = start_time + pd.Timedelta(hours=K)
    df_k_hours = selected_group[(selected_group['timestamp'] >= start_time) & (selected_group['timestamp'] <= end_time)]
    
    if df_k_hours.empty:
        print(f"Not enough data in the contiguous block to extract {K} hours.")
        return
    
    # -------------------------------
    # 3. Publication-Quality Plotting
    # -------------------------------
    # Set scientific paper style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use a single color for all plots
    main_color = '#0072B2'  # A nice blue color that works well for scientific plots
    
    def format_x_axis(ax, data_length):
        """Format x-axis with adaptive time intervals based on data density"""
        # Dynamically set interval based on number of data points
        if data_length <= 10:
            interval = 1  # Every point
        elif data_length <= 30:
            interval = 2  # Every other point
        elif data_length <= 60:
            interval = 5  # Every 5 points
        elif data_length <= 120:
            interval = 10  # Every 10 points
        else:
            interval = 15  # Every 15 points
        
        # Calculate time interval in minutes based on data timespan
        timespan_minutes = (df_k_hours['timestamp'].iloc[-1] - df_k_hours['timestamp'].iloc[0]).total_seconds() / 60
        
        # Adjust interval for very long or very short time spans
        if timespan_minutes > 360:  # More than 6 hours
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif timespan_minutes > 120:  # More than 2 hours
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(5, int(timespan_minutes/10))))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Add minor ticks for better readability
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
        
        # Rotate labels and adjust their position
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Ensure starts from HH:00 or HH:15 or HH:30 or HH:45
        ax.set_xlim([df_k_hours['timestamp'].iloc[0].replace(minute=0),
                     df_k_hours['timestamp'].iloc[-1].replace(minute=0)])
    
    # Format date string in fancy format
    def fancy_date_format(dt):
        # Example: "March 10, 2025 • 14:30"
        return dt.strftime("%B %d, %Y • %H:%M")
    
    # Get timespan info for labeling
    start_fancy = fancy_date_format(start_time)
    end_fancy = fancy_date_format(end_time)
    period_str = f"Period: {start_fancy} to {end_fancy}"
    
    # Create a combined figure with 2-column grid layout
    # We now have 8 subplots (separating systolic and diastolic BP)
    figure_width = 18  # Wider figure for 2-column layout
    figure_height = 5
    fig_combined = plt.figure(figsize=(figure_width, figure_height))
    
    # Define a grid
    gs = fig_combined.add_gridspec(1, 5, hspace=0.2, wspace=0.2)
    
    # # Custom marker for all plots
    # marker = 'o'
    # markersize = 3
    # Remove markers for better visibility
    marker = None
    markersize = 0
    
    # Create all subplots
    # Column 1
    ax_systolic = fig_combined.add_subplot(gs[0, 0])
    ax_breath = fig_combined.add_subplot(gs[0, 1])
    ax_spo2 = fig_combined.add_subplot(gs[0, 2])
    ax_pulse = fig_combined.add_subplot(gs[0, 3])
    ax_diastolic = fig_combined.add_subplot(gs[0, 4])
    
    # Column 2
    # ax_hrv = fig_combined.add_subplot(gs[1, 1])
    # ax_cvrr = fig_combined.add_subplot(gs[2, 1])
    # ax_steps = fig_combined.add_subplot(gs[3, 1])
    
    # 1. Systolic Blood Pressure plot
    ax_systolic.plot(df_k_hours['timestamp'], df_k_hours['systolic'], 
                    color=main_color, marker=marker, markersize=markersize, label='Systolic')
    # ax_systolic.set_ylabel('Systolic (mmHg)', fontweight='bold')
    ax_systolic.set_title('Systolic Blood Pressure (mmHg)')
    
    if show_healthy_ranges:
        # Healthy range for systolic BP is 90-120 mmHg
        ax_systolic.axhspan(90, 120, color='lightgreen', alpha=0.2)
        legend_elements = [
            plt.Line2D([0], [0], color=main_color, marker=marker, markersize=5, label='Systolic BP'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.2, label='Healthy Range (90-120)')
        ]
        ax_systolic.legend(handles=legend_elements, loc='best', fontsize=8, 
                          frameon=True, fancybox=True, shadow=True)
        # Elevated or Hypertension stage 1 (120-140) and Hypotension (60-90)
        ax_systolic.axhspan(120, 140, color='yellow', alpha=0.2)
        ax_systolic.axhspan(60, 90, color='yellow', alpha=0.2)
        # Hypertension stage 2 (140 and above) and Severe Hypotension (below 60)
        ax_systolic.axhspan(140, 200, color='red', alpha=0.2)
        # ax_systolic.axhspan(0, 60, color='red', alpha=0.2)
        # Set y limit based on data but making also visible the start of the red zones
        ax_systolic.set_ylim(min(80, df_k_hours['systolic'].min() - 2), max(141, df_k_hours['systolic'].max() + 2))
    else:
        ax_systolic.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # 2. Diastolic Blood Pressure plot
    ax_diastolic.plot(df_k_hours['timestamp'], df_k_hours['diastolic'], 
                     color=main_color, marker=marker, markersize=markersize, label='Diastolic')
    # ax_diastolic.set_ylabel('Diastolic (mmHg)', fontweight='bold')
    ax_diastolic.set_title('Diastolic Blood Pressure (mmHg)')
    
    if show_healthy_ranges:
        # Healthy range for diastolic BP is 60-80 mmHg
        ax_diastolic.axhspan(60, 80, color='lightgreen', alpha=0.2)
        legend_elements = [
            plt.Line2D([0], [0], color=main_color, marker=marker, markersize=5, label='Diastolic BP'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.2, label='Healthy Range (60-80)')
        ]
        ax_diastolic.legend(handles=legend_elements, loc='best', fontsize=8, 
                           frameon=True, fancybox=True, shadow=True)
        # Yellow: Prehypertension (80-90) and prehypotension (50-60)
        ax_diastolic.axhspan(80, 90, color='yellow', alpha=0.2)
        ax_diastolic.axhspan(40, 60, color='yellow', alpha=0.2)
        # Unhealthy range for diastolic BP is 90 and above, or 50 and below
        ax_diastolic.axhspan(90, 150, color='red', alpha=0.2)
        # ax_diastolic.axhspan(0, 50, color='red', alpha=0.2)
        # Set y limit based on data but making also visible the start of the red zones
        ax_diastolic.set_ylim(min(56, df_k_hours['diastolic'].min() - 2), max(91, df_k_hours['diastolic'].max() + 2))
    else:
        ax_diastolic.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # 3. Breath Rate plot
    ax_breath.plot(df_k_hours['timestamp'], df_k_hours['breath'], 
                  color=main_color, marker=marker, markersize=markersize, label='Breath Rate')
    # ax_breath.set_ylabel('Breaths/min', fontweight='bold')
    ax_breath.set_title('Respiratory Rate (Breaths/min)')
    
    if show_healthy_ranges:
        # Normal range for breath rate is 12-20 breaths per minute
        ax_breath.axhspan(12, 20, color='lightgreen', alpha=0.2)
        legend_elements = [
            plt.Line2D([0], [0], color=main_color, marker=marker, markersize=5, label='Breath Rate'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.2, label='Normal Range (12-20)')
        ]
        ax_breath.legend(handles=legend_elements, loc='best', fontsize=8, 
                        frameon=True, fancybox=True, shadow=True)
        # Yellow: Tachypnea (20-24) and bradypnea (8-12)
        ax_breath.axhspan(20, 24, color='yellow', alpha=0.2)
        ax_breath.axhspan(8, 12, color='yellow', alpha=0.2)
        # Red: Tachypnea (30 and above) and bradypnea (8 and below)
        ax_breath.axhspan(24, 100, color='red', alpha=0.2)
        ax_breath.axhspan(0, 8, color='red', alpha=0.2)
        # Set y limit based on data but making also visible the start of the red zones
        ax_breath.set_ylim(min(8, df_k_hours['breath'].min() - 2), max(25, df_k_hours['breath'].max() + 2))
    else:
        ax_breath.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # 4. SpO2 plot
    ax_spo2.plot(df_k_hours['timestamp'], df_k_hours['spo2'], 
                color=main_color, marker=marker, markersize=markersize, label='SpO2')
    # ax_spo2.set_ylabel('SpO2 (%)', fontweight='bold')
    ax_spo2.set_ylim(min(91, df_k_hours['spo2'].min() - 2), 100)
    ax_spo2.set_title('Oxygen Saturation (SpO2 %)')
    
    if show_healthy_ranges:
        # Normal range for SpO2 is 95-100%
        ax_spo2.axhspan(95, 100, color='lightgreen', alpha=0.2)
        legend_elements = [
            plt.Line2D([0], [0], color=main_color, marker=marker, markersize=5, label='SpO2'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.2, label='Healthy Range (95-100%)')
        ]
        ax_spo2.legend(handles=legend_elements, loc='best', fontsize=8, 
                      frameon=True, fancybox=True, shadow=True)
        # Yellow: Hypoxemia (90-95)
        ax_spo2.axhspan(90, 95, color='yellow', alpha=0.2)
        # Red: Severe Hypoxemia (below 90)
        ax_spo2.axhspan(0, 90, color='red', alpha=0.2)
    else:
        ax_spo2.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # 5. Heart Rate (HR) plot
    ax_pulse.plot(df_k_hours['timestamp'], df_k_hours['pulse'], 
                 color=main_color, marker=marker, markersize=markersize, label='HR')
    # ax_pulse.set_ylabel('Heart Rate (bpm)', fontweight='bold')
    ax_pulse.set_title('Heart Rate (bpm)')
    
    if show_healthy_ranges:
        # Normal range for heart rate of 70 years old
        ax_pulse.axhspan(60, 128, color='lightgreen', alpha=0.2)
        legend_elements = [
            plt.Line2D([0], [0], color=main_color, marker=marker, markersize=5, label='Heart Rate'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.2, label='Normal Range (60-100)')
        ]
        ax_pulse.legend(handles=legend_elements, loc='best', fontsize=8, 
                       frameon=True, fancybox=True, shadow=True)
        # # Yellow: Tachycardia and bradycardia
        ax_pulse.axhspan(128, 155, color='yellow', alpha=0.2)
        # ax_pulse.axhspan(60, 75, color='yellow', alpha=0.2)
        # Red: Tachycardia and bradycardia
        ax_pulse.axhspan(155, 200, color='red', alpha=0.2)
        ax_pulse.axhspan(0, 60, color='red', alpha=0.2)
        # Set y limit based on data but making also visible the start of the red zones
        ax_pulse.set_ylim(min(51, df_k_hours['pulse'].min() - 2), max(160, df_k_hours['pulse'].max() + 2))
    else:
        ax_pulse.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # # 6. HRV plot
    # ax_hrv.plot(df_k_hours['timestamp'], df_k_hours['hrv'], 
    #            color=main_color, marker=marker, markersize=markersize, label='HRV')
    # ax_hrv.set_ylabel('RMSSD (ms)', fontweight='bold')
    # ax_hrv.set_title('Heart Rate Variability')
    # ax_hrv.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # # 7. CVRR plot
    # ax_cvrr.plot(df_k_hours['timestamp'], df_k_hours['cvrr'], 
    #             color=main_color, marker=marker, markersize=markersize, label='CVRR')
    # ax_cvrr.set_ylabel('CVRR', fontweight='bold')
    # ax_cvrr.set_title('Coefficient of Variation of RR Intervals')
    # ax_cvrr.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # # 8. Steps plot
    # ax_steps.plot(df_k_hours['timestamp'], df_k_hours['steps'], 
    #              color=main_color, marker=marker, markersize=markersize, label='Steps')
    # ax_steps.set_ylabel('Total Steps', fontweight='bold')
    # ax_steps.set_title('Physical Activity')
    # ax_steps.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # Add period information as a small caption at the bottom
    fig_combined.text(0.5, 0.005, period_str, ha='center', fontsize=8, fontstyle='italic')
    
    # Format x-axes for all subplots
    all_axes = [
        ax_systolic, ax_diastolic, ax_breath, ax_spo2, ax_pulse
    ]
    
    for i, ax in enumerate(all_axes):
        # Format x-axis for the last row
        format_x_axis(ax, len(df_k_hours))

        # # Hide x labels but show grid and thicks
        # if i < len(all_axes) - 2:
        #     ax.set_xticklabels([])
        #     ax.tick_params(axis='x', which='both', bottom=False, top=False)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Make plot lines thinner
        for line in ax.get_lines():
            line.set_linewidth(0.9)
            
        # Hide legends for all subplots
        ax.get_legend().set_visible(False)

    
    # Add main title to the figure
    recording_date = start_time.strftime("%B %d, %Y")
    # fig_combined.suptitle(f"Physiological Monitoring Dataset • {recording_date}", 
    #                     fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout for the figure
    fig_combined.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    
    # Save combined figure
    dataset = "-pilot" if "PREATHOME" in filename else ""
    # fig_combined.savefig(f'{output_dir}/dataset{dataset}.svg', format='svg', bbox_inches='tight')
    fig_combined.savefig(f'{output_dir}/dataset{dataset}.pdf', dpi=300, bbox_inches='tight')
    
    print(f"Created publication-ready plots for {K}-hour data block with 2-column grid layout")
    print(f"Period: {start_fancy} to {end_fancy}")
    print("Saved plots in both SVG and PDF formats")
    
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot contiguous K-hour physiological data with 2-column grid layout.")
    parser.add_argument("filename", type=str, help="Path to the CSV file")
    parser.add_argument("K", type=int, help="Number of contiguous hours to plot")
    parser.add_argument("--no-healthy-ranges", action="store_true", help="Disable display of healthy reference ranges")
    args = parser.parse_args()
    
    main(args.filename, args.K, not args.no_healthy_ranges)