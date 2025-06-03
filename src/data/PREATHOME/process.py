import os
import pandas as pd
import numpy as np

from data.utils.process_steps import process_step_data
from data.utils.process_sleep import process_sleep_data
from data.utils.interpolation import custom_interp

def interpolate_csv(input_file: str, output_file: str):
    """
    Interpolate the missing values in a CSV file and save the processed file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to save the processed CSV file.
    """
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Replace all 0s with NaNs
    df = df.replace(0, np.nan)

    # Resample the data to 10-minute intervals
    df = df.resample('10T').nearest(limit=1)

    # Interpolate NaNs
    df, _ = custom_interp(df, resample_freq="10T")

    # Round the float columns to 2 decimal places
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[float_cols] = df[float_cols].round(2)

    # Remove the label column
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # Save the processed file
    df.to_csv(output_file, index=True)
    print(f"Processed file saved as: {output_file}")

if __name__ == '__main__':
    os.makedirs('../processed', exist_ok=True)
    script_dir = os.path.dirname(__file__)
    
    # Interpolate the measurement data
    interpolate_csv(
        script_dir + '/original/measurements.csv',
        script_dir + '/processed/measurements.csv'
    )

    # Process the sleep data and extract features
    process_sleep_data(
        script_dir + '/original/sleep.csv',
        script_dir + '/processed/sleep_features.csv'
    )

    # Process the step data
    process_step_data(
        script_dir + '/original/measurements.csv',
        script_dir + '/processed/steps_features.csv'
    )
