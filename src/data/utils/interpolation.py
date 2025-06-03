import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter

from athome.data.ATHOME.constants import LIMIT, MAX_DISTANCE, INTERP_METHOD

def fill_consecutive_nans(
    series: pd.Series,
    prefer_distance_1: bool = False
) -> pd.Series:
    """
    Fills consecutive NaNs groups of any size (from 1 to LIMIT) by finding the nearest
    valid consecutive samples of the same size within a maximum distance. Optionally
    prioritizes samples at a distance of at least 1.

    Args:
        series (pd.Series): The input series with missing values (NaNs).
        prefer_distance_1 (bool): If True, prioritizes windows at a distance of at least 1.
                                  If False, selects the nearest window regardless of distance.

    Returns:
        pd.Series: The series with consecutive NaNs filled.
    """
    # Identify groups of consecutive NaNs
    is_nan = series.isna()
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    group_sizes = is_nan.groupby(nan_groups).transform('size')

    # Iterate over each group of NaNs
    for group_id in nan_groups.unique():
        group_mask = (nan_groups == group_id)
        if not is_nan[group_mask].all():
            continue  # Skip groups that are not entirely NaNs
        group_size = group_sizes.loc[group_mask].iloc[0]
        if group_size > LIMIT:
            continue  # Skip groups larger than the limit

        # Get the indices of the NaN group
        nan_indices = series.index[group_mask]
        pos_start = series.index.get_loc(nan_indices[0])
        pos_end = pos_start + group_size - 1

        # Search for valid consecutive samples before the NaN group
        before_region_start = max(0, pos_start - MAX_DISTANCE)
        before_region = series.iloc[before_region_start:pos_start].values

        # Search for valid consecutive samples after the NaN group
        after_region_start = pos_end + 1
        after_region_end = min(len(series) - 1, pos_end + MAX_DISTANCE)
        after_region = series.iloc[after_region_start:after_region_end + 1].values

        def find_consecutive(region: np.ndarray, direction: str) -> tuple[np.ndarray, int]:
            """
            Find valid consecutive samples of the same size in a given region.

            Args:
                region (np.ndarray): The region to search for consecutive samples.
                direction (str): The direction to search for consecutive samples ('before' or 'after').

            Returns:
                np.ndarray: The valid consecutive samples of the same size.
                int: The index of the first valid sample in the region.
            """
            if len(region) < group_size:
                return None, None
            mask = ~pd.isna(region)
            kernel = np.ones(group_size, dtype=int)
            conv = np.convolve(mask, kernel, mode='valid')
            valid_indices = np.where(conv == group_size)[0]
            if len(valid_indices) == 0:
                return None, None
            idx = valid_indices[-1] if direction == 'before' else valid_indices[0]
            start = idx
            end = start + group_size
            return region[start:end], idx

        # Find valid consecutive samples before and after the NaN group
        before_window, before_idx = find_consecutive(before_region, 'before')
        after_window, after_idx = find_consecutive(after_region, 'after')

        # Select the nearest valid consecutive samples
        selected_data = None

        # Prioritize samples at a distance of at least 1
        if before_window is not None and after_window is not None:
            # Calculate distances
            before_distance = pos_start - (before_region_start + before_idx + group_size - 1)
            after_distance = (after_region_start + after_idx) - (pos_end + 1)

            # Choose the nearest window based on the distance
            if prefer_distance_1:
                # Prioritize windows at a distance of at least 1
                if before_distance >= 1 and after_distance >= 1:
                    selected_data = before_window if before_distance <= after_distance else after_window
                elif before_distance >= 1:
                    selected_data = before_window
                elif after_distance >= 1:
                    selected_data = after_window
                else:
                    # Fall back to the nearest window if no windows at distance >= 1 are found
                    selected_data = before_window if before_distance <= after_distance else after_window
            else:
                # Simply choose the nearest window
                selected_data = before_window if before_distance <= after_distance else after_window
        elif before_window is not None:
            before_distance = pos_start - (before_region_start + before_idx + group_size - 1)
            if prefer_distance_1:
                if before_distance >= 1:
                    selected_data = before_window
                else:
                    selected_data = before_window  # Fall back to adjacent window
            else:
                selected_data = before_window  # Always use the nearest window
        elif after_window is not None:
            after_distance = (after_region_start + after_idx) - (pos_end + 1)
            if prefer_distance_1:
                if after_distance >= 1:
                    selected_data = after_window
                else:
                    selected_data = after_window  # Fall back to adjacent window
            else:
                selected_data = after_window  # Always use the nearest window

        # If no valid window is found, search for the largest possible window in the entire series
        if selected_data is None:
            largest_window = find_largest_window(series)
            if largest_window is not None:
                selected_data = np.tile(largest_window, (group_size // len(largest_window)) + 1)
                selected_data = selected_data[:group_size]

        # Fill the NaN group with the selected data
        if selected_data is not None:
            series.loc[nan_indices] = selected_data

    return series

def find_largest_window(series: pd.Series) -> np.ndarray:
    """
    Find the largest possible window of valid consecutive samples in the series.

    Args:
        series (pd.Series): The input series.

    Returns:
        np.ndarray: The largest window of valid consecutive samples.
    """
    mask = ~series.isna()
    kernel = np.ones(len(series), dtype=int)
    conv = np.convolve(mask, kernel, mode='valid')
    max_size = np.max(conv)
    if max_size == 0:
        return None
    max_indices = np.where(conv == max_size)[0]
    start = max_indices[0]
    end = start + max_size
    return series.iloc[start:end].values

def custom_interp(
        window: pd.DataFrame,
        method: str = INTERP_METHOD,
        resample_freq: str = "1T"
) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    """
    Interpolates missing values in a given pandas DataFrame window using different methods.

    Args:
        window (pd.DataFrame): A dataframe window with missing values (NaNs) to interpolate.
        method (str): The interpolation method to use. Options:
                      - 'nearest', 'linear', 'spline', 'pchip'
                      - 'loess', 'savitzky_golay', 'none'

    Returns:
        pd.DataFrame: Interpolated window with missing values filled.
        dict[str, pd.Index]: Dictionary mapping column names to indices of interpolated values.
    """
    # Copy the window to avoid modifying the original
    interpolated_window = window.copy()
    interpolated_idxs = {}

    # Iterate over each column in the window
    for column in window.columns:
        series = window[column]

        #* Handle special columns (e.g., steps, location data)
        if column == "timestamp":
            continue # Skip interpolation for this column
        if column == "steps":
            # Forward fill grouping by day so that each day starts from 0 steps
            dates: np.ndarray = interpolated_window.index.date
            unique_dates = np.unique(dates)

            for day in unique_dates:
                # Get the first timestamp of the day
                day_data = interpolated_window.loc[interpolated_window.index.date == day]
                first_timestamp = day_data.index[0]  # First timestamp of the day
                
                # Check if the first timestamp has a NaN step value
                if pd.isna(day_data['steps'][0]):
                    # If NaN, set the first step value to zero
                    interpolated_window.loc[first_timestamp, 'steps'] = 0
                else:
                    # If not NaN, insert a new row just before it with zero steps
                    new_timestamp = first_timestamp - pd.Timedelta(resample_freq)
                    interpolated_window.loc[new_timestamp, 'steps'] = 0

            # Sort the DataFrame by index to ensure the new rows are in the correct order
            interpolated_window = interpolated_window.sort_index()

            # Group by day and interpolate steps column only
            interpolated_window = interpolated_window.groupby(
                interpolated_window.index.date, 
                group_keys=False
            ).apply(lambda x: x.assign(steps=x['steps'].ffill().bfill()))
            
            # Convert steps to integer
            interpolated_window['steps'] = interpolated_window['steps'].astype(int)
            continue  # Skip interpolation for this column
        if column == "location":
            # Forward fill location data
            interpolated_window[column] = series.ffill(limit=LIMIT).bfill(limit=LIMIT)
            continue  # Skip interpolation for this columns

        #* Find missing values and check their gap size
        is_nan = series.isna()
        group_sizes = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).transform('size')
        valid_nan_mask = (is_nan) & (group_sizes <= LIMIT)  # Only interpolate if within limit

        if not valid_nan_mask.any():
            interpolated_idxs[column] = pd.Index([])
            continue  # No eligible missing values, skip

        #* Get known and missing points for interpolation
        mask = series.notna()
        x_known = series.index[mask]
        y_known = series[mask]
        x_missing = series.index[valid_nan_mask]

        #* Define interpolation function based on the method
        if method == "nearest":
            f = interp1d(x_known, y_known, kind="nearest", fill_value="extrapolate")
        elif method == "linear":
            f = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate")
        elif method == "spline":
            f = CubicSpline(x_known, y_known)
        elif method == "pchip":
            f = PchipInterpolator(x_known, y_known)

        #* Define more complex interpolation methods
        elif method == "loess":
            y_smooth = lowess(y_known, x_known, frac=0.2, return_sorted=False)
            interpolated_series = pd.Series(y_smooth, index=x_known)
            f = lambda x: np.interp(x, x_known, interpolated_series)
        elif method == "savitzky_golay":
            smoothed = savgol_filter(series.dropna(), window_length=7, polyorder=3, mode='nearest')
            interpolated_series = pd.Series(smoothed, index=x_known)
            f = lambda x: np.interp(x, x_known, interpolated_series)
        elif method == "none":
            interpolated_idxs[column] = pd.Index([])
            continue
        elif method == "nearest_consec":
            # Store original values to identify interpolated indices
            original_values = interpolated_window[column].copy()
            
            interpolated_window[column] = fill_consecutive_nans(series)
            
            # Find indices where values changed due to interpolation
            interpolated_mask = (original_values.isna() & ~interpolated_window[column].isna())
            interpolated_idxs[column] = interpolated_window.index[interpolated_mask]
            continue
        elif method == "nearest_consec_1":
            # Store original values to identify interpolated indices
            original_values = interpolated_window[column].copy()
            
            interpolated_window[column] = fill_consecutive_nans(series, prefer_distance_1=True)
            
            # Find indices where values changed due to interpolation
            interpolated_mask = (original_values.isna() & ~interpolated_window[column].isna())
            interpolated_idxs[column] = interpolated_window.index[interpolated_mask]
            continue
        else:
            raise ValueError(f"Invalid interpolation method: {method}")

        #* Apply interpolation only within the limit
        interpolated_window.loc[x_missing, column] = f(x_missing)
        
        # Store the indices of interpolated values
        interpolated_idxs[column] = x_missing

    return interpolated_window, interpolated_idxs


def process_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    data: dict[str, dict[str, str]] = None,
    df: pd.DataFrame = None,
    interpolate: bool = True,
    resample_freq: str = "1T"
) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    """
    Process the data dictionary containing dataframes for each sensor.

    Args:
        start_date (pd.Timestamp): The start date of the window.
        end_date (pd.Timestamp): The end date of the window
        data (dict): A dictionary containing sensor dataframes. E.g.
            {
                "sensor1": [{
                    "timestamp": "2021-01-01 00:00:00",
                    "value": 10
                }, ...],
                "sensor2": [{
                    "timestamp": "2021-01-01 00:00:00",
                    "value": 20
                }, ...],
                ...
            }
        df (pd.DataFrame): The dataframe to merge the sensor dataframes into.
        interpolate (bool): Whether to interpolate the missing values in the dataframe.

    Returns:
        pd.DataFrame: Processed and interpolated dataframe.
        dict[str, pd.Index]: Dictionary mapping column names to indices of interpolated
    """
    # Ensure the df or data is provided
    if data is None and df is None:
        raise ValueError("Either 'data' or 'df' must be provided.")

    # Check if a single dataframe is provided
    if df is not None:
        # Check if the dataframe is empty
        if df.empty:
            df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq=resample_freq))
            df.index.name = "timestamp"
        else:
            # Set the index to the timestamp column
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        # Add nan value to the beginning and the end timestamp if not present
        if start_date not in df.index:
            df.loc[start_date] = np.nan
        if end_date not in df.index:
            df.loc[end_date] = np.nan

        # Resample at 1-minute intervals to ensure consistent interpolation
        df = df.resample(resample_freq).nearest(limit=1)

        # Sort the dataframe by timestamp
        df = df.sort_index()
    else:
        # Create a dataframe with all the sensor data
        df = pd.DataFrame(columns=["timestamp"])
        
        # Create a date range for empty sensors
        date_range = pd.date_range(start=start_date, end=end_date, freq=resample_freq)
        empty_sensors = []
        
        # Merge all sensor dataframes into a single dataframe
        for sensor, records in data.items():
            # Handle empty records case
            if not records:  
                empty_sensors.append(sensor)
                continue
                
            # Convert records to a dataframe
            sensor_df = pd.DataFrame(records)

            # Convert the timestamp column to datetime
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
            sensor_df = sensor_df.set_index("timestamp")

            # Add nan value to the beginning and the end timestamp if not present
            if start_date not in sensor_df.index:
                sensor_df.loc[start_date] = np.nan
            if end_date not in sensor_df.index:
                sensor_df.loc[end_date] = np.nan

            # Handle steps for cumulative sum
            if "steps" in sensor:
                # Group by day and apply cumulative sum
                sensor_df = sensor_df.groupby(
                    sensor_df.index.date
                ).apply(lambda x: x.assign(value=x["value"].cumsum()))

            # Remove duplicate timestamps which may arise fromc change of timezone (also count them)
            duplicates = sensor_df.index.duplicated(keep="first")
            if duplicates.any():
                sensor_df = sensor_df[~duplicates]
                # Count the number of duplicates
                num_duplicates = duplicates.sum()
                print(f"Found {num_duplicates} duplicate timestamps for sensor '{sensor}'")

            # Resample at resample_freq intervals to ensure consistent interpolation
            sensor_df = sensor_df.resample(resample_freq).nearest(limit=1)

            # Sort the dataframe by timestamp
            sensor_df = sensor_df.sort_index()

            # Rename the value column to the sensor name
            sensor_df = sensor_df.rename(columns={"value": sensor})

            # Merge the sensor dataframe with the main dataframe
            df = df.merge(sensor_df, on="timestamp", how="outer")

        # Sort the dataframe by timestamp
        df = df.set_index("timestamp")
        df = df.sort_index()
        
        # Handle case where all sensors have no data
        if len(df.columns) == 0 and empty_sensors:
            df = pd.DataFrame(index=date_range)
            df.index.name = "timestamp"
        
        # Add empty sensors with NaN values
        for sensor in empty_sensors:
            df[sensor] = np.nan

    # Interpolate missing values in the dataframe
    if interpolate:
        interpolated_df, interpolated_idxs = custom_interp(df, resample_freq=resample_freq)
    else:
        interpolated_df, interpolated_idxs = custom_interp(df, method="none", resample_freq=resample_freq)
    interpolated_df.reset_index(inplace=True)
    return interpolated_df, interpolated_idxs
