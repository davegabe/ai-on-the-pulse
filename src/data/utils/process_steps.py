import pandas as pd
from datetime import datetime, timedelta

def compute_steps_in_window(entries: list[dict[str, any]], window_start: datetime, window_end: datetime) -> float:
    """
    Calculate estimated steps in a time window using linear interpolation between step entries.

    Args:
        entries (list[dict[str, any]]): List of step entries, each containing 'timestamp' and 'steps' keys.
                 Steps should represent cumulative counts up to each timestamp.
        window_start (datetime): Start time of the analysis window.
        window_end (datetime): End time of the analysis window.

    Returns:
        float: Estimated step count within the specified window.
    """
    if not entries:
        return 0.0

    # Sort entries chronologically
    sorted_entries = sorted(entries, key=lambda x: x['timestamp'])
    times = [e['timestamp'] for e in sorted_entries]
    steps = [e['steps'] for e in sorted_entries]

    # Handle cases with no relevant data
    if not times:
        return 0.0
    if times[0] > window_end or times[-1] < window_start:
        return 0.0

    # Add boundary markers if needed to capture intervals before first/after last entry
    if window_start < times[0]:
        times.insert(0, window_start)
        steps.insert(0, 0.0)
    if window_end > times[-1]:
        times.append(window_end)
        steps.append(steps[-1])  # Maintain final step count after last measurement

    total_steps = 0.0

    # Calculate contribution from each time interval
    for i in range(1, len(times)):
        t_prev, t_curr = times[i-1], times[i]
        s_prev, s_curr = steps[i-1], steps[i]
        delta_steps = s_curr - s_prev

        # Skip intervals with negative or zero step changes
        if delta_steps <= 0:
            continue

        # Calculate overlap between current interval and analysis window
        overlap_start = max(t_prev, window_start)
        overlap_end = min(t_curr, window_end)
        if overlap_start >= overlap_end:
            continue

        # Calculate time proportions
        interval_duration = (t_curr - t_prev).total_seconds()
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        if interval_duration == 0:
            continue  # Prevent division by zero

        # Add proportional contribution from this interval
        step_contribution = delta_steps * (overlap_duration / interval_duration)
        total_steps += step_contribution

    return total_steps

def calculate_steps_features(step_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily step features from timestamped step data.
    
    Args:
        step_data: DataFrame containing 'timestamp' and 'steps' columns
        
    Returns:
        DataFrame with daily features: total_steps, steps_morning (6-12),
        steps_afternoon (12-18), steps_evening (18-24), steps_night (0-6)
    """
    if step_data.empty:
        return pd.DataFrame(columns=["timestamp", "total_steps", "steps_morning", 
                                    "steps_afternoon", "steps_evening", "steps_night"])
    
    # Ensure timestamp is datetime
    step_data = step_data.copy()
    step_data['timestamp'] = pd.to_datetime(step_data['timestamp'], utc=True)
    step_data['date'] = step_data['timestamp'].dt.date
    step_data = step_data.sort_values('timestamp')
    
    daily_features = []

    for date, group in step_data.groupby('date'):
        date_pd = pd.Timestamp(date, tz='UTC')
        entries = group.to_dict('records')

        # Basic daily total (sum of steps during the day)
        total_steps = group['steps'].sum() if not group.empty else 0.0

        # Define time periods
        time_windows = [
            ('morning', 6, 12),     # 6:00-11:59
            ('afternoon', 12, 18),  # 12:00-17:59
            ('evening', 18, 24),    # 18:00-23:59
            ('night', 0, 6)         # 00:00-5:59
        ]

        # Calculate steps in each period
        window_steps = {}
        for name, start_hour, end_hour in time_windows:
            window_start = date_pd + timedelta(hours=start_hour)
            window_end = date_pd + timedelta(hours=end_hour) if end_hour != 24 \
                else date_pd + timedelta(days=1)
            
            # Filter data for the current window
            window_data = group[(group['timestamp'] >= window_start) & 
                                (group['timestamp'] < window_end)]
            
            window_steps[f'steps_{name}'] = window_data['steps'].sum() if not window_data.empty else 0.0

        # Compile daily features
        daily_features.append({
            'timestamp': date_pd,
            'total_steps': total_steps,
            **window_steps,
        })

    # Create final dataframe
    feature_df = pd.DataFrame(daily_features)
    
    # Ensure all expected columns exist
    expected_columns = ["timestamp", "total_steps", "steps_morning", 
                       "steps_afternoon", "steps_evening", "steps_night"]
    for col in expected_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    
    return feature_df

def process_step_data(input_file: str, output_file: str) -> None:
    """
    Process raw step data into daily activity features.

    Args:
        input_file: Path to CSV file containing timestamped step data
        output_file: Path to save processed features CSV
    """
    # Load data
    df = pd.read_csv(input_file)
    
    # Calculate features
    feature_df = calculate_steps_features(df)

    # Round float columns for nicer output
    float_cols = feature_df.select_dtypes(include=[float]).columns
    feature_df[float_cols] = feature_df[float_cols].round(2)
    
    # Save to output file
    feature_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
