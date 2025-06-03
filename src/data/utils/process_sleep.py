import pandas as pd
import numpy as np

def compute_overlap(ep_start: pd.Timestamp, ep_end: pd.Timestamp, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """
    Compute the overlap between two time intervals.

    Args:
        ep_start (datetime): start time of the sleep episode.
        ep_end (datetime): end time of the sleep episode.
        window_start (datetime): start time of the analysis window.
        window_end (datetime): end time of the analysis window.

    Returns:
        float: duration of the overlap.
    """
    latest_start = max(ep_start, window_start)
    earliest_end = min(ep_end, window_end)
    if latest_start >= earliest_end:
        return 0
    else:
        return (earliest_end - latest_start).total_seconds() / 60

def process_sleep_data(input_file: str, output_file: str):
    """
    Process the sleep data and extract features.

    Args:
        input_file (str): path to the input CSV file.
        output_file (str): path to save the processed CSV file.
    """
    # Read the sleep CSV file
    df = pd.read_csv(input_file)

    # Convert the start and end columns to datetime
    df['start'] = pd.to_datetime(df['start'])
    df['end']   = pd.to_datetime(df['end'])

    # Assign sleep to the correct "sleep date"
    start_hour = 18
    df['sleep_date'] = df['start'].dt.date
    df.loc[df['start'].dt.hour < start_hour, 'sleep_date'] = df['sleep_date'] - pd.Timedelta(days=1)

    # Prepare a list to accumulate daily feature dictionaries
    daily_results = []

    # Define analysis windows (nighttime and afternoon nap)
    for sleep_date, group in df.groupby('sleep_date'):
        # Build timestamps for the current sleep period
        day_start = pd.Timestamp(sleep_date)
        night_window_start = day_start + pd.Timedelta(hours=start_hour)  # 18:00 of previous evening
        night_window_end   = day_start + pd.Timedelta(hours=start_hour + 17) # 11:00 of the next day

        afternoon_window_start = day_start + pd.Timedelta(hours=12)  # 12:00
        afternoon_window_end   = day_start + pd.Timedelta(hours=17)  # 17:00

        # Initialize accumulators for Insomnia/Hypersomnia
        total_night_sleep = 0.0  # total sleep minutes in nighttime window
        total_day_sleep   = 0.0  # sleep minutes outside nighttime window
        night_intervals   = []  # to store details for sleep quality features
        deep_sleep_total  = 0.0  # total deep sleep (phase 241) minutes in nighttime

        # Process each sleep episode
        for _, row in group.iterrows():
            ep_start = row['start']
            ep_end   = row['end']
            # Total duration of the episode (in minutes)
            ep_duration = (ep_end - ep_start).total_seconds() / 60  # in minutes

            # Compute overlap with the nighttime window
            night_overlap = compute_overlap(ep_start, ep_end, night_window_start, night_window_end)
            total_night_sleep += night_overlap

            # The remaining duration (if any) is considered daytime sleep
            day_duration = ep_duration - night_overlap
            total_day_sleep += day_duration if day_duration > 0 else 0
            if day_duration > 0:
                print(f"Daytime sleep episode: {ep_start} - {ep_end}")  

            # If the episode overlaps with the nighttime window, store details
            if night_overlap > 0:
                clipped_start = max(ep_start, night_window_start)
                clipped_end   = min(ep_end, night_window_end)
                night_intervals.append({
                    'start': clipped_start,
                    'end': clipped_end,
                    'phase': row['phase'],
                    'duration': night_overlap
                })
                if row['phase'] == 241:
                    deep_sleep_total += night_overlap

        # Additional IH features: total sleep, sleep onset, wake-up, deep sleep ratio
        total_sleep = total_night_sleep + total_day_sleep
        if night_intervals:
            # Sort intervals by start time and extract sleep onset and wake-up times
            sorted_night_intervals = sorted(night_intervals, key=lambda x: x['start'])
            clipped_starts = [interval['start'] for interval in sorted_night_intervals]
            clipped_ends = [interval['end'] for interval in sorted_night_intervals]
            sleep_onset = (min(clipped_starts) - day_start).total_seconds() / 60 # in minutes after midnight
            wake_up = (max(clipped_ends) - day_start).total_seconds() / 60 # in minutes after midnight
            deep_sleep_ratio = deep_sleep_total / total_night_sleep if total_night_sleep > 0 else np.nan
        else:
            sleep_onset = np.nan
            wake_up = np.nan
            deep_sleep_ratio = np.nan

        #* Insomnia/Hypersomnia (IH) features
        # Nighttime Sleep Duration, Daytime Sleep Duration, Sleep Efficiency
        if night_intervals:
            clipped_starts = [interval['start'] for interval in night_intervals]
            clipped_ends   = [interval['end'] for interval in night_intervals]
            time_in_bed = (max(clipped_ends) - min(clipped_starts)).total_seconds() / 60  # in minutes
            sleep_efficiency = total_night_sleep / time_in_bed if time_in_bed > 0 else np.nan
        else:
            time_in_bed = 0
            sleep_efficiency = np.nan

        #* Sleep Quality (SQ) features
        if night_intervals:
            # Sort intervals by start time
            night_intervals_sorted = sorted(night_intervals, key=lambda x: x['start'])
            # Fragmentation index: count transitions where phase changes
            frag_index = 0
            durations = [interval['duration'] for interval in night_intervals_sorted]
            for i in range(1, len(night_intervals_sorted)):
                if night_intervals_sorted[i]['phase'] != night_intervals_sorted[i-1]['phase']:
                    frag_index += 1
            # Normalized Fragmentation: frag index divided by number of intervals
            normalized_frag = frag_index / len(night_intervals_sorted) if len(night_intervals_sorted) > 0 else np.nan
            # Average sleep session duration
            avg_session_duration = np.mean(durations)
            # Minimum sleep session duration
            min_session_duration = min(durations)
            # Maximum sleep session duration
            max_session_duration = max(durations)
            # Standard deviation of sleep session durations
            std_session_duration = np.std(durations)
            # Average deep sleep duration (for sessions that are deep sleep)
            deep_sleep_durations = [interval['duration'] for interval in night_intervals_sorted if interval['phase'] == 241]
            avg_deep_sleep_duration = np.mean(deep_sleep_durations) if deep_sleep_durations else np.nan
            # Standard deviation of deep sleep durations
            std_deep_sleep_duration = np.std(deep_sleep_durations) if deep_sleep_durations else np.nan
        else:
            frag_index = np.nan
            normalized_frag = np.nan
            avg_session_duration = np.nan
            max_session_duration = np.nan
            std_session_duration = np.nan
            avg_deep_sleep_duration = np.nan

        # #* Abnormal Daytime Sleep (ADS) features
        # nap_durations = []
        # nap_start_times = []  # nap start time relative to noon (in hours)
        # nap_count = 0
        # for _, row in group.iterrows():
        #     ep_start = row['start']
        #     ep_end   = row['end']
        #     nap_overlap = compute_overlap(ep_start, ep_end, afternoon_window_start, afternoon_window_end)
        #     if nap_overlap > 0:
        #         nap_durations.append(nap_overlap)
        #         nap_count += 1
        #         # Use the clipped start time (if the episode started before noon, use 12:00)
        #         clipped_start = max(ep_start, afternoon_window_start)
        #         # Relative start time in hours (0 means exactly 12:00)
        #         nap_start_rel = (clipped_start - afternoon_window_start).total_seconds() / 3600
        #         nap_start_times.append(nap_start_rel)
        
        # total_nap_duration = sum(nap_durations) if nap_durations else 0
        # avg_nap_start = np.mean(nap_start_times) if nap_start_times else np.nan
        # max_nap_duration = max(nap_durations) if nap_durations else np.nan
        # nap_start_std = np.std(nap_start_times) if nap_start_times else np.nan

        # Collect the computed features into one dictionary
        daily_results.append({
            'timestamp': sleep_date,

            # Insomnia/Hypersomnia (IH) features:
            'sleep_duration': total_night_sleep,  # minutes
            'deep_sleep_ratio': deep_sleep_ratio,         # proportion
            'sleep_onset': sleep_onset,                   # minutes after midnight
            'wake_up': wake_up,                           # minutes after midnight

            'EXTRA_day_sleep_duration': total_day_sleep,        # minutes
            'EXTRA_sleep_efficiency': sleep_efficiency,         # ratio
            'EXTRA_total_sleep_duration': total_sleep,          # minutes
            'EXTRA_time_in_bed': time_in_bed,                  # minutes

            # Sleep Quality (SQ) features:
            'fragmentation_index': frag_index,
            'avg_session_duration': avg_session_duration,  # minutes
            'max_session_duration': max_session_duration,  # minutes
            'avg_deep_sleep_duration': avg_deep_sleep_duration,  # minutes

            'EXTRA_normalized_fragmentation': normalized_frag,
            'EXTRA_min_session_duration': min_session_duration,  # minutes
            'EXTRA_std_session_duration': std_session_duration,  # minutes
            'EXTRA_std_deep_sleep_duration': std_deep_sleep_duration,  # minutes

            # # Abnormal Daytime Sleep (ADS) features:
            # 'ADS_nap_duration': total_nap_duration,  # minutes
            # 'ADS_nap_count': nap_count,
            # 'ADS_avg_nap_start': avg_nap_start,       # hours relative to 12:00 noon
            # 'ADS_max_nap_duration': max_nap_duration,  # minutes
            # 'ADS_nap_start_std': nap_start_std         # variability in nap start time (hours)
        })
    
    # Convert the results list into a DataFrame and save as CSV
    result_df = pd.DataFrame(daily_results)

    # Resample to 1-day frequency
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    result_df = result_df.set_index('timestamp').resample('1D').asfreq().reset_index()

    # Round float columns to 5 decimal places
    float_cols = result_df.select_dtypes(include=['float64', 'float32']).columns
    result_df[float_cols] = result_df[float_cols].round(5)
    result_df.to_csv(output_file, index=False)
    print(f"Processed sleep features saved to: {output_file}")
