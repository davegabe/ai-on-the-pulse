import csv
from datetime import datetime, timedelta

from retrieve import PATIENTS

def parse_timestamps(input_csv: str):
    """
    Parses timestamps from a CSV file and returns them as a sorted list of datetime objects.

    Args:
        input_csv (str): The path to the input CSV file containing timestamps.

    Returns:
        List[datetime]: A sorted list of datetime objects parsed from the CSV file.

    Notes:
        - The CSV file is expected to have a column named 'timestamp'.
        - Timestamps should be in the format '%Y-%m-%d %H:%M:%S'.
        - Invalid timestamp formats are skipped.
    """
    timestamps = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row['timestamp']
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                timestamps.append(ts)
            except ValueError:
                continue  # Skip invalid timestamp formats
    timestamps.sort()
    return timestamps

def find_gaps(timestamps: list):
    """
    Identifies gaps in a list of timestamps where the difference between consecutive timestamps
    exceeds one minute. For each gap found, it returns the start and end of the gap, as well as
    the duration of the gap.

    Args:
        timestamps (list): A list of datetime objects representing the timestamps.

    Returns:
        list: A list of tuples, each containing the start of the gap (datetime), the end of the gap (datetime),
              and the duration of the gap (timedelta).
    """
    gaps = []
    for i in range(1, len(timestamps)):
        prev = timestamps[i-1]
        curr = timestamps[i]
        delta = curr - prev
        if delta > timedelta(minutes=1):
            gap_start = prev + timedelta(minutes=1)
            gap_end = curr - timedelta(minutes=1)
            gap_duration = (curr - prev) - timedelta(minutes=1)
            gaps.append((gap_start, gap_end, gap_duration))
    return gaps

def write_gaps_csv(gaps: list, output_csv: str):
    """
    Writes a list of gaps to a CSV file.

    Each gap is a tuple containing a start timestamp, an end timestamp, and a duration.
    The CSV file will contain columns for the start timestamp, end timestamp, and duration in minutes.

    Args:
        gaps (list): A list of tuples, where each tuple contains:
            - start (datetime): The start timestamp of the gap.
            - end (datetime): The end timestamp of the gap.
            - duration (timedelta): The duration of the gap.
        output_csv (str): The file path to the output CSV file.

    Returns:
        None
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_timestamp', 'end_timestamp', 'duration_minutes'])
        for gap in gaps:
            start, end, duration = gap
            duration_minutes = int(duration.total_seconds() // 60)
            writer.writerow([
                start.strftime('%Y-%m-%d %H:%M:%S'),
                end.strftime('%Y-%m-%d %H:%M:%S'),
                duration_minutes
            ])

def count_consecutive_windows(timestamps: list, window_size: int):
    """
    Counts the number of consecutive windows of a given size in a list of timestamps.

    Args:
        timestamps (list): A list of datetime objects representing timestamps.
        window_size (int): The size of the window to count consecutive timestamps.

    Returns:
        int: The number of consecutive windows of the specified size.
    """
    if not timestamps:
        return 0
    blocks = []
    current_block = [timestamps[0]]
    for ts in timestamps[1:]:
        if ts == current_block[-1] + timedelta(minutes=1):
            current_block.append(ts)
        else:
            blocks.append(current_block)
            current_block = [ts]
    if current_block:
        blocks.append(current_block)
    count = 0
    for block in blocks:
        n = len(block)
        if n >= window_size:
            count += (n - window_size + 1)
    return count

def main():
    for patient in range(len(PATIENTS)):
        input_csv = f'processed_{patient}/all_data.csv'
        output_csv = f'gaps_{patient}.csv'
        window_size = 16
        
        timestamps = parse_timestamps(input_csv)
        if len(timestamps) < 1:
            print("No valid timestamps found in the input CSV.")
            return
        
        gaps = find_gaps(timestamps)
        write_gaps_csv(gaps, output_csv)
        
        window_count = count_consecutive_windows(timestamps, window_size)
        print(f"Number of valid consecutive windows of size {window_size}: {window_count}")

        # Count average gap duration
        total_duration = sum(gap[2].total_seconds() for gap in gaps)
        avg_duration = total_duration / len(gaps) if gaps else 0
        print(f"Average gap duration: {avg_duration:.2f} seconds")

        # Count gaps bigger than 10 samples
        big_gaps = [gap for gap in gaps if gap[2] > timedelta(minutes=10)]
        print(f"Number of gaps bigger than 10 minutes: {len(big_gaps)}")

if __name__ == '__main__':
    main()