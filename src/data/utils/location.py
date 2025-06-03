import pandas as pd

def location_to_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert location data columns to one-hot encoded columns.

    Args:
        df (pd.DataFrame): The dataframe containing location data columns.

    Returns:
        pd.DataFrame: The dataframe with one-hot encoded location columns.
    """
    # Convert "location" column to numeric based on enum values
    values = df["location"].unique()
    mapping = {value: i for i, value in enumerate(values)}
    mapping["Outside"] = -1
    df["location"] = df["location"].map(mapping)

    # Save the mapping to a csv file
    loc = pd.DataFrame(mapping.items(), columns=["location", "location_id"])
    save_data(loc, "location_mapping.csv", index=False)

    # Count nan values in location
    print(f"Missing values in location: {df['location'].isnull().sum()}")
    df["location"] = df["location"].fillna(-1)
    df["location"] = df["location"].astype(int)
    
    # Replace location values with mapped values
    print(df["location"].unique())
    df["location"] = df["location"].map(loc["location"]).str.lower()
    
    # Perform one-hot encoding
    df = pd.concat([df, pd.get_dummies(df["location"], prefix="", prefix_sep="")], axis=1)
    df = df.drop(columns=["location"])

    # Ensure outside column is present
    if "outside" not in df.columns:
        df["outside"] = 0

    # Create a column for each one-hot encoded state with the cumulative time spent in that state
    # It should reset when the column "DAY" changes
    df["day"] = df.index.day
    for state in loc["location"].str.lower():
        df[f"{state}_time"] = df[state].groupby(df["day"]).cumsum()
    df = df.drop(columns=["day"])
