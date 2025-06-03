import os
import openai
import pandas as pd
from dotenv import load_dotenv

# OpenAI API key
load_dotenv()
client = openai.Client(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Define field descriptions for the data
field_descriptions = {
    # General
    "timestamp": "**Timestamp**: The exact date and time the data was recorded.",
    # "time": "**Time**: A continuous measure of time, often used for time-based analysis or to mark specific intervals within the recording.",
    # "weekday": "**Weekday**: The day of the week when the data was recorded (e.g., Monday, Tuesday), which can be used for identifying patterns related to routine.",
    # "day": "**Day**: The day of the month when the data was recorded, indicating specific days for analysis of trends or anomalies.",
    
    # Vital Signs
    "systolic": "**Systolic BP**: Arterial pressure during the heart's contraction phase (systole), measured in millimeters of mercury (mmHg).",
    "diastolic": "**Diastolic BP**: Arterial pressure during the heart's relaxation phase (diastole), measured in millimeters of mercury (mmHg).",
    "spo2": "**SpO2**: Blood Oxygen Saturation, representing the percentage of oxygen-saturated hemoglobin in the blood, captured by the wearable device.",
    "hrv": "**HRV**: Heart Rate Variability, the variation in time intervals between consecutive heartbeats, expressed in milliseconds (ms), reflecting autonomic nervous system function.",
    "cvrr": "**CVRR**: Coefficient of Variation for RR intervals, which measures heart rhythm variability by quantifying the variation in the time intervals between successive heartbeats.",
    "breath": "**Breath**: The respiratory rate, representing the number of breaths taken per minute, indicating respiratory function.",
    "pulse": "**Pulse**: The number of heartbeats per minute, representing the patient's heart rate.",
    
    # Activity Signals
    "steps": "**Steps**: The total count of steps taken in a given time interval, captured by the wearable device's motion sensors.",
    "total_steps": "**Total Steps**: The aggregate number of steps taken over the entire period, typically computed as the sum across different time intervals.",
    "steps_morning": "**Morning Steps**: The number of steps taken during the morning period, often used to analyze activity patterns and circadian rhythms.",
    "steps_afternoon": "**Afternoon Steps**: The number of steps taken during the afternoon period, capturing daytime physical activity.",
    "steps_evening": "**Evening Steps**: The number of steps taken during the evening period, used to assess late-day activity levels.",
    "steps_night": "**Night Steps**: The number of steps taken during the night period, which can help identify restlessness or unusual activity during typical sleep hours.",
    
    # Environmental Signals
    "temperature": "**Environmental Temperature**: The temperature of the surrounding environment, measured in degrees Celsius, recorded by environmental sensors.",
    "humidity": "**Humidity**: The relative humidity percentage, capturing the moisture content in the air, recorded by environmental sensors.",
    "co2": "**CO2**: The concentration of carbon dioxide in the air, measured in parts per million (ppm), which can reflect the quality of the air the patient is exposed to.",
    "pressure": "**Pressure**: The atmospheric pressure, measured in hectopascals (hPa), recorded by environmental sensors to assess changes in weather and its impact on health.",
    "tvoc": "**TVOC (Total VOC)**: The concentration of volatile organic compounds in the air, measured in parts per billion (ppb), which could indicate environmental pollutants.",
    "luminosity": "**Luminosity**: The level of ambient light, measured in lux, which can be useful for assessing environmental conditions affecting the patient.",
    
    # Location & Timing Signals
    "bagno": "**Bagno**: A binary indicator showing whether the user is in the bathroom (bagno). This is determined by proximity to the corresponding environmental sensor or beacon.",
    "outside": "**Outside**: A binary indicator showing whether the user is outside the designated home environment, as detected by environmental sensors or beacons.",
    "stanza1": "**Stanza1**: Binary indicator representing the user's location within a specific room (stanza1), detected by proximity to Bluetooth beacons.",
    "stanza2": "**Stanza2**: Binary indicator representing the user's location within a specific room (stanza2), detected by proximity to Bluetooth beacons.",
    "stanza3": "**Stanza3**: Binary indicator representing the user's location within a specific room (stanza3), detected by proximity to Bluetooth beacons.",
    "stanza4": "**Stanza4**: Binary indicator representing the user's location within a specific room (stanza4), detected by proximity to Bluetooth beacons.",
    "stanza1_time": "**Stanza1 Time**: The duration in seconds that the user spends in the specific room (stanza1) during the recording.",
    "stanza2_time": "**Stanza2 Time**: The duration in seconds that the user spends in the specific room (stanza2) during the recording.",
    "stanza3_time": "**Stanza3 Time**: The duration in seconds that the user spends in the specific room (stanza3) during the recording.",
    "stanza4_time": "**Stanza4 Time**: The duration in seconds that the user spends in the specific room (stanza4) during the recording.",
    "bagno_time": "**Bagno Time**: The duration in seconds that the user spends in the bathroom (bagno), as detected by the corresponding environmental sensor or beacon.",
    "outside_time": "**Outside Time**: The duration in seconds that the user spends outside, as determined by proximity to the corresponding environmental sensor or beacon.",
    
    # Sleep Signals
    "sleep_duration": "**Sleep Duration**: The total amount of time spent asleep during the sleep period, typically measured in hours or minutes.",
    "deep_sleep_ratio": "**Deep Sleep Ratio**: The proportion of deep sleep relative to the total sleep duration, indicating sleep quality.",
    "sleep_onset": "**Sleep Onset**: The time it takes for the user to fall asleep after going to bed, or the recorded start time of sleep.",
    "wake_up": "**Wake Up**: The recorded time of awakening or the number of awakenings during the sleep period.",
    "fragmentation_index": "**Fragmentation Index**: A metric that quantifies the degree of sleep disruption based on the number and frequency of brief awakenings or movements.",
    "avg_session_duration": "**Average Session Duration**: The average duration of individual sleep sessions, calculated over a given period.",
    "max_session_duration": "**Maximum Session Duration**: The longest continuous sleep session recorded within a given period.",
    "avg_deep_sleep_duration": "**Average Deep Sleep Duration**: The average length of deep sleep episodes during the sleep period, reflecting restorative sleep quality."
}

def gpt_explain(df: pd.DataFrame, patient_info: str):
    """
    Generate an explanation for the provided patient data using GPT-4o.

    Args:
        df (pd.DataFrame): The filtered patient data to analyze.
        patient_info (str): Additional patient context from the API.

    Returns:
        str: The explanation translated into Italian.
    """

    # Construct the explanation prompt
    prefix = (
        "You are an advanced AI system analyzing physiological and environmental signals from a wearable sensor to detect health anomalies. "
        "Your goal is to provide a structured, clinically relevant explanation for detected anomalies. "
        "Below is a description of the provided signals:\n\n"
    )

    # Append field descriptions for all available columns
    for col in df.columns:
        if col in field_descriptions:
            prefix += f"- {field_descriptions[col]}\n"
        else:
            print(f"[ERR] Field description not found for column: {col}")

    # Ensure "Anomaly Energy" has at least one sample with >= 1 value
    assert "Anomaly Energy" in df.columns and df["Anomaly Energy"].max() >= 1, "No anomalies detected."

    # Include anomaly explanation
    prefix += (
        "\n- **Anomaly Energy**: Ratio of the detected anomaly energy to the threshold, indicating the intensity of the anomaly. "
        "Values < 1 are normal; values >= 1 indicate an anomaly.\n\n"
        "### Patient Information:\n"
        f"{patient_info}\n\n"
        "### Patient Data:\n"
    )

    # Format data into structured rows
    data_rows = []
    column_headers = " | ".join(df.columns)
    data_rows.append(column_headers)

    for row in df.itertuples(index=False):
        data_rows.append(" | ".join(map(str, row)))

    patient_data_section = "\n".join(data_rows)

    # Construct the analysis prompt
    analysis_prompt = (
        "\n### Provide a concise medical assessment with the following structure:\n"
        "- **Analysis**: Identify and describe key anomalies, focusing on significant variations in signals. Avoid explicitly mentioning the anomaly energy. "
        "Explain the observed trends over time, highlighting potential health implications. "
        "Use a bullet list format for clarity.\n\n"
        "- **Causes**: Hypothesize potential causes based on the patient's demographic data, anomaly characteristics (e.g., sudden, prolonged), "
        "and environmental influences. Differentiate between likely clinical events and possible sensor artifacts.\n\n"
        "- **Criticality**: Assess the clinical importance of the detected anomalies. Specify if immediate action is needed, "
        "if monitoring is advised, or if the anomaly is likely insignificant.\n\n"
        "- **Scores**:\n"
        "    - **Significance Score (0-10)**: Rate the clinical relevance of the anomaly (0 = noise, 10 = critical issue).\n"
        "    - **Criticality Score (0-10)**: Rate the urgency of the anomaly (0 = minimal concern, 10 = requires immediate attention).\n\n"
        "Ensure the explanation is clear, structured, and clinically useful."
    )

    # Combine all parts into final input
    input_text = prefix + patient_data_section + analysis_prompt

    # Call GPT-4o for anomaly explanation
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.15,
        messages=[{"role": "system", "content": input_text}]
    )
    explanation = response.choices[0].message.content

    return ita_explanation
