import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from openai import OpenAI
from markdown_pdf import MarkdownPdf, Section
import matplotlib.dates as mdates
from tqdm import tqdm
import matplotlib.ticker as ticker

from data.utils.unit import feature_units
from data.utils.anomalies import main_features, anomaly_types
from data.ATHOME.constants import ANOMALIES as ATHOME_ANOMALIES
from data.PREATHOME.constants import ANOMALIES as PREATHOME_ANOMALIES
from backend.explainer import gpt_explain, field_descriptions, translate_to_italian

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ANOMALIES_PER_TYPE = 5

# Base directory for saving plots and markdown files
basepath = "plots/all_anomalies/"

patients_info = {
    "PREATHOME": "The patient is a 60-year-old male with no known medical conditions.",
    "ATHOME_0": "The patient is a 74-year-old male, weighing 78 kg and standing 172 cm tall. He has Parkinson’s Disease, Type 2 Diabetes Mellitus, Hypertension, and Stage II Peripheral Artery Disease. His medical therapy includes Inderal (40 mg), Depakin (500 mg), Olprezide (20mg/12.5mg), Entact (10 mg), Brimica Genuaie (340mcg/12mcg), Jardiance (10 mg), Cardioaspirin (100 mg), Pantorc (40 mg), Vytorin (10 mg/40 mg), and Nodigap (colecalciferol – 50000 IU). Medications are taken at specific times throughout the day, with Nodigap administered once monthly."
}


def load_data_all(data_type: str, anomaly_id: int) -> tuple[pd.DataFrame, pd.Timestamp, list[str], int]:
    """
    Load data for a given anomaly type (for ATHOME or PREATHOME), aggregate predictions,
    and create a DataFrame with relevant features, Prediction, Ground Truth, and Anomaly Energy.
    Returns also the list of relevant features and the window size used for aggregation.

    Args:
        data_type (str): The data type (PREATHOME or ATHOME_0).
        anomaly_id (int): The index of the anomaly type.

    Returns:
        tuple[pd.DataFrame, pd.Timestamp, list[str], int]: The DataFrame, first date, relevant features, and window size.
    """
    script_dir = os.path.dirname(__file__)
    results_path = os.path.join(
        script_dir,
        "checkpoints/contextual_anomaly_detection_UniTS_sup__UniTS_All_ftM_dm128_el3_Exp_0/results"
    )

    if data_type == "PREATHOME":
        anomaly_type = PREATHOME_ANOMALIES[anomaly_id]
        rel_features = main_features[anomaly_type]
        freq = "10T"  # 10-minute intervals
    elif data_type.startswith("ATHOME"):
        anomaly_type = ATHOME_ANOMALIES[anomaly_id]
        rel_features = main_features[anomaly_type]
        freq = "1T"   # 1-minute intervals
    else:
        raise ValueError(f"Data {data_type} not supported")

    # For some anomaly types force daily frequency
    if anomaly_type.startswith("SLP_") or anomaly_type.startswith("ACT_"):
        freq = "1D"

    # Load prediction, sensor data, and ground truth arrays
    scores = np.load(os.path.join(results_path, f"{data_type}_{anomaly_type}/energy.npy"))
    pred = np.load(os.path.join(results_path, f"{data_type}_{anomaly_type}/pred.npy"))
    data = np.load(os.path.join(results_path, f"{data_type}_{anomaly_type}/data.npy"))
    gt = np.load(os.path.join(results_path, f"{data_type}_{anomaly_type}/gt.npy"))
    threshold = np.load(os.path.join(results_path, f"{data_type}_{anomaly_type}/threshold.npy"))

    # Load the starting date from file and convert to Timestamp
    with open(os.path.join(results_path, f"{data_type}_{anomaly_type}/first_date.txt"), "r") as f:
        first_date_str = f.read().strip()
    first_date = pd.Timestamp(first_date_str)

    # Create a datetime index for the x-axis using the determined frequency
    date_values = pd.date_range(first_date, periods=data.shape[0], freq=freq)

    # Extract the first sample of each window (for the relevant features) for plotting
    data_series_array = data[:, 0, :len(rel_features)]
    gt = gt[::data.shape[1]]  # Downsample ground truth to match data shape
    print(f"Data shape: {data.shape}, Scores shape: {scores.shape}, Pred shape: {pred.shape}, GT shape: {gt.shape}")
    print(f"Anomalies (GT sum): {gt.sum()}")
    print(f"Anomalies (Pred sum): {pred.sum()}")

    # Aggregate predictions to match the shape of data_series_array
    window_size = data.shape[1]  # Number of samples in each window (e.g., 16)
    pred = pred.reshape(-1, window_size)
    new_scores = scores[::window_size]  # Downsample scores to match data shape

    # new_scores /= overlap_counts
    new_scores = new_scores / threshold  # Normalize scores based on threshold
    print(f"Min Score: {new_scores.min()}, Max Score: {new_scores.max()}")

    # Convert aggregated predictions to binary (1 for anomaly, 0 for normal) using a threshold of 0.5
    new_pred_binary = (new_scores >= 1).astype(int)
    print(f"Anomalies (New Pred sum): {new_pred_binary.sum()}")

    # Create DataFrame with all features and metadata
    df = pd.DataFrame(data_series_array, columns=rel_features)
    df['timestamp'] = date_values
    df["Prediction"] = new_pred_binary
    df["Ground Truth"] = gt
    df["Anomaly Energy"] = new_scores

    return df, first_date, rel_features, window_size


def get_anomaly_segments(pred_array: np.ndarray, threshold: int = 1) -> list[tuple[int, int]]:
    """
    Extract contiguous segments from the binary prediction array.
    Returns a list of (start, end) index tuples.

    Args:
        pred_array (np.ndarray): The binary prediction array.
        threshold (int): The threshold for detecting anomalies.

    Returns:
        list[tuple[int, int]]: List of anomaly segments.
    """
    segments = []
    anomaly_start = -1
    for i, val in enumerate(pred_array):
        if val >= threshold:
            if anomaly_start == -1:
                anomaly_start = i
        else:
            if anomaly_start != -1:
                segments.append((anomaly_start, i))
                anomaly_start = -1
    if anomaly_start != -1:
        segments.append((anomaly_start, len(pred_array)))
    return segments



def time_formatter(x, pos):
    """
    Formatter for y-axis: x is in hours.
    If x is 24 or more (i.e. representing next-day times), subtract 24.
    Then format as HH:MM.
    """
    # Adjust x if it is 24 or higher (i.e. wake_up values that are recorded as >24 hours)
    while x >= 24:
        x -= 24
    total_minutes = int(round(x * 60))
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"

def plot_anomaly_segment(seg_df: pd.DataFrame, anomaly_name: str, label: str, path: str, rel_features: list[str], debug: bool = False):
    """
    Plot the features for a given anomaly segment.
    Highlights the anomaly region (core in red, padded region in light red)
    and shades ground truth anomaly areas in green.

    Args:
        seg_df (pd.DataFrame): The DataFrame containing the data.
        anomaly_name (str): The anomaly name.
        label (str): The label for the segment.
        path (str): The path to save the plot.
        rel_features (list[str]): The relevant features.
        debug (bool): Whether to save the debug version of the plot.
    """
    time = seg_df["timestamp"].values
    pred = seg_df["Prediction"].values

    # Determine time unit and set pad_k, locator, and formatter based on the time granularity.
    if len(time) > 1:
        # Calculate delta in seconds
        delta_seconds = (time[1] - time[0]) / np.timedelta64(1, 's')
    else:
        delta_seconds = 60  # default to 1-minute intervals if not enough points

    if delta_seconds == 60:
        pad_k = 5
        locator = mdates.MinuteLocator(interval=1)
        formatter = mdates.DateFormatter('%H:%M')
    elif delta_seconds == 600:
        pad_k = 3
        locator = mdates.MinuteLocator(interval=10)
        formatter = mdates.DateFormatter('%H:%M')
    elif delta_seconds == 86400:
        pad_k = 0
        locator = mdates.DayLocator()
        formatter = mdates.DateFormatter('%Y-%m-%d')
    else:
        pad_k = 5
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')

    num_features = len(rel_features)
    fig, axs = plt.subplots(num_features, 1, figsize=(12, 4 * num_features), sharex=True)
    # fig.suptitle(f"Anomaly: {anomaly_types.get(anomaly_name, anomaly_name)} - Segment {label.split('_')[-1]}", fontsize=16)
    if num_features == 1:
        axs = [axs]

    for i, feature in enumerate(rel_features):
        ax = axs[i]
        # If the feature represents a time-of-day in minutes from midnight,
        # convert the values to hours for plotting.
        if feature in ["wake_up", "sleep_onset"]:
            # Here we convert minutes to hours (e.g. 1500 minutes becomes 25.0 hours)
            # so that if wake_up > sleep_onset (i.e. wake_up from next day),
            # the continuity is preserved.
            values = seg_df[feature].values / 60.0
        else:
            values = seg_df[feature].values

        ax.plot(time, values, color="#3499cd", linewidth=1.5)
        unit = feature_units.get(feature, "")
        ax.set_ylabel(f"{feature.replace('_', ' ').title()} ({unit})")
        ax.grid(True, linestyle='--', alpha=0.7)

        # Shade ground truth anomaly areas
        where_mask = seg_df["Ground Truth"].values == 1
        ax.fill_between(time, values.min(), values.max(), where=where_mask, color='green', alpha=0.3, label='Ground Truth Anomaly')

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', rotation=45)

        # If the feature is a time-of-day feature, set a custom formatter on the y-axis
        if feature in ["wake_up", "sleep_onset"]:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))

    # Highlight detected anomaly segments within the sliced window
    anomaly_segments = get_anomaly_segments(pred, threshold=1)
    for a, b in anomaly_segments:
        unit_time = time[1] - time[0]
        core_start_time = time[a] - unit_time / 2
        core_end_time = time[b - 1] + unit_time / 2
        padded_start_time = time[max(0, a - pad_k)]
        padded_end_time = time[min(len(time) - 1, b - 1 + pad_k)]
        for ax in axs:
            ax.axvspan(padded_start_time, padded_end_time, color='lightcoral', alpha=0.3,
                       label='Padded Anomaly Region' if ax == axs[0] else None)
            ax.axvspan(core_start_time, core_end_time, color='red', alpha=0.8,
                       label='Detected Anomaly' if ax == axs[0] else None)
    if axs[0].get_legend() is not None:
        axs[0].legend(loc='upper right', fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(path, exist_ok=True)
    filename = f"{label}_split_debug.png" if debug else f"{label}_split.png"
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.close()
    seg_df.to_csv(os.path.join(path, f"{label}_split.csv"), index=False)


def gpt_explain_segment(seg_df: pd.DataFrame, label: str, path: str, patient_info: str, data_type: str, anomaly_name: str):
    """
    Generate an Italian explanation using the explainer module.
    Includes the anomaly name in the prompt for better context.

    Args:
        seg_df (pd.DataFrame): The DataFrame containing the data.
        label (str): The label for the segment.
        path (str): The path to save the markdown file.
        patient_info (str): Information about the patient.
        data_type (str): The data type (PREATHOME or ATHOME or ATHOME_0).
        anomaly_name (str): The anomaly name.
    """
    # Select relevant columns (timestamp, sensor features present in field_descriptions, and Anomaly Energy)
    sensor_columns = ['timestamp'] + [col for col in seg_df.columns if col in field_descriptions] + ['Anomaly Energy']
    seg_df_filtered = seg_df[sensor_columns]

    # Generate explanation using GPT
    ita_explanation = gpt_explain(seg_df_filtered, patient_info)

    # Determine sensor type for the report
    sensor = "ATHOME" if data_type.startswith("ATHOME") else "PREATHOME"

    # Construct and save the Italian explanation markdown
    ita_input = (
        # f"### {anomaly_types.get(anomaly_name, anomaly_name)}\n\n"
        f"Il seguente campione di dati, raccolto tramite il sistema {sensor}, è stato analizzato.\n"
    )
    img_ref = f"![image]({os.path.join(path, f'{label}_split.png')})"
    page_break = "\n\n<div style='page-break-after: always;'></div>\n\n"
    with open(os.path.join(path, f"{label}_ita.md"), "w") as f:
        f.write(ita_input + img_ref + page_break + f"## Spiegazione Generata\n\n" + ita_explanation)


def render_pdf_all(base_dir: str, data_type: str, file_ext: str = "ita.md"):
    """
    Walk through all subfolders under the given base directory,
    collate markdown files (with the specified extension) and render them into a single PDF.
    Adds patient information to the PDF header.

    Args:
        base_dir (str): The base directory containing markdown files.
        data_type (str): The data type (PREATHOME or ATHOME or ATHOME_0).
        file_ext (str): The file extension to search for in subfolders.
    """
    nested_list_css = "li {list-style-type: disc;}"
    rating_css = (
        "table {border-collapse: collapse; width: 100%;}"
        "th, td {border: 1px solid black; padding: 8px; text-align: center;}"
        "th {background-color: #f2f2f2;}"
    )
    rating_table = (
        "\n\n## Valutazione del Medico \n"
        "\n| Domanda  | Valutazione  |\n"
        "| :--------------------------------------------------------------------------------------------------------------------------------- | :----------------: |\n"
        "| **Quanto ritiene che i dati mostrati rappresentino un'anomalia rispetto ai parametri fisiologici abituali del paziente?** Scala 0-5<br>(0 = Normale, 5 = Fortemente anomalo) |   |\n"
        "| **L'anomalia rilevata è clinicamente rilevante?** Scala 0-5<br>(0 = Non rilevante, 5 = Necessita di attenzione immediata) |   |\n"
        "| **Ritiene che tale anomalia debba essere mostrata sulla dashboard di monitoraggio del paziente?** (Opzioni: Sì / No) |   |\n"
    )
    pdf = MarkdownPdf()

    info = translate_to_italian(patients_info[data_type])
    patient_header = f"# Report Anomalie Paziente - {data_type}\n\n**Informazioni Paziente:** {info}\n\n"
    pdf.add_section(Section(patient_header, toc=False), user_css=nested_list_css)

    instructions = (
        "## Istruzioni per il Medico\n"
        "Questo documento presenta una serie di anomalie rilevate dal sistema di monitoraggio in base a uno o più parametri fisiologici. "
        "Per ogni anomalia, troverete un grafico dei dati rilevanti, evidenziando l'anomalia in rosso e le aree circostanti in rosa. "
        # "Le aree verdi indicano anomalie note dal ground truth (se disponibili). "
        "Segue una spiegazione generata automaticamente dal sistema basata sull'intelligenza artificiale GPT-4o.\n\n"
        "**Si prega di valutare ogni anomalia utilizzando la tabella di valutazione alla fine di ogni sezione.** \n"
        # "La vostra valutazione è fondamentale per migliorare l'accuratezza e l'utilità clinica del sistema.\n"
    )

    parameters_description = (
        "## Parametri e Unità di Misura\n\n"
        "### **Segnali Cardiovascolari**\n"
        "- **Systolic (Pressione Sistolica)** – mmHg\n"
        "- **Diastolic (Pressione Diastolica)** – mmHg\n"
        "- **Pulse (Frequenza Cardiaca)** – bpm\n"
        "- **HRV (Variabilità della Frequenza Cardiaca)** – ms\n\n"
        
        "### **Attività Fisica**\n"
        "- **Total Steps (Passi Totali)** – numero di passi/giorno\n"
        "- **Steps Morning (Passi Mattina)** – numero di passi\n"
        "- **Steps Afternoon (Passi Pomeriggio)** – numero di passi\n"
        "- **Steps Evening (Passi Sera)** – numero di passi\n"
        "- **Steps Night (Passi Notte)** – numero di passi\n\n"
        
        "### **Sonno**\n"
        "- **Sleep Duration (Durata del Sonno)** – ore\n"
        "- **Sleep Onset (Inizio Sonno)** – orario\n"
        "- **Wake Up (Risveglio)** – orario\n"
        "- **Deep Sleep Ratio (Rapporto Sonno Profondo)** – Durata del sonno profondo / Durata totale del sonno\n"
        "- **Fragmentation Index (Indice di Frammentazione del Sonno)** – numero di passaggi tra fasi del sonno\n"
        "- **Avg Session Duration (Durata Media Fasi del Sonno)** – ore\n"
        "- **Max Session Duration (Durata Massima Fasi del Sonno)** – ore\n"
        "- **Avg Deep Sleep Duration (Durata Media Fase Sonno Profondo)** – ore\n"
    )

    pdf.add_section(Section(instructions + parameters_description, toc=False), user_css=nested_list_css)

    i = 1
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(file_ext):
                anomaly_type_dir = os.path.basename(root)
                anomaly_name = anomaly_type_dir
                with open(os.path.join(root, file), "r") as f:
                    md_content = f.read()
                md_section = f"## Anomalia #{i}\n\n" + md_content + rating_table
                pdf.add_section(Section(md_section, toc=False), user_css=nested_list_css + rating_css)
                i += 1
    pdf.meta["title"] = f"Report Anomalie Paziente - {data_type} - Valutazione Medica"
    output_path = os.path.join(base_dir, f"report_anomalie_{data_type}_{file_ext.split('.')[0]}.pdf")
    pdf.save(output_path)
    print(f"PDF report saved to: {output_path}")


def process_anomaly_type(data_type: str, anomaly_id: int, start_idx: int = 0) -> int:
    """
    For a given data type and anomaly index:
      - Load the data.
      - Extract anomaly segments.
      - For a random subset of segments, process the segment (i.e. plot and generate GPT explanation)
        only if the padded window (min_idx-20, max_idx+20) does not overlap with a previously processed segment.
      - Continue until no segments remain or until we have processed ANOMALIES_PER_TYPE segments.

    Args:
        data_type (str): The data type to process (PREATHOME, ATHOME, or ATHOME_0).
        anomaly_id (int): The index of the anomaly to process.
        start_idx (int): The starting index for processing segments.

    Returns:
        int: The number of segments processed.
    """
    if data_type == "PREATHOME":
        anomaly_name = PREATHOME_ANOMALIES[anomaly_id]
    elif data_type.startswith("ATHOME"):
        anomaly_name = ATHOME_ANOMALIES[anomaly_id]
    else:
        raise ValueError("Unsupported data type")

    df, first_date, rel_features, window_size = load_data_all(data_type, anomaly_id)
    segments = get_anomaly_segments(df["Prediction"].values, threshold=1)
    print(f"{data_type} - {anomaly_name}: Detected segments: {segments}")

    output_dir = os.path.join(basepath, data_type, anomaly_name)
    os.makedirs(output_dir, exist_ok=True)

    patient_info = patients_info[data_type]

    # Process segments ensuring no padded region overlap
    processed_segments = 0
    used_intervals = []  # List to store padded intervals of already processed segments
    candidate_segments = np.random.permutation(segments)
    
    for min_idx, max_idx in candidate_segments:
        # Calculate padded window for context
        padded_start = max(0, min_idx - 20)
        padded_end = min(len(df), max_idx + 20)

        # Check for overlap with previously processed segments
        overlap = False
        for used_start, used_end in used_intervals:
            if padded_start <= used_end and used_start <= padded_end:
                overlap = True
                break
        if overlap:
            print(f"Skipping segment {(min_idx, max_idx)} due to overlap with previous segments.")
            continue

        # Process the segment
        label = f"{anomaly_name}_SEG_{processed_segments}"
        seg_path = os.path.join(output_dir, label)
        os.makedirs(seg_path, exist_ok=True)

        # Extract the segment with padding
        seg_df = df.iloc[padded_start:padded_end].copy()
        plot_anomaly_segment(seg_df, anomaly_name, label, seg_path, rel_features, debug=False)
        gpt_explain_segment(seg_df, label, seg_path, patient_info, data_type, anomaly_name)

        # Record this interval as used
        used_intervals.append((padded_start, padded_end))
        processed_segments += 1
        
        if processed_segments >= ANOMALIES_PER_TYPE:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Process all anomaly types for a given data type and generate plots and GPT explanations."
    )
    parser.add_argument("--data", type=str, help="Data type to process (ATHOME or PREATHOME)", default="ATHOME_0")
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(444)
    random.seed(444)

    # Determine which anomaly types to process
    if args.data == "PREATHOME":
        anomaly_ids = list(range(len(PREATHOME_ANOMALIES)))
    elif args.data.startswith("ATHOME"):
        anomaly_ids = list(range(len(ATHOME_ANOMALIES)))
    else:
        raise ValueError("Unsupported data type")

    # Process every anomaly type
    for anomaly_id in tqdm(anomaly_ids, desc=f"Processing {args.data} anomalies"):
        process_anomaly_type(args.data, anomaly_id)

    # Finally, render a PDF report collating all generated markdown explanations
    render_pdf_all(os.path.join(basepath, args.data), args.data, file_ext="ita.md")

if __name__ == "__main__":
    main()

