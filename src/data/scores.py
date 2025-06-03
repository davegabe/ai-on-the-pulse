import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ------------------------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------------------------
ROOT_DIR = '../plots/all_anomalies'

ANOMALY_NAME_MAP = {
    "VS_HYPERHYPOTENSION": "Hyper/Hypotension",
    "VS_ABN_HRV": "Abnormal HRV",
    "VS_STRESS": "Stress",
    "SLP_INSOMNIA_HYPERSOMNIA": "Insomnia/Hypersomnia",
    "SLP_SLEEP_QUALITY": "Sleep Quality",
    "ACT_ABN_STEPS": "Abnormal Steps",
}

def rename_anomaly_dir(dirname):
    return ANOMALY_NAME_MAP.get(dirname, dirname)

def get_dataset_label(path):
    if 'ATHOME_0' in path:
        return 'Pilot'
    elif 'PREATHOME' in path:
        return 'Prod'
    return 'Unknown'

# ------------------------------------------------------------------------------
# 2) Gather Data with Filtering
# ------------------------------------------------------------------------------
def gather_data(root_dir=ROOT_DIR, get_max=False, include_types=None):
    rows = []
    for root, dirs, files in os.walk(root_dir):
        if 'evaluation.csv' in files:
            csv_path = os.path.join(root, 'evaluation.csv')
            df = pd.read_csv(csv_path)

            dataset_label = get_dataset_label(root)
            anomaly_dirname = os.path.basename(root)
            anomaly_type = rename_anomaly_dir(anomaly_dirname)

            # **Apply filter to include only selected anomaly types**
            if include_types and anomaly_type not in include_types:
                continue

            # Process *_split.csv to get anomaly energy
            results = []
            for i, row in df.iterrows():
                seg_folder = os.path.join(root, f"{anomaly_dirname}_SEG_{i}")
                seg_file = f"{anomaly_dirname}_SEG_{i}_split.csv"
                seg_path = os.path.join(seg_folder, seg_file)

                if os.path.exists(seg_path):
                    df_seg = pd.read_csv(seg_path)
                    df_pred1 = df_seg[df_seg["Prediction"] == 1]
                    val = df_pred1["Anomaly Energy"].max() if get_max else df_pred1["Anomaly Energy"].mean()
                else:
                    val = None  # If no file, use None

                results.append([
                    i + 1,  # Unique anomaly index
                    dataset_label,
                    anomaly_type,
                    val,
                    row['human_critic']  # Keep the human criticality score
                ])

            rows.extend(results)

    columns = ["Anomaly", "Dataset", "Type", "Score_Anomalia", "Score_Criticita"]
    return pd.DataFrame(rows, columns=columns)

# ------------------------------------------------------------------------------
# 3) Plot Function
# ------------------------------------------------------------------------------
def plot_scores(df):
    sns.set_theme(style="whitegrid", context="notebook", palette="deep", font_scale=1.2)

    group_stats = df.groupby(["Dataset", "Type"]).agg(
        mean_anomalia=('Score_Anomalia', 'mean'),
        std_anomalia=('Score_Anomalia', 'std'),
        mean_criticita=('Score_Criticita', 'mean'),
        std_criticita=('Score_Criticita', 'std')
    ).reset_index()

    print("Group statistics:\n", group_stats, "\n")

    plt.figure(figsize=(12, 8))
    error_bar_alpha = 0.25
    marker_size = 120
    edge_width = 1.5
    marker_alpha = 0.7
    jitter_amount = 0.02
    plt.gca().set_facecolor("#f8f9fa")

    dataset_markers = {'Pilot': 'o', 'Prod': 's'}
    unique_types = group_stats['Type'].unique()
    color_map = dict(zip(unique_types, plt.cm.tab10(np.linspace(0, 1, len(unique_types)))))

    point_positions = {}

    for _, row in group_stats.iterrows():
        x = row['mean_anomalia']
        y = row['mean_criticita']
        dataset = row['Dataset']
        anomaly_type = row['Type']

        key = (round(x, 2), round(y, 2))
        if key in point_positions:
            x += np.random.uniform(-jitter_amount, jitter_amount)
            y += np.random.uniform(-jitter_amount, jitter_amount)
        point_positions[key] = True

        marker = dataset_markers.get(dataset, 'x')
        color = color_map[anomaly_type]

        plt.scatter(x, y, marker=marker, color=color,
                    s=marker_size, edgecolor='black', linewidth=edge_width,
                    zorder=3, alpha=marker_alpha)

        plt.errorbar(x, y,
                     xerr=row['std_anomalia'],
                     yerr=row['std_criticita'],
                     fmt='none', ecolor=color, capsize=5,
                     alpha=error_bar_alpha, capthick=1.5, zorder=2)

    plt.grid(True, linestyle='--', alpha=0.7, zorder=1)
    plt.xlabel("Average Anomaly Energy (UniTS)")
    plt.ylabel("Average Criticality Score (Doctor)")
    # plt.title("Anomaly and Criticality Scores by Dataset and Type", fontsize=14, pad=15)

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    for spine in plt.gca().spines.values():
        spine.set_color('#cccccc')

    type_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=10, label=t, alpha=marker_alpha)
        for t, c in color_map.items()
    ]
    dataset_legend_elements = [
        Line2D([0], [0], marker=m, color='black',
               markersize=10, label=d, alpha=marker_alpha)
        for d, m in dataset_markers.items()
    ]
    plt.legend(handles=type_legend_elements + dataset_legend_elements,
            #    bbox_to_anchor=(1.05, 1), loc='upper left',
                loc="lower right",
               borderaxespad=0, frameon=True, fancybox=True,
               shadow=True, title='Legend')

    # plt.annotate(
    #     'Note: Points with same mean values have been slightly jittered for visibility',
    #     xy=(0.5, 0.01), xycoords='figure fraction',
    #     ha='center', va='bottom', fontsize=9,
    #     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    # )

    plt.xlim(0, 4)
    plt.ylim(0, 4)

    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/scores.pdf", dpi=300, bbox_inches='tight')

# ------------------------------------------------------------------------------
# 4) Main Driver with Options
# ------------------------------------------------------------------------------
def main(root_dir=ROOT_DIR, get_max=False, include_types=None):
    df = gather_data(root_dir=root_dir, get_max=get_max, include_types=include_types)
    if df.empty:
        print("No data found for the selected anomaly types.")
    else:
        plot_scores(df)

if __name__ == '__main__':
    selected_types = list(ANOMALY_NAME_MAP.values())
    selected_types.remove("Sleep Quality")
    use_max_energy = True
    main(root_dir=ROOT_DIR, get_max=use_max_energy, include_types=selected_types)
