import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scikit_posthocs as sp
import scipy.stats as stats
import seaborn as sns


VERSION = "final"
WANDB_USER = "wandb_user"
DATASETS = ["wesad", "hci", "dreamer"]
# AVERAGED = False
VALID_LRS = {
    "tadgan": "lr0.0005",
    "hypad": "lr0.005", 
    "units": "lr0.0005",
    "contextualunits": "",
    "usad": "lr0.0001",
    "cae_m": "lr0.001",
    "omnianomaly": "lr0.002",
    "mscred": "lr0.0001",
    "lstm_ad": "lr0.002",
    "tranad": "lr0.005",
    "mtad_gat": "lr0.0001",
    "gdn": "lr0.0001",
    "mad_gan": "lr0.0001",
    "dagmm": "lr0.0001",
}
METRICs = {
    "tranad": ["f1_score_ar3"],
    "units": ["f1_score_ar3"],
    "hypad": ["f1_score"],
    "tadgan": ["f1_score"],
}
AUCs = {
    "tranad": ["ROC/AUC", "PR/AUC"],
    "units": ["AUC_ROC", "AUC_PR"],
    "hypad": ["AUC_ROC", "AUC_PR"],
    "tadgan": ["AUC_ROC", "AUC_PR"],
}
RUNS = {
    "tranad": 25,
    "units": 25,
    "hypad": 25,
    "tadgan": 25,
}

api = wandb.Api()

def plot_confidence_intervals():
    models = ["UniTS"] + list(reversed(["MAD-GAN", "TadGAN", "USAD", "GDN", "DAGMM"]))
    datasets = ["DREAMER", "HCI", "WESAD (ECG)", "WESAD (BVP)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Load the dataset kf1s
    if "dataset_kf1s.npy" in os.listdir():
        with open("dataset_kf1s.npy", "rb") as f:
            model_kf1s = np.load(f, allow_pickle=True).item()
    else:
        # Get the F1 scores for each model in each dataset
        model_kf1s = {dataset: {model: [] for model in models} for dataset in datasets}
        for dataset in datasets:
            for model in models:
                d_name = dataset.split(" ")[0].lower()
                m_name = model.lower().replace("-", "_")
                if "units" in m_name:
                    project = f"{d_name}_units_pretrained_d128_kfold-win5-final"
                else:
                    project = f"{d_name}-{m_name}-kfold-final"
                _, l_kf1s = process_project(project, ignore_version=True)
                if "BVP" in dataset:
                    model_kf1s[dataset][model] = l_kf1s["BVP"]
                else:
                    model_kf1s[dataset][model] = l_kf1s["ECG"]
            
    # Save the dataset kf1s
    with open("dataset_kf1s.npy", "wb") as f:
        np.save(f, model_kf1s)

    print(model_kf1s["WESAD (ECG)"]["UniTS"])
    print(model_kf1s["WESAD (BVP)"]["UniTS"])

    # Plot for each dataset
    for i, dataset in enumerate(datasets):
        # Convert F1 scores to an array where each column is a model
        data_for_plotting = [model_kf1s[dataset][model] for model in models]

        # Set colors
        palette = ["lightgrey" for _ in models]
        # Highlight the best-performing model by median F1 score
        # best_model_idx = np.argmax([np.median(scores) for scores in data_for_plotting])
        palette[0] = "#3499CD"

        # Create a boxplot for each model's F1 scores in this dataset
        sns.boxplot(
            data=data_for_plotting, 
            ax=axes[i],
            # showfliers=False,
            palette=palette
        )
        
        # Set labels and title
        axes[i].set_title(dataset)
        axes[i].set_xticklabels(models)#, rotation=45)
        axes[i].set_ylabel("F1 Score")
        axes[i].set_ylim(0.4, 1.0)
        
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("boxplot.eps")

def posthocs():
    models = ["mad_gan", "tadgan", "usad", "gdn", "dagmm"]

    # For all the datasets, get list of best k-fold f1s
    model_kf1s = {model: [] for model in models + ["units"]}

    if "posthocs_kf1s.npy" in os.listdir():
        with open("posthocs_kf1s.npy", "rb") as f:
            model_kf1s = np.load(f, allow_pickle=True).item()
    else:
        for dataset in DATASETS:
            # Add units
            project = f"{dataset}_units_pretrained_d128_kfold-win5-final"
            _, l_kf1s = process_project(project, ignore_version=True)
            model_kf1s["units"] += l_kf1s["ECG"]
            if "wesad" in dataset:
                model_kf1s["units"] += l_kf1s["BVP"]

            # Add the models in the list
            for model in tqdm(models, desc=f"Processing {dataset}"):
                project = f"{dataset}-{model}-kfold-final"
                _, l_kf1s = process_project(project, ignore_version=True)
                model_kf1s[model] += l_kf1s["ECG"]
                if "wesad" in dataset:
                    model_kf1s[model] += l_kf1s["BVP"]

    # Save the dataset kf1s
    with open("posthocs_kf1s.npy", "wb") as f:
        np.save(f, model_kf1s)

    # Compute the posthocs (units, [models])
    data = []
    data.append(model_kf1s["units"])
    for model in models:
        if len(model_kf1s[model]) != len(model_kf1s["units"]):
            raise ValueError(f"Length of {model} is different from units")
        data.append(model_kf1s[model])

    p_values = sp.posthoc_dunn(data)
    print(f"Posthocs\n{p_values}")

    # Compute the friedman test
    f, p = stats.friedmanchisquare(*data)
    print(f"Friedman Test\nF: {f}, p: {p}")

def process_project(project_name: str, model: str = "tranad", csvs: list = [], auc: bool = False, ignore_version: bool = False):
    # Skip the project not related to this experiment
    if not any([dataset in project_name for dataset in DATASETS]):
        return
    # Skip the project if it is not the correct version (not end with VERSION)
    if not ignore_version and not project_name.endswith(VERSION):
        return

    # Get Model name
    for k in RUNS.keys():
        if k.lower() in project_name:
            model = k
            if model == "contextual":
                break

    # Get the runs from the project
    runs = api.runs(f"{WANDB_USER}/{project_name}")

    # Get other metrics
    metrics = METRICs[model]
    aucs = AUCs[model] if auc else []

    print(f"Processing {project_name} with model {model}, metrics: {metrics}, aucs: {aucs}")

    # Skip the project if there are missing runs
    if "wesad" in project_name:
        if len(runs) != RUNS[model]*2:
            print(f"Skipping {project_name} because of missing runs ({len(runs)}/{RUNS[model]*2})")
            # csv.write(f"{project_name},,,{len(runs)}\n")
            return
    else:
        if len(runs) != RUNS[model]:
            print(f"Skipping {project_name} because of missing runs ({len(runs)}/{RUNS[model]})")
            # csv.write(f"{project_name},,,{len(runs)}\n")
            return

    # Group the runs by the lr and source signal
    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}

    # For each run, add the run to the group
    for run in runs:
        # Tags of the run
        tags = run.tags

        # Get the lr and source signal
        lr = [tag for tag in tags if "lr" == tag[:2]][0]
        source = [tag for tag in tags if "ECG" == tag or "BVP" == tag][0]

        # Skip the run if the lr is not valid
        model_name = project_name.replace("_units_", "-units-").split("-")[1]
        if lr not in VALID_LRS[model_name]:
            # print(f"Skipping {project_name} because of invalid lr ({lr})")
            continue

        # Get the data from the run
        datas = [run.scan_history(keys=[metric]) for metric in metrics]
        if len(aucs) > 0:
            datas += [run.scan_history(keys=aucs)]
        datas = [pd.DataFrame(data) for data in datas]
        data = pd.concat(datas, axis=1)

        # Add the run to the group
        if (lr, source) not in groups:
            groups[(lr, source)] = []
        groups[(lr, source)].append(data)
    
    print(f"Project: {project_name} - Groups: {groups.keys()}")

    # For each group, get the average f1 based on best f1 in each fold
    avg_f1s: dict[str, np.ndarray] = {}
    std_f1s: dict[str, np.ndarray] = {}
    l_kf1s: dict[str, list[float]] = {} # list of f1s for each fold
    avg_aucrocs: dict[str, np.ndarray] = {}
    avg_aucprs: dict[str, np.ndarray] = {}
    for group, data_runs in groups.items():
        best_f1s = []
        best_aucrocs = []
        best_aucprs = []
        # For each fold
        for data in data_runs:
            # Get the last non-zero f1
            best_f1 = 0
            best_idx = 0
            for metric in metrics:
                f1s = data.loc[:, metric]
                f1s = f1s.astype(float)
                # Get the last non-zero index
                idx = f1s.last_valid_index()
                if idx is None:
                    f1 = 0
                else:
                    f1 = float(data.loc[idx, metric])
                best_f1 = f1
                best_idx = idx
            # Add the metrics to the lists
            best_f1s.append(best_f1)

            if len(aucs) > 0:
                # Get the corresponding aucroc and aucpr
                best_aucroc = float(data.loc[best_idx, aucs[0]])
                best_aucpr = float(data.loc[best_idx, aucs[1]])
                best_aucrocs.append(best_aucroc)
                best_aucprs.append(best_aucpr)
        # Compute the average f1, aucroc, and aucpr over the folds
        avg_f1s[group] = np.mean(best_f1s)
        # Compute the standard error
        std_f1s[group] = np.std(best_f1s)
        l_kf1s[group] = best_f1s
        avg_aucrocs[group] = np.mean(best_aucrocs)
        avg_aucprs[group] = np.mean(best_aucprs)

    print(f"Project: {project_name} - Average F1:\n {avg_f1s}")
    with open("results.txt", "a") as f:
        f.write(str({project_name: avg_f1s})+"\n")

    # For each source, get the best score among the lrs
    best_metrics: dict[str, list[float]] = {}
    best_l_kf1s: dict[str, list[float]] = {}
    best_lr: dict[str, str] = {}
    for source in ["ECG", "BVP"]:
        best_f1 = 0
        best_f1_std = 0
        best_aucroc = 0
        best_aucpr = 0

        # Get the best f1 among the lrs
        for (lr, src), f1 in avg_f1s.items():
            if src == source:
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_std = std_f1s[(lr, src)]
                    # Get the corresponding l_kf1s
                    best_l_kf1s[source] = l_kf1s[(lr, src)]
                    if len(aucs) > 0:
                        best_aucroc = avg_aucrocs[(lr, src)]
                        best_aucpr = avg_aucprs[(lr, src)]
                    # Save best lr
                    best_lr[source] = lr
        # if AVERAGED:
        #     best_f1 = np.mean([f1 for (lr, src), f1 in avg_f1s.items() if src == source])
        #     best_f1_std = np.mean([std for (lr, src), std in std_f1s.items() if src == source])
        #     best_l_kf1s[source] = [f1 for (lr, src), f1 in l_kf1s.items() if src == source] # Get all the f1s for posthocs (?!)
        #     if len(aucs) > 0:
        #         best_aucroc = np.mean([aucroc for (lr, src), aucroc in avg_aucrocs.items() if src == source])
        #         best_aucpr = np.mean([aucpr for (lr, src), aucpr in avg_aucprs.items() if src == source])
        best_metrics[source] = [best_f1, best_aucroc, best_aucpr, best_f1_std]

    # Print the best scores
    print(f"Project: {project_name} - Best Scores: {best_metrics}")
    print(f"Project: {project_name} - Best l_kf1s: {best_l_kf1s}")

    # Write the best scores to the csv
    if len(csvs) != len(DATASETS):
        print("ERROR: csvs should have the same length as DATASETS")
    else:
        for i in range(len(DATASETS)):
            if DATASETS[i] in project_name:
                # Format ECG metrics with 3 decimal places: F1, F1_STD, AUCROC, AUCPR
                ecg_f1 = f"{best_metrics['ECG'][0]:.3f}"
                ecg_f1_std = f"{best_metrics['ECG'][3]:.3f}"
                ecg_aucroc = f"{best_metrics['ECG'][1]:.3f}"
                ecg_aucpr = f"{best_metrics['ECG'][2]:.3f}"
                ecg_metrics = f"{ecg_f1},{ecg_f1_std},{ecg_aucroc},{ecg_aucpr}"
                
                # Format BVP metrics with 3 decimal places: F1, F1_STD, AUCROC, AUCPR
                bvp_f1 = f"{best_metrics['BVP'][0]:.3f}"
                bvp_f1_std = f"{best_metrics['BVP'][3]:.3f}"
                bvp_aucroc = f"{best_metrics['BVP'][1]:.3f}"
                bvp_aucpr = f"{best_metrics['BVP'][2]:.3f}"
                bvp_metrics = f"{bvp_f1},{bvp_f1_std},{bvp_aucroc},{bvp_aucpr}"
                
                print(f"{project_name},{ecg_metrics},{bvp_metrics},{len(runs)},{metric}\n")
                csvs[i].write(
                    f"{project_name},{ecg_metrics},{bvp_metrics},{len(runs)},{metric}\n")
                break
    return best_metrics, best_l_kf1s


def plot_step_projects(projects: list):
    """
    Plot the f1 scores of the step projects.
    """
    best_f1s = {
        "DREAMER (ECG)": [0]*6,
        "MAHNOB-HCI (ECG)": [0]*6,
        "WESAD (ECG)": [0]*6,
        "WESAD (BVP)": [0]*6,
    }
    symbols = ["o", "^", "s", "D"]
    for dataset in list(best_f1s.keys()):
        best_f1s[f"{dataset}_std"] = [0]*6

    # Read from file if it exists
    if "step_projects.csv" in os.listdir():
        # CSV is composed by columns: DREAMER (ECG), HCI (ECG), WESAD (ECG), WESAD (BVP)
        df = pd.read_csv("step_projects.csv")
        for key in best_f1s.keys():
            best_f1s[key] = df[key].tolist()
    else:
        for project_name in tqdm(projects, desc="Processing step projects"):
            try:
                # Get the dataset name
                dataset = project_name.split("_")[0].upper()
                dataset = dataset.replace("HCI", "MAHNOB-HCI")
                ecg_dataset = f"{dataset} (ECG)"
                bvp_dataset = f"{dataset} (BVP)"

                # Get the number of steps from "step*" in the project name
                if "step" not in project_name:
                    step = 0
                else:
                    step = project_name.split("-")[-3].replace("step", "")
                    step = int(step) - 1

                # Skip if the best f1 is already present
                if best_f1s[ecg_dataset][step] != 0:
                    continue

                # Process the project
                best_metrics, _ = process_project(project_name, ignore_version=True)
                if best_metrics is None:
                    print(f"Skipping {project_name}")
                    continue

                best_f1s[ecg_dataset][step] = best_metrics["ECG"][0]
                best_f1s[f"{ecg_dataset}_std"][step] = best_metrics["ECG"][3]

                # If BVP is present, add it to the best f1s
                if "wesad" in project_name:
                    best_f1s[bvp_dataset][step] = best_metrics["BVP"][0]
                    best_f1s[f"{bvp_dataset}_std"][step] = best_metrics["BVP"][3]
            except Exception as e:
                print(f"Error in processing project {project_name}: {e}")

            # Save the best f1s to a csv
            df = pd.DataFrame(best_f1s)
            df.to_csv("step_projects.csv", index=False)

    # Plot the best f1 scores with std area
    plt.figure(figsize=(10, 6))
    x_ticks = [f"{i*10}s" for i in range(1, 7)]
    filt_keys = filter(lambda x: "std" not in x, best_f1s.keys())
    for i, dataset in enumerate(filt_keys):
        plt.plot(x_ticks, best_f1s[dataset], label=dataset, marker=symbols[i])
        std = best_f1s[f"{dataset}_std"]
        std_err = np.array([std[i]/np.sqrt(5) for i in range(len(std))])
        plt.fill_between(x_ticks, np.array(best_f1s[dataset]) - std_err, np.array(best_f1s[dataset]) + std_err, alpha=0.2)
    plt.xlabel("Sampling interval (s)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()

    # Save fig as svg
    plt.savefig("step_projects.svg")
    plt.close()

    # Draw only the split [4, 5, 6] in a square plot
    plt.figure(figsize=(6, 6))
    x_ticks = [f"{i*10}s" for i in range(4, 7)]
    filt_keys = filter(lambda x: "std" not in x, best_f1s.keys())
    for i, dataset in enumerate(filt_keys):
        plt.plot(x_ticks, best_f1s[dataset][3:], label=dataset, marker=symbols[i])
        std = best_f1s[f"{dataset}_std"]
        std_err = np.array([std[i]/np.sqrt(5) for i in range(3, 6)])
        plt.fill_between(x_ticks, np.array(best_f1s[dataset][3:]) - std_err, np.array(best_f1s[dataset][3:]) + std_err, alpha=0.2)
    plt.xlabel("Sampling interval (s)")
    plt.ylabel("F1 Score")
    plt.ylim(0.0, 0.63)
    # plt.xlim(0,2)
    # plt.legend()

    # Save fig as svg
    plt.savefig("step_projects_zoomed.svg")



def main():
    # Get list of projects
    projects = api.projects(f"{WANDB_USER}")

    # CSV of best scores
    if "results.csv" in os.listdir():
        print("WARNING: results.csv already exists.")
        # Ask the user if they want to delete the file
        delete = input("Do you want to delete the file? (y/N): ")
        if delete.lower() == "y":
            os.remove("results.csv")
        else:
            return

    filenames = [f"results_{dataset}.csv" for dataset in DATASETS]
    # if AVERAGED:
    #     filenames = [filename.replace(".csv", "_avg.csv") for filename in filenames]
        
    csvs = [open(filename, "a") for filename in filenames]

    for csv in csvs:
        if csv.tell() == 0:
            csv.write("Project,ECG_F1,ECG_F1_STD,ECG_AUCROC,ECG_AUCPR,BVP_F1,BVP_F1_STD,BVP_AUCROC,BVP_AUCPR,Runs,Metric\n")

    # Read the csvs
    pds = [pd.read_csv(csv.name) for csv in csvs]

    # Filter out the projects having step* in the name
    step_projects = [project for project in projects if "-step" in project.name.lower() and project.name.endswith("v6")]
    projects = [project for project in projects if project not in step_projects]

    # Make plot for step projects
    step_projects = [project.name for project in step_projects]
    step_projects += [
        "wesad_units_pretrained_d128_kfold-win5-final",
        "hci_units_pretrained_d128_kfold-win5-final",
        "dreamer_units_pretrained_d128_kfold-win5-final"
    ]
    # plot_step_projects(step_projects)
    # raise ValueError("Step projects are done")

    # Process the projects
    for project in tqdm(projects, desc="Processing projects"):
        try:
            # Flush the csv
            for csv in csvs:
                csv.flush()

            # Get the name of the project
            project_name = project.name.lower()

            # Skip if the project is already in a pd
            if any([project_name in pd["Project"].tolist() for pd in pds]):
                print(f"Skipping {project_name}. Already in the csv.")
                continue

            # Process the project
            process_project(project_name, csvs=csvs)
        except Exception as e:
            print(f"Error in processing project {project.name}: {e}")

    # Close the csv
    csv.close()


if __name__ == "__main__":
    main()
    # posthocs()
    # plot_confidence_intervals()
