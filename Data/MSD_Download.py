import multiprocessing
from joblib import Parallel, delayed
from monai.apps.datasets import DecathlonDataset
from MSD import MSD_PATH

TASK_LOOKUP = {
    "Task01": "Task01_BrainTumour",
    "Task02": "Task02_Heart",
    "Task03": "Task03_Liver",
    "Task04": "Task04_Hippocampus",
    "Task05": "Task05_Prostate",
    "Task06": "Task06_Lung",
    "Task07": "Task07_Pancreas",
    "Task08": "Task08_HepaticVessel",
    "Task09": "Task09_Spleen",
    "Task10": "Task10_Colon",
}


def download(task):
    DecathlonDataset(
        MSD_PATH, task, "training", download=True, cache_num=0,
    )


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(download)(task) for task in TASK_LOOKUP.values())

