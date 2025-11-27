import glob
import pandas as pd
import os

def load_latest_results(results_dir='results'):
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    latest_file = max(csv_files, key=os.path.getmtime)
    df = pd.read_csv(latest_file)

    return df, latest_file