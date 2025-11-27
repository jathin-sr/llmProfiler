import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.profiling_helpers import run_profiling_experiment

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to profiling config YAML')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    config = load_config(args.config)
    results = run_profiling_experiment(config)

    # Saving
    output_dir = Path(args.output_dir)
    output_dir.mkdir( exist_ok=True)
    timestamp = datetime.now().strftime(config['output']['timestamp_format'])
    output_path = output_dir / f"{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    main()