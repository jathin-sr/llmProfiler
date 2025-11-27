import re
from collections import defaultdict

def parse_training_metrics(training_metrics, components):
    step_times = []
    losses = []
    profile_data = defaultdict(list)
    
    for metric in training_metrics:
        line = metric['output']
        
        if 'time' in line and 'ms' in line:
            time_match = re.search(r'time\s+([\d.]+)ms', line)
            if time_match:
                step_times.append(float(time_match.group(1)))

        if 'loss' in line and 'iter' in line:
            loss_match = re.search(r'loss\s+([\d.]+)', line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
        
        for component in components:
            pattern = rf"{component}:\s+([\d.]+)s"
            match = re.search(pattern, line)
            if match:
                total_seconds = float(match.group(1))
                profile_data[component].append(total_seconds)
    
    return step_times, losses, profile_data