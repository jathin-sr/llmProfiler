import subprocess
import re
import time

def measure_training_power(training_cmd, duration_seconds=15):
    power_samples = []
    training_metrics = []
    
    power_cmd = f"sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n {duration_seconds}"
    power_process = subprocess.Popen(power_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    train_process = subprocess.Popen(training_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    start_train_time = time.time()
    while time.time() - start_train_time < duration_seconds:
        line = train_process.stdout.readline()
        if not line and train_process.poll() is not None:
            break 
        if line:
            training_metrics.append({
                'timestamp': time.time(),
                'output': line.strip()
            })
            print(f"  {line.strip()}")
    training_duration = time.time()-start_train_time

    power_output, _ = power_process.communicate()

    current_sample = {}
    for line in power_output.split('\n'):
        if 'CPU Power:' in line:
            match = re.search(r'CPU Power:\s+(\d+)\s*mW', line)
            if match:
                current_sample['cpu_power_mw'] = int(match.group(1))
        elif 'GPU Power:' in line:
            match = re.search(r'GPU Power:\s+(\d+)\s*mW', line)
            if match:
                current_sample['gpu_power_mw'] = int(match.group(1))
                if current_sample:
                    current_sample['timestamp'] = time.time()
                    power_samples.append(current_sample.copy())
                    current_sample = {}
    
    if train_process.poll() is None:
        train_process.terminate()
        train_process.wait()

    return power_samples, training_metrics, training_duration