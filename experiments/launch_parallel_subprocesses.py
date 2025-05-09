import subprocess

sample_sizes = [10000, 25000, 30000, 50000, 75000, 90000]
processes = []

for s in sample_sizes:
    cmd = ["python3", "run_sample_sweep.py", "--num_samples", str(s)]
    p = subprocess.Popen(cmd)
    processes.append(p)

print("Launched", len(processes), "subprocesses.")
