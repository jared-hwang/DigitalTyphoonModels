import glob
from pathlib import Path
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process logs.')
parser.add_argument('--logroot', required=True, type=str, help='path to the log directory')
args = parser.parse_args()
logroot = Path(args.logroot)

# absolute path to search all text files inside a specific folder
path = str(logroot / "*.txt")
files = glob.glob(path)

losses = []
acc = []
micro_f1 = []
macro_f1 = []
weighted_f1 = []
for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()[-5:]
        losses.append(float(re.findall("\d+\.\d+",lines[0])[0]))
        acc.append(float(re.findall("\d+\.\d+",lines[1])[0]))
        micro_f1.append(float(re.findall("\d+\.\d+",lines[2])[0]))
        macro_f1.append(float(re.findall("\d+\.\d+",lines[3])[0]))
        weighted_f1.append(float(re.findall("\d+\.\d+",lines[4])[0]))

print(losses)
print()

print(acc)
print(np.max(acc), np.min(acc), np.mean(acc), np.std(acc))
print()

print(micro_f1)
print(np.max(micro_f1), np.min(micro_f1), np.mean(micro_f1), np.std(micro_f1))
print()

print(macro_f1)
print(np.max(macro_f1), np.min(macro_f1), np.mean(macro_f1), np.std(macro_f1))
print()

print(weighted_f1)
print(np.max(weighted_f1), np.min(weighted_f1), np.mean(weighted_f1), np.std(weighted_f1))
