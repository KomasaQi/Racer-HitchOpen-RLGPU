from matplotlib import pyplot as plt
from pandas import read_csv
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import glob
import os
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
args = parser.parse_args()

list_of_files = [f for f in glob.glob('data/*') if not os.path.basename(f).startswith('traj')]
latest_file = max(list_of_files, key=os.path.getctime)
data = torch.load(latest_file, weights_only=False)

list_of_files_traj = glob.glob('data/traj*')
latest_file_traj = max(list_of_files_traj, key=os.path.getctime)
data_traj = read_csv(latest_file_traj, header=None)

filename = latest_file if args.filename is None else args.filename
pedals = all(len(u) == 3 for (s, u, es, o) in data)

fig, axs = plt.subplots(11, 1, sharex=True, figsize=(10, 18))
fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.95)

axs[0].plot([s[0] for (s, u, es, o) in data], label="x")
axs[0].set_ylabel(r"$x$")
axs[1].plot([s[1] for (s, u, es, o) in data], label="y")
axs[1].set_ylabel(r"$y$")
axs[2].plot([s[2] for (s, u, es, o) in data], label="psi")
axs[2].set_ylabel(r"$\psi$")
axs[3].plot([es[2] for (s, u, es, o) in data])
axs[3].set_ylabel(r"$r$")
axs[4].plot([es[7] for (s, u, es, o) in data])
axs[4].set_ylabel(r"$\beta$")
axs[5].plot([es[6] for (s, u, es, o) in data])
axs[5].set_ylabel(r"$V$")

axs[6].plot([u[0] for (s, u, es, o) in data])
axs[6].set_ylabel(r"$\delta$")
if pedals:
    axs[7].plot([u[1] for (s, u, es, o) in data], label="throttle")
    axs[7].plot([u[2] for (s, u, es, o) in data], label="brake")
else:
    axs[7].plot([u[1] for (s, u, es, o) in data], label="r")
    axs[7].plot([u[2] for (s, u, es, o) in data], label="l")
axs[7].legend()
axs[7].set_ylabel(r"input")

# actual wheel surface speeds (state), front / rear
axs[8].plot([es[8] for (s, u, es, o) in data], label="FL")
axs[8].plot([es[9] for (s, u, es, o) in data], label="FR")
axs[8].plot([es[10] for (s, u, es, o) in data], label="RL")
axs[8].plot([es[11] for (s, u, es, o) in data], label="RR")
axs[8].set_ylabel(r"$v_{wheel}$")
axs[8].legend()

axs[9].plot([es[16] for (s, u, es, o) in data], label="FL")
axs[9].plot([es[17] for (s, u, es, o) in data], label="FR")
axs[9].set_ylabel(r"$sf_x$")
axs[9].legend()

axs[10].plot([es[18] for (s, u, es, o) in data], label="RL")
axs[10].plot([es[19] for (s, u, es, o) in data], label="RR")
axs[10].set_ylabel(r"$sr_x$")
axs[10].legend()

axs[10].set_xlabel("Step")
plt.savefig(f"{filename}_state.png", dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1)
x = [s[0] for (s, u, es, o) in data]
y = [s[1] for (s, u, es, o) in data]
psi = [s[2] for (s, u, es, o) in data]
ax.plot(x, y)
ax.plot(data_traj[0], data_traj[1], 'r--')
ax.set_aspect('equal', 'box')
indices = [i for i in range(0, len(data), max(1, len(data) // 30))]
for idx in indices:
    x_, y_ = x[idx], y[idx]
    psi_ = psi[idx] * 180 / np.pi
    length = 0.15 * (max(x) - min(x) + 1e-3) / 10
    dx = length * np.cos(np.radians(psi_))
    dy = length * np.sin(np.radians(psi_))
    ax.arrow(x_, y_, dx, dy, head_width=length * 0.35, head_length=length * 0.35, fc='r', ec='r')
plt.savefig(f"{filename}_trajectory.png")
