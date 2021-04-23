import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", help="Number of nodes in the graph to generate. (default=20)")
args = parser.parse_args()

if args.nodes is None:
    args.nodes = 20

n = args.nodes 
df_mean = pd.DataFrame()
x = list(map(lambda x: round(x,2),np.linspace(0.05,0.95,19)))
ax = list(map(lambda x: round(x,2),np.linspace(0,1,11)))
for p in x:

    folder = f"../data/({n}_{p})/"

    data_file_names = os.listdir(folder)
    result_file_names = [fn for fn in data_file_names if fn.startswith('results_')]

    mean_result_file_names = [
        fn for fn in data_file_names if fn.startswith('results_mean_')]

    df = pd.DataFrame()
    for fn in mean_result_file_names:
        df = df.append(pd.read_pickle(f"{folder}{fn}"), ignore_index=True)

    print(f"(n,p) = ({n},{p}) : Based on {df.shape[0]} iterations.")
    df_mean = df_mean.append(df.mean(), ignore_index=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# (0.1,0.9,9) -> (0.05,0.95,18), wenn die 0.05er dazukommen
plt.plot(x,df_mean['reach_precision'], color='red', linewidth=3, marker='*', markersize=10)
plt.plot(x,df_mean['initH_precision'], color='darkblue', linewidth=3, marker='o', markersize=10)
plt.plot(x,df_mean['iterH_precision'], color='orange', linestyle='dashed', linewidth=2, marker='x', markersize=10)
plt.plot(x,df_mean['deg_precision'], color='green', linewidth=3, marker='+', markersize=10)
plt.xticks(ax, rotation=45, fontsize=20)
plt.yticks(ax, fontsize=20) 
plt.gca().invert_xaxis()
plt.legend(['ARS', 'Initial HS', 'Iterative HS', 'IRS'], loc='lower left', prop={'size':16})

plt.title(f'n = {n}', fontsize=32)
plt.xlabel(r'\textit{Edge Probability p}', fontsize=32)
plt.ylabel(r'\textit{Precision}', fontsize=32)
plt.grid(True)
plt.tight_layout()


if not os.path.exists(f"../img/({n})/"):
        os.mkdir(f"../img/({n})/")
plt.savefig(f'../img/({n})/precision.pdf')


plt.figure()
plt.plot(x,df_mean['reach_recall'], color='red', linewidth=3, marker='*', markersize=10)
plt.plot(x,df_mean['initH_recall'], color='darkblue', linewidth=3, marker='o', markersize=10)
plt.plot(x,df_mean['iterH_recall'], color='orange', linestyle='dashed', linewidth=2, marker='x', markersize=10)
plt.plot(x,df_mean['deg_recall'], color='green', linewidth=3, marker='+', markersize=10)
plt.xticks(ax, rotation=45, fontsize=20)
plt.yticks(ax, fontsize=20) 
plt.gca().invert_xaxis()
plt.legend(['ARS', 'Initial HS', 'Iterative HS', 'IRS'], loc='lower left', prop={'size':16})

plt.title(f'n = {n}', fontsize=32)
plt.xlabel(r'\textit{Edge Probability p}', fontsize=32)
plt.ylabel(r'\textit{Recall}', fontsize=32)
plt.grid(True)
plt.tight_layout()

if not os.path.exists(f"../img/({n})/"):
        os.mkdir(f"../img/({n})/")
plt.savefig(f'../img/({n})/recall.pdf')