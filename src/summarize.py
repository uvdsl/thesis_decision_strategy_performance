import os
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", help="Number of nodes in the graph to generate. (default=10)")
parser.add_argument("-p", "--probability", help="Probability of an edge to exist in the graph to generate. (default=0.1)")
args = parser.parse_args()

if args.nodes is None:
    args.nodes = 10

if args.probability is None:
    args.probability = 0.1



folder = f"../data/({args.nodes}_{args.probability })/"

data_file_names = os.listdir(folder)
result_file_names = [fn for fn in data_file_names if fn.startswith('results_')]




mean_result_file_names = [
    fn for fn in data_file_names if fn.startswith('results_mean_')]

df = pd.DataFrame()
for fn in mean_result_file_names:
    df = df.append(pd.read_pickle(f"{folder}{fn}"), ignore_index=True)


print(f"Based on {df.shape[0]} iterations:")
print()
print(df.mean())
print()












def print_results(analysis_results):
    analysis_results_compact = np.array(
        list(map(np.all, analysis_results)), dtype=bool)
    print()
    print()
    print("Total decisions:\t", len(analysis_results_compact))
    print("------------------------------")
    print("Optimal (all aggreed):\t", np.sum(analysis_results_compact))
    print("Where not all aggreed:\t", len(
        analysis_results_compact)-np.sum(analysis_results_compact))
    print()
    not_all_positive_indicies = np.where(analysis_results_compact == False)
    # extract not all positive cases:
    not_all_positive_results_bool = analysis_results[not_all_positive_indicies]
    print()
    print()
    print("Decisions where _ was optimal and one other wasn't:")
    print("------------------------------")
    reach_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 0] == True))
    print("Reach:\t\t", reach_positive_indicies.size)
    iter_score_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 1] == True))
    print("Iter Score:\t", iter_score_positive_indicies.size)
    init_score_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 2] == True))
    print("Init Score:\t", init_score_positive_indicies.size)
    init_deg_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 3] == True))
    print("Init Degree\t", init_deg_positive_indicies.size)
    print()
    print()
    print()
    print("Pairwise Performance Comparison: (true, while the other false)")
    print("------------------------------")
    print("Reach")
    both = np.sum(not_all_positive_results_bool[:,0] & not_all_positive_results_bool[:,1]) # and
    r = np.sum(not_all_positive_results_bool[:,0] > not_all_positive_results_bool[:,1])
    s = np.sum(not_all_positive_results_bool[:,0] < not_all_positive_results_bool[:,1]) 
    none = np.sum(~not_all_positive_results_bool[:,0] & ~not_all_positive_results_bool[:,1]) # nor
    print("\t(both,Reach,IterScore,none):\t",(both,r,s,none))
    both = np.sum(not_all_positive_results_bool[:,0] & not_all_positive_results_bool[:,2]) # and
    r = np.sum(not_all_positive_results_bool[:,0] > not_all_positive_results_bool[:,2])
    s = np.sum(not_all_positive_results_bool[:,0] < not_all_positive_results_bool[:,2]) 
    none = np.sum(~not_all_positive_results_bool[:,0] & ~not_all_positive_results_bool[:,2]) # nor
    print("\t(both,Reach,InitScore,none):\t",(both,r,s,none))
    both = np.sum(not_all_positive_results_bool[:,0] & not_all_positive_results_bool[:,3]) # and
    r = np.sum(not_all_positive_results_bool[:,0] > not_all_positive_results_bool[:,3])
    s = np.sum(not_all_positive_results_bool[:,0] < not_all_positive_results_bool[:,3]) 
    none = np.sum(~not_all_positive_results_bool[:,0] & ~not_all_positive_results_bool[:,3]) # nor
    print("\t(both,Reach,InitDeg,none):\t",(both,r,s,none))
    print("IterScore")
    both = np.sum(not_all_positive_results_bool[:,1] & not_all_positive_results_bool[:,2]) # and
    r = np.sum(not_all_positive_results_bool[:,1] > not_all_positive_results_bool[:,2])
    s = np.sum(not_all_positive_results_bool[:,1] < not_all_positive_results_bool[:,2]) 
    none = np.sum(~not_all_positive_results_bool[:,1] & ~not_all_positive_results_bool[:,2]) # nor
    print("\t(both,IterScore,InitScore,none):",(both,r,s,none))
    both = np.sum(not_all_positive_results_bool[:,1] & not_all_positive_results_bool[:,3]) # and
    r = np.sum(not_all_positive_results_bool[:,1] > not_all_positive_results_bool[:,3])
    s = np.sum(not_all_positive_results_bool[:,1] < not_all_positive_results_bool[:,3]) 
    none = np.sum(~not_all_positive_results_bool[:,1] & ~not_all_positive_results_bool[:,3]) # nor
    print("\t(both,IterScore,InitDeg,none):\t",(both,r,s,none))
    print("InitScore")
    both = np.sum(not_all_positive_results_bool[:,2] & not_all_positive_results_bool[:,3]) # and
    r = np.sum(not_all_positive_results_bool[:,2] > not_all_positive_results_bool[:,3])
    s = np.sum(not_all_positive_results_bool[:,2] < not_all_positive_results_bool[:,3]) 
    none = np.sum(~not_all_positive_results_bool[:,2] & ~not_all_positive_results_bool[:,3]) # nor
    print("\t(both,InitScore,InitDeg,none):\t",(both,r,s,none))

  


  
# eq_result_file_names = [
#     fn for fn in data_file_names if fn.startswith('results_eq_')]

# df = pd.DataFrame()
# for fn in eq_result_file_names:
#     df = df.append(pd.read_pickle(f"{folder}{fn}"), ignore_index=True)



# print()
# print("Analysing...")
# print()

# print("###########")
# print("# Equality #")
# print("###########")
# print_results(df.to_numpy()) # eig numpy arrays

# print("\n\n")

sub_result_file_names = [
    fn for fn in data_file_names if fn.startswith('results_sub_')]

df = pd.DataFrame()
for fn in sub_result_file_names:
    df = df.append(pd.read_pickle(f"{folder}{fn}"), ignore_index=True)
    print(fn)

print("###########")
print("# Subsets #")
print("###########")
print_results(df.to_numpy()) # eig numpy arrays

print("\n\n")

# print("###########")
# print("# Superset #")
# print("###########")

# df.to_pickle(f"./data/results_sup_{name}.pkl") 
# df = pd.DataFrame(analysis_results, columns=['reach','iterH','initH','deg'])
# print_results(analysis_results)


