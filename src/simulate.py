import os
import argparse
from _hits import calculate_hub_scores
from _transform import transform_states_to_index, transform_decision_states_to_index
from collections import deque
import networkx as nx
import numpy as np
import pandas as pd
from time import time, ctime



def generate(n,p):
    while True:
        G = nx.gnp_random_graph(node_count,p,directed=True)
        m = np.tril(nx.to_numpy_matrix(G))
        G = nx.from_numpy_matrix(m, create_using=nx.DiGraph)

        if nx.is_weakly_connected(G) and nx.is_directed_acyclic_graph(G):
            G_t = nx.transitive_closure(G)
            M = nx.to_numpy_array(G, dtype=bool)
            M_t = nx.to_numpy_array(G_t, dtype=bool)
            return M, M_t

#####



#####





def transform_states_to_index(states):
    state_index_dict = dict()
    for index in range(0,len(states)):
        state_index_dict.update({tuple(states[index]):index})
    return state_index_dict

def transform_decision_states_to_index(decision_states):
    # decision state = [ (index, a state), (index, a state), (index, b state)]
    global_decision_indicies = dict()
    for index in range(0,len(decision_states)):
        current_state = tuple(decision_states[index])
        if global_decision_indicies.get(current_state) is None:
            global_decision_indicies.update({current_state:[]})
        global_decision_indicies[current_state].append(index)
    return global_decision_indicies

def get_decisions_in_state_from_index(current_state):
    global decisions,global_decision_indicies
    return decisions[global_decision_indicies.get(tuple(current_state))]

def get_reaches_in_state_from_index(current_state):
    global reach,global_decision_indicies
    return reach[global_decision_indicies.get(tuple(current_state))]

#####

def eval_optimal_decision(current_state):
    global hubs
    options = get_decisions_in_state_from_index(current_state)
    current_decision_options = get_reaches_in_state_from_index(current_state)
    scores = np.empty((current_decision_options.shape[0],1), dtype=float)
    for reach_index in range(len(current_decision_options)):
        next_reach = current_decision_options[reach_index]
        next_state = np.array(current_state | next_reach).flatten()
        next_state_index = state_index_dict.get(tuple(next_state))
        scores[reach_index] = hubs.iloc[next_state_index, -1]
    opt_score_value = np.max(scores)
    opt_score_decisions = options[np.where(np.array(scores).flatten() == opt_score_value)]
    return opt_score_decisions, opt_score_value


def eval_reach_ordering_decision(current_state):
    options = get_decisions_in_state_from_index(current_state)
    current_decision_options = get_reaches_in_state_from_index(current_state)
    current_decision_reach = current_decision_options.sum(axis=1)
    max_reach_value = np.max(current_decision_options.sum(axis=1))
    max_reach_decisions = options[np.where(current_decision_options.sum(axis=1) == max_reach_value)]
    return max_reach_decisions, max_reach_value


def eval_iter_hub_odering_decision(current_state):
    global hubs
    options = get_decisions_in_state_from_index(current_state)
    current_state_index = state_index_dict.get(tuple(current_state))
    hub_scores = hubs.iloc[current_state_index,options]
    max_score = np.max(hub_scores)
    max_score_decisions = options[np.where(np.array(hub_scores).flatten() == max_score)]
    return max_score_decisions, max_score


def eval_init_hub_ordering_decision(current_state):
    global hubs
    options = get_decisions_in_state_from_index(current_state)
    source_state = np.zeros_like(current_state)
    source_state[-1] = True
    current_state_index = state_index_dict.get(tuple(current_state))
    hub_scores = hubs.iloc[current_state_index,options]
    max_score = np.max(hub_scores)
    max_score_decisions = options[np.where(np.array(hub_scores).flatten() == max_score)]
    return max_score_decisions, max_score

def eval_init_deg_ordering_decision(current_state):
    global M_t
    options = get_decisions_in_state_from_index(current_state)
    deg = M_t.sum(axis=1)[options]
    max_deg_value = np.max(deg)
    max_deg_decisions = options[np.where(deg == max_deg_value)]
    return max_deg_decisions, max_deg_value


def eval_current_state_strategies(current_state):
    opt_score_decisions, opt_score = eval_optimal_decision(current_state)
    # # print(opt_score_decisions, opt_score)
    max_reach_nodes, max_reach_value = eval_reach_ordering_decision(
        current_state)
    # # print(max_reach_nodes, max_reach_value)
    max_iter_score_decisions, max_iter_score = eval_iter_hub_odering_decision(
        current_state)
    # # print(max_iter_score_decisions, max_iter_score)
    max_init_score_decisions, max_init_score = eval_init_hub_ordering_decision(
        current_state)
    # # print(max_init_score_decisions, max_init_score)
    max_init_deg_decisions, max_init_deg = eval_init_deg_ordering_decision(
        current_state)
    # # print(max_init_deg_decisions, max_init_deg)
    return [
        np.asarray(list(current_state)),
        np.asarray(opt_score_decisions),
        opt_score,
        np.asarray(max_reach_nodes),
        max_reach_value,
        np.asarray(max_iter_score_decisions),
        max_iter_score,
        np.asarray(max_init_score_decisions),
        max_init_score,
        np.asarray(max_init_deg_decisions),
        max_init_deg
    ]


def analyse_strategy_equality(state_strategy):
    return np.asarray([
        # solution set must be optimal
        np.array_equal(state_strategy[1], state_strategy[3]),  # reach
        np.array_equal(state_strategy[1], state_strategy[5]),  # itersc
        np.array_equal(state_strategy[1], state_strategy[7]),  # initsc
        np.array_equal(state_strategy[1], state_strategy[9]),  # deg
    ], dtype=bool)

def analyse_strategy_subset_ok(state_strategy):
    opt = set(state_strategy[1])
    return np.asarray([
        # subsets are ok
        opt.issuperset(state_strategy[3]),  # if opt is superset of solution set then solution set is subset of opt.
        opt.issuperset(state_strategy[5]),
        opt.issuperset(state_strategy[7]),
        opt.issuperset(state_strategy[9]),
    ], dtype=bool)

def analyse_strategy_superset_ok(state_strategy):
    opt = set(state_strategy[1])
    return np.asarray([
        # e.g. reach finds superset instead of only optimals
        opt.issubset(state_strategy[3]),
        opt.issubset(state_strategy[5]),
        opt.issubset(state_strategy[7]),
        opt.issubset(state_strategy[9]),
    ], dtype=bool)


def print_results(analysis_results):
    analysis_results_compact = np.array(
        list(map(np.all, analysis_results)), dtype=bool)
    # print()
    # print()
    # print("Total decisions:\t", len(analysis_results_compact))
    # print("------------------------------")
    # print("Optimal (all aggreed):\t", sum(analysis_results_compact))
    # print("Where not all aggreed:\t", len(analysis_results_compact)-sum(analysis_results_compact))
    # print()
    not_all_positive_indicies = np.where(analysis_results_compact == False)
    # extract not all positive cases:
    not_all_positive_strats = np.array(state_strategies, dtype=object)[
        not_all_positive_indicies]
    not_all_positive_results_bool = analysis_results[not_all_positive_indicies]
    # print()
    # print()
    # print("Decisions where _ was optimal and one other wasn't:")
    # print("------------------------------")
    # # print("Reach:\t\t",len(not_all_positive_strats[np.where(not_all_positive_results[:,0] == True)]))
    reach_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 0] == True))
    # print("Reach:\t\t", reach_positive_indicies.size)
    iter_score_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 1] == True))
    # print("Iter Score:\t", iter_score_positive_indicies.size)
    init_score_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 2] == True))
    # print("Init Score:\t", init_score_positive_indicies.size)
    init_deg_positive_indicies = np.asarray(
        np.where(not_all_positive_results_bool[:, 3] == True))
    # print("Init Degree\t", init_deg_positive_indicies.size)
    # print()
    # print()


    # print()
    # print("Pairwise Performance Comparison: (true, while the other false)")
    # print("------------------------------")
    # print("Reach")
    both = np.sum(not_all_positive_results_bool[:,0] & not_all_positive_results_bool[:,1]) # and
    r = np.sum(not_all_positive_results_bool[:,0] > not_all_positive_results_bool[:,1])
    s = np.sum(not_all_positive_results_bool[:,0] < not_all_positive_results_bool[:,1]) 
    none = np.sum(~not_all_positive_results_bool[:,0] & ~not_all_positive_results_bool[:,1]) # nor
    # print("\t(both,Reach,IterScore,none):\t",(both,r,s,none))
    both = np.sum(not_all_positive_results_bool[:,0] & not_all_positive_results_bool[:,2]) # and
    r = np.sum(not_all_positive_results_bool[:,0] > not_all_positive_results_bool[:,2])
    s = np.sum(not_all_positive_results_bool[:,0] < not_all_positive_results_bool[:,2]) 
    none = np.sum(~not_all_positive_results_bool[:,0] & ~not_all_positive_results_bool[:,2]) # nor
    # print("\t(both,Reach,InitScore,none):\t",(both,r,s,none))
    both = np.sum(not_all_positive_results_bool[:,0] & not_all_positive_results_bool[:,3]) # and
    r = np.sum(not_all_positive_results_bool[:,0] > not_all_positive_results_bool[:,3])
    s = np.sum(not_all_positive_results_bool[:,0] < not_all_positive_results_bool[:,3]) 
    none = np.sum(~not_all_positive_results_bool[:,0] & ~not_all_positive_results_bool[:,3]) # nor
    # print("\t(both,Reach,InitDeg,none):\t",(both,r,s,none))
    # print("IterScore")
    both = np.sum(not_all_positive_results_bool[:,1] & not_all_positive_results_bool[:,2]) # and
    r = np.sum(not_all_positive_results_bool[:,1] > not_all_positive_results_bool[:,2])
    s = np.sum(not_all_positive_results_bool[:,1] < not_all_positive_results_bool[:,2]) 
    none = np.sum(~not_all_positive_results_bool[:,1] & ~not_all_positive_results_bool[:,2]) # nor
    # print("\t(both,IterScore,InitScore,none):",(both,r,s,none))
    both = np.sum(not_all_positive_results_bool[:,1] & not_all_positive_results_bool[:,3]) # and
    r = np.sum(not_all_positive_results_bool[:,1] > not_all_positive_results_bool[:,3])
    s = np.sum(not_all_positive_results_bool[:,1] < not_all_positive_results_bool[:,3]) 
    none = np.sum(~not_all_positive_results_bool[:,1] & ~not_all_positive_results_bool[:,3]) # nor
    # print("\t(both,IterScore,InitDeg,none):\t",(both,r,s,none))
    # print("InitScore")
    both = np.sum(not_all_positive_results_bool[:,2] & not_all_positive_results_bool[:,3]) # and
    r = np.sum(not_all_positive_results_bool[:,2] > not_all_positive_results_bool[:,3])
    s = np.sum(not_all_positive_results_bool[:,2] < not_all_positive_results_bool[:,3]) 
    none = np.sum(~not_all_positive_results_bool[:,2] & ~not_all_positive_results_bool[:,3]) # nor
    # print("\t(both,InitScore,InitDeg,none):\t",(both,r,s,none))



#####





def calculate_pr(ground_truth, dec):
    tp = 0
    fp = 0
    fn = 0
    for titem in ground_truth:
        if titem in dec:
            tp += 1
        else:
            fn += 1

    fp = dec.size - tp # rest of decisions is fp
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return precision, recall




def analyse_strategy_pr(state_strategy):
    result = np.zeros(8, dtype=float)
    p, r = calculate_pr(state_strategy[1], state_strategy[3])  # reach
    result[0] = p
    result[1] = r
    p, r = calculate_pr(state_strategy[1], state_strategy[5])  # itersc
    result[2] = p
    result[3] = r
    p, r = calculate_pr(state_strategy[1], state_strategy[7])  # initsc
    result[4] = p
    result[5] = r
    p, r = calculate_pr(state_strategy[1], state_strategy[9])  # deg
    result[6] = p
    result[7] = r
    return result

####










































parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", help="Number of nodes in the graph to generate. (default=10)")
parser.add_argument("-p", "--probability", help="Probability of an edge to exist in the graph to generate. (default=0.1)")
parser.add_argument("-i", "--iterations", help="Number of iterations of graphs to simulate. (default=100)")
args = parser.parse_args()

if args.nodes is None:
    args.nodes = 10

if args.probability is None:
    args.probability = 0.1

if args.iterations is None:
    args.iterations = 100





node_count = int(args.nodes)
p = float(args.probability)
iterations = int(args.iterations)
start = time()
for i in range(iterations):

    t = ctime()
    t = t.replace(" ", "_")
    t = t.replace(":", "_")



    folder = f"../data/({node_count}_{p})/"

    if not os.path.exists(folder):
        os.mkdir(folder)

    name = f"{t}_{i}"



    # print(f"Generating graph {folder}{name} ...", end="")
    M, M_t = generate(node_count,p)
    # print("\t Done!")
    # SAVE
    with open(f"{folder}graph_{name}.npy", 'wb') as graph_file:
        np.save(graph_file, M) # first object
        np.save(graph_file, M_t) # second object



















    # print("Adding source node ...", end="")
    source = M.shape[0] # additional node has index of one dimension of M
    M = np.concatenate((M,np.zeros((1,M.shape[1]), dtype=bool)), axis=0)
    M = np.concatenate((M,np.zeros((M.shape[0],1), dtype=bool)), axis=1)
    M_t = np.concatenate((M_t,np.zeros((1,M_t.shape[1]), dtype=bool)), axis=0)
    M_t = np.concatenate((M_t,np.zeros((M_t.shape[0],1), dtype=bool)), axis=1)
    # print("\t Done!")

    # print("Starting Reach ...")
    # start = time()
    R = np.matrix(M_t, dtype=bool)
    R_transposed = R.transpose().copy()
    np.fill_diagonal(R, True)

    visited_reach = []
    reachability_queue = deque()
    reachability_queue.append(np.array(R[source]).flatten())
    states = [np.array(R[source]).flatten()]
    states_set = {tuple(np.array(R[source]).flatten())}

    while True:
        if len(reachability_queue) == 0:
            break
        reachable = reachability_queue.popleft()
        # source = reachable.size-1 and then -1 for all other but source
        for index in range(source):
            if not reachable[index] and not np.any(R_transposed[index]): # optimised: in-degreee == 0 only
                add_reach = np.array(R[index] > reachable).flatten()
                visited_reach.append((reachable,index,add_reach)) # the index to reference must be the highest index in that vector due to DAG and topological ordering
                new_reachability = add_reach | reachable
                nr_tuple = tuple(new_reachability)
                if not nr_tuple in states_set:
                    states_set.add(nr_tuple)
                    states.append(new_reachability)
                    reachability_queue.append(new_reachability)


    decision_states = np.array(list(map(lambda x: x[0], visited_reach)), dtype=bool)
    decisions = np.array(list(map(lambda x: x[1], visited_reach)),dtype=int)
    reach = np.array(list(map(lambda x: x[2], visited_reach)), dtype=bool)

    print("Results:\t\t",np.array(states).shape[0])
    # print("Reach by Matrix:\t", time()-start)


    # SAVE
    # with open(f"{folder}reach_{name}.npy", 'wb') as log_file:
    #     np.save(log_file, states) # states
    #     np.save(log_file, decision_states) # current_state
    #     np.save(log_file, decisions)
    #     np.save(log_file, reach)

    # print("Done with Reach!\n")










    # print("Starting Hubs ...")

    states = np.asarray(states, dtype=bool)
    M = M.astype(np.float64)
    M_t = M_t.astype(np.float64)

    # start = time()
    hubs_array = None
    try:
        hubs_array = calculate_hub_scores(M_t, states)
    except RuntimeError as re:
        print(i, re)
        continue
    # print("Numba (sec):\t\t", time()-start)

    hubs = pd.DataFrame(hubs_array)
    # hubs.to_pickle(f"{folder}hubs_{name}.pkl") # SAVE

    # print("Done with Hubs!\n")
















    state_index_dict = transform_states_to_index(states)
    global_decision_indicies = transform_decision_states_to_index(decision_states)
    # print("Transformed Reach Data.")




    state_strategies = []
    for  state_index in range(len(states)):
        current_state = states[state_index]
        if np.all(current_state):  # when all reachable
            pass  # final state: all nodes are referenced
        else:
            state_strategies.append(eval_current_state_strategies(current_state))
        

    # print()
    # print("Analysing...")
    # print()

    # print("###########")
    # print("# Equality #")
    # print("###########")
    analysis_results = np.array(
        list(map(analyse_strategy_equality, state_strategies)), dtype=bool)
    df = pd.DataFrame(analysis_results, columns=['reach','iterH','initH','deg'])
    df.to_pickle(f"{folder}results_eq_{name}.pkl") # SAVE
    df = pd.DataFrame()
    # print_results(analysis_results)

    # print("\n\n")

    # print("###########")
    # print("# Subsets #")
    # print("###########")
    analysis_results = np.array(
        list(map(analyse_strategy_subset_ok, state_strategies)), dtype=bool)
    df = pd.DataFrame(analysis_results, columns=['reach','iterH','initH','deg'])
    df.to_pickle(f"{folder}results_sub_{name}.pkl") # SAVE
    df = pd.DataFrame()
    # print_results(analysis_results)

    # print("\n\n")

    # # print("###########")
    # # print("# Superset #")
    # # print("###########")
    # analysis_results = np.array(
    #     list(map(analyse_strategy_superset_ok, state_strategies)), dtype=bool)
    # df = pd.DataFrame(analysis_results, columns=['reach','iterH','initH','deg'])
    # df.to_pickle(f"{folder}results_sup_{name}.pkl") # SAVE
    # print_results(analysis_results)

    # # print()
    # # print("Analysis:\t", time()-start)

    # # print()






    # print("\n\n\n")
    # print("########################")
    # print("# Precision and Recall #")
    # print("########################")
    # print("\n")
    analysis_results = np.array(
        list(map(analyse_strategy_pr, state_strategies)), dtype=float)
    df = pd.DataFrame(analysis_results, columns=['reach_precision', 'reach_recall','iterH_precision', 'iterH_recall','initH_precision', 'initH_recall','deg_precision', 'deg_recall'])
    df.to_pickle(f"{folder}results_pr_{name}.pkl") # SAVE
    df = df.mean(axis=0)
    # print(df)
    df.to_pickle(f"{folder}results_mean_{name}.pkl")
    df = pd.DataFrame()
    
    if i % 10 == 0:
        print(i)
    # print("\n\n\n")
    # print("\n\n\n")
print("Done.")
print("Time:\t\t\t", time()-start)