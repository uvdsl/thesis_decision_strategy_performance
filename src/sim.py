import os
import argparse
import numpy as np
import pandas as pd
from numba import njit, prange
from _hits import hits_hubs
from time import time, ctime
import networkx as nx

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



@njit(nogil=True, parallel=True)
def get_zero_degree_nodes(A):
    '''
    Get nodes with out_degree of zero from matrix A.
    Providing A.T results in nodes with in_degree of zero.
    '''
    is_source_node = np.zeros_like(A[0])
    for rindex in prange(A.shape[0]):
        in_deg = np.sum(A[rindex])
        if in_deg == 0:
            is_source_node[rindex] = True
    return is_source_node


@njit(nogil=True)
def hipow(n):
    ''' Return the highest power of 2 within n. '''
    exp = 0
    while 2**exp <= n:
        exp += 1
    return exp


@njit(nogil=True)
def to_bool_array(v, result):
    '''
    Given a value v, provide an numpy array (bool) of the binary represenation of v with given length.
    '''
    length = result.size
    if v > 2**length:
        raise RuntimeError('To Bool Array: Insufficient length provided.')
    for index in range(1, hipow(v)+1):
        result[-index] = v % 2
        v = v >> 1
    return result

@njit(nogil=True, parallel=True)
def ars(A, state, reachable_source_nodes_indicies):
    '''
    Additonal Reachability Strategy
    '''
    additional_reachabilities = np.zeros_like(reachable_source_nodes_indicies)
    for i in prange(reachable_source_nodes_indicies.size):
        additional_reachabilities[i] = np.sum(state < A[reachable_source_nodes_indicies[i]])
    max_value = np.max(additional_reachabilities)
    return np.where(additional_reachabilities == max_value)[0], max_value


@njit(nogil=True, parallel=True)
def irs(A, reachable_source_nodes_indicies):
    '''
    Initial Reachability Strategy
    '''
    additional_reachabilities = np.zeros_like(reachable_source_nodes_indicies)
    for i in prange(reachable_source_nodes_indicies.size):
        additional_reachabilities[i] = np.sum(
            A[reachable_source_nodes_indicies[i]])
    max_value = np.max(additional_reachabilities)
    return np.where(additional_reachabilities == max_value)[0], max_value


@njit(nogil=True, parallel=True)
def ith(A, state, reachable_source_nodes_indicies):
    '''
    Iterative Hub Score Strategy
    '''
    M = A.copy()  # .copy() # not sure if needed, but I feel more comfortable if passed by copy.
    M[-1] = state
    np.fill_diagonal(M, False)  # account for reachability
    hubs = hits_hubs(M.astype(np.float64))[reachable_source_nodes_indicies]
    max_value = np.max(hubs)
    return np.where(hubs == max_value)[0], max_value


@njit(nogil=True, parallel=True)
def inh(A, reachable_source_nodes_indicies):
    '''
    Initial Hub Score Strategy
    '''
    M = A.copy() # .copy() # not sure if needed, but I feel more comfortable if passed by copy.
    np.fill_diagonal(M, False)  # account for reachability
    hubs = hits_hubs(M.astype(np.float64))[reachable_source_nodes_indicies]
    max_value = np.max(hubs)
    return np.where(hubs == max_value)[0], max_value


@njit(nogil=True, parallel=True)
def opt(A, state, reachable_source_nodes_indicies):
    '''
    Optimal Strategy
    '''
    scores = np.zeros_like(reachable_source_nodes_indicies).astype(np.float64)
    for i in prange(reachable_source_nodes_indicies.size):
        next_state = state | A[reachable_source_nodes_indicies[i]]
        # .copy() # not sure if needed, but I feel more comfortable if passed by copy.
        M = A.copy()
        M[-1] = next_state
        np.fill_diagonal(M, False)  # account for reachability
        scores[i] = hits_hubs(M.astype(np.float64))[-1]
    max_value = np.max(scores)
    return np.where(scores == max_value)[0]


@njit(nogil=True)
def calculate_pr(decisions, ground_truth):
    tp = 0
    fp = 0
    fn = 0
    for titem in ground_truth:
        if titem in decisions:
            tp += 1
        else:
            fn += 1
    fp = decisions.size - tp  # rest of decisions is fp
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return precision, recall


@njit(nogil=True)
def eval_strat(options_strat, options_opt, result):
    result[0],  result[1] = calculate_pr(options_strat, options_opt)
    result[2] = (result[0] == 1)
    result[3] = (result[0] == 1) and (result[1] == 1)
    return result


@njit(nogil=True)
def analyse(A, abstr_state_dummy, reach_state_dummy, source_node_name_array, result):
    '''
    1) calculate reachability state
    2) calculate decisions
    3) evaluate strategies
    4) return metrics
    '''
    states_count = 2**source_node_name_array.size
    for i in prange(states_count-1):
        # reset
        abstr_state_dummy = np.zeros_like(abstr_state_dummy)
        reach_state = np.zeros_like(reach_state_dummy)
        # get state
        abstract_state = to_bool_array(i, abstr_state_dummy)
        reached_source_nodes_indicies = source_node_name_array[abstract_state]
        # not yet reached source nodes
        nyrsn = source_node_name_array[~abstract_state]
        for reachable_index in reached_source_nodes_indicies:
            reach_state = reach_state | A[reachable_index]
        # decisions in state
        # given state => find option set of nyrsn
        option_set_from_nyrsn_ars, v_ars = ars(A, reach_state, nyrsn)
        option_set_from_nyrsn_irs, v_irs = irs(A, nyrsn)
        option_set_from_nyrsn_ith, v_ith = ith(A, reach_state, nyrsn)
        option_set_from_nyrsn_inh, v_inh = inh(A, nyrsn)
        option_set_from_nyrsn_opt = opt(A, reach_state, nyrsn)
        # print(option_set_from_nyrsn_ars, option_set_from_nyrsn_irs, option_set_from_nyrsn_ith, option_set_from_nyrsn_inh, option_set_from_nyrsn_opt)
        # print(v_ars,v_irs,v_ith,v_inh)
        # eval decision
        result[i, 0:4] = eval_strat(
            option_set_from_nyrsn_ars, option_set_from_nyrsn_opt, result[i, 0:4])
        result[i, 4:8] = eval_strat(
            option_set_from_nyrsn_irs, option_set_from_nyrsn_opt, result[i, 4:8])
        result[i, 8:12] = eval_strat(
            option_set_from_nyrsn_ith, option_set_from_nyrsn_opt, result[i, 8:12])
        result[i, 12:16] = eval_strat(
            option_set_from_nyrsn_inh, option_set_from_nyrsn_opt, result[i, 12:16])
    return result























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
for i in prange(iterations):

    t = ctime()
    t = t.replace(" ", "_")
    t = t.replace(":", "_")



    folder = f"../data/({node_count}_{p})/"

    if not os.path.exists(folder):
        os.mkdir(folder)

    name = f"{t}_{i}"

    print(f"Generating graph {folder}{name} ...", end="")
    M, M_t = generate(node_count,p)
    # print("\t Done!")
    # SAVE
    with open(f"{folder}graph_{name}.npy", 'wb') as graph_file:
        np.save(graph_file, M) # first object
        np.save(graph_file, M_t) # second object
    M = None # memory cleanup
    print("\t Done!")


    source_nodes_index_array = get_zero_degree_nodes(M_t.T)
    source_nodes_names = np.array(np.where(source_nodes_index_array)).flatten()


    length = source_nodes_names.size  # decision state length
    # There are 2**(number of source nodes) decision states
    number_of_states = 2**length
    print("Decision States:\t\t", number_of_states-1)
    # abstract_decision_states = np.zeros((2**length,length), dtype=bool)
    # # We can derive the decision state simply by the binary representation of the corresponding index of the decision state, e.g. 4 [padded, true, false, false].
    # # e.g. np.binary_repr(4) -> 100 or to_bool_array



    # print("Adding new node ...", end="")
    new_node = M_t.shape[0]  # additional node has index of one dimension of M
    M_t = np.concatenate((M_t, np.zeros((1, M_t.shape[1]), dtype=bool)), axis=0)
    M_t = np.concatenate((M_t, np.zeros((M_t.shape[0], 1), dtype=bool)), axis=1)
    # print("\t Done!")


    
    R = np.matrix(M_t, dtype=bool)
    np.fill_diagonal(R, True)

    dummy_abstr_state = np.zeros(length, dtype=bool)
    dummy_reach_state = np.zeros(R.shape[1], dtype=bool)
    # 16 due to number of strategies * number of KPIs
    results_table = np.zeros((number_of_states-1, 16), dtype=np.float64)

    try:

        results_table = analyse(R, dummy_abstr_state,
                            dummy_reach_state, source_nodes_names, results_table)

    except RuntimeError as re:
        print(i, re)
        continue


    
    


    results_table = pd.DataFrame(results_table, columns=['asr_precision', 'asr_recall', 'asr_subset', 'asr_equality', 'isr_precision', 'isr_recall',
                                                        'isr_subset',  'isr_equality', 'ith_precision', 'ith_recall',  'ith_subset',  'ith_equality', 'inh_precision', 'inh_recall', 'inh_subset', 'inh_equality'])
    results_table = results_table.mean()
    results_table['source_nodes_count'] = length
    results_table.to_pickle(f"{folder}results_{name}.pkl") 

    # print(results_table)
    if i % 10 == 0:
        print(i)


print("Time:\t\t\t", time()-start)
# Load on memory:

# calc hub scores for all states
# calc reaches for all states for NYRSN

# eval optimal
# eval ARS
# eval ISR
# eval Iter Hub
# eval Init Hub


# Load on CPU:


# eval ARS
# for each state
# for each NYRSN
# calc AR
# choose best from NYRSN

# eval ISR
# for each NYRSN
# calc AR in initial state
# for each state
# choose best from NYRSN

# eval Iter Hub
# for each state
# calc hub scores
# choose best from NYRSN

# eval Init Hub
# calc hub scores in initial state
# for each state
# choose best from NYRSN

# eval optimal
# for each state
# for each NYRSN
# put edge
# calc hub scores
# check new_nodes score
# discard hub scores matrix
# choose best from NYRSN
