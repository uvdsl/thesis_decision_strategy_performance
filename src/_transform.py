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