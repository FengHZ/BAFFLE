from itertools import permutations, combinations
import torch


def create_client_weight(client_num):
    global_federated_matrix = [1 / client_num] * client_num
    return global_federated_matrix


def federated_initialize(model_list, global_model):
    for model in model_list:
        model.load_state_dict(global_model.state_dict())


def avg_util(model_list, coefficient_matrix):
    """
    :param model_list: a list of all models needed in federated average.
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :return: model list after federated average
    """
    dict_list = [it.state_dict() for it in model_list]
    dict_item_list = [dic.items() for dic in dict_list]
    for key_data_pair_list in zip(*dict_item_list):
        source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                            enumerate(key_data_pair_list)]
        dict_list[0][key_data_pair_list[0][0]] = sum(source_data_list)
    for model in model_list:
        model.load_state_dict(dict_list[0])
