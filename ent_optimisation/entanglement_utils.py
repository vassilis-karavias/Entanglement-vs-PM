import copy

import pandas as pd

def get_key_dict(data_file):

    # get a list of the edges that need more capacity and the value of this capacity
    key_dicts = {}
    possible_ids = data_file["ID"].unique()
    for id in possible_ids:
        key_dict = {}
        for index, row in data_file.iterrows():
            if row["ID"] == id:
                source = int(row["source"])
                target = int(row["target"])
                eps_node = int(row["eps_node"])
                cap = row["capacity"]
                if source < target:
                    key_dict[(target, source, eps_node)] = int(round(cap))
                elif source > target:
                    key_dict[(source, target, eps_node)] = int(round(cap))
        print("Complete Ent: " + str(id))
        key_dicts[id] = copy.deepcopy(key_dict)
    return key_dicts

def get_key_dict_pm(data_file):

    # get a list of the edges that need more capacity and the value of this capacity
    key_dicts = {}
    possible_ids = data_file["ID"].unique()
    for id in possible_ids:
        key_dict = {}
        for index, row in data_file.iterrows():
            if row["ID"] == id:
                source = int(row["source"])
                target = int(row["target"])
                cap = row["capacity"]
                if source < target:
                    key_dict[(target, source)] = int(round(cap))
                elif source > target:
                    key_dict[(source, target)] = int(round(cap))#
        print("Complete PnM: " + str(id))
        key_dicts[id] = copy.deepcopy(key_dict)
    return key_dicts

def get_required_connections(data_file):
    required_connections = {}
    possible_ids = data_file["ID"].unique()
    for id in possible_ids:
        required_conn = []
        for index, row in data_file.iterrows():
            if row["ID"] == id:
                source = int(row["source"])
                target = int(row["target"])
                if source < target:
                    if (target, source) not in required_conn:
                        required_conn.append((target, source))
                elif source > target:
                    if (source, target) not in required_conn:
                        required_conn.append((source, target))
        print("Complete Ent Req Conns: " + str(id))
        required_connections[id] = copy.deepcopy(required_conn)
    return required_connections




def import_data_ent(data_file_location):
    data_file = pd.read_csv(data_file_location)
    key_dict = get_key_dict(data_file)
    required_connections = get_required_connections(data_file)
    return key_dict, required_connections

def import_data_pm(data_file_location):
    data_file = pd.read_csv(data_file_location)
    key_dict = get_key_dict_pm(data_file)
    return key_dict