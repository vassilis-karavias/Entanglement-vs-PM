import os.path

import pandas as pd
import networkx as nx
import numpy as np
import csv
from key_rate import KeyRateCalculator

def import_graphs(position_node_file, edge_file):
    node_data = pd.read_csv(position_node_file + ".csv")
    edge_data = pd.read_csv(edge_file + ".csv")
    possible_ids = node_data["ID"].unique()
    graphs =  {}
    for id in possible_ids:
        node_data_current = node_data[node_data["ID"] == id].drop(["ID"], axis = 1)
        edge_data_current = edge_data[edge_data["ID"] == id].drop(["ID"], axis=1)
        graph =  nx.from_pandas_edgelist(edge_data_current, "source", "target", ["distance"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_data_current.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph
        print("Finished ID: " + str(id))
    return graphs


class Capacity_Graph_BB84():

    def __init__(self, graph, switch_loss = 0):
        self.g = graph
        self.switch_loss = switch_loss


    def get_user_node_list(self):
        user_node_list = []
        for node_1 in self.g.nodes():
            if self.g.nodes()[node_1]["type"] == "S":
                for node_2 in self.g.nodes():
                    if self.g.nodes()[node_2]["type"] == "S":
                        if node_1 != node_2 and (node_2, node_1) not in user_node_list:
                            user_node_list.append((node_1, node_2))
        return user_node_list

    def get_capacity(self, node_1, node_2, key_rates_dict):
        distance = nx.shortest_path_length(self.g, node_1, node_2, weight = "distance") + self.switch_loss * 5
        distance = round(distance,2)
        if distance not in key_rates_dict.keys():
            return 0
        else:
            return key_rates_dict[distance]

    def get_capacity_edge_list(self, user_node_list, key_rates_dict):
        capacity_edge_list = []
        for node_1, node_2 in user_node_list:
            capacity = self.get_capacity(node_1, node_2, key_rates_dict)
            capacity_edge_list.append([node_1, node_2, capacity])
        return capacity_edge_list

    def get_capacity_graph(self, key_rates_dict):
        user_node_list = self.get_user_node_list()
        capacity_edge_list = self.get_capacity_edge_list(user_node_list, key_rates_dict)
        capacity_edge_df = pd.DataFrame(np.array(capacity_edge_list), columns = ["source", "target", "capacity"])
        # graph =  nx.from_pandas_edgelist(capacity_edge_df, "source", "target", ["capacity"])
        # graph = graph.to_undirected()
        # graph = graph.to_directed()
        return capacity_edge_df


class Capacity_Graph_Entanglement():

    def __init__(self, graph, loss_per_km, efficiency_detector, detector_errors, dark_count_alice, dark_count_bob, f_e, repetition_rate, switch_loss = 0):
        self.g = graph
        self.loss_per_km = loss_per_km
        self.eff_det = efficiency_detector
        self.detector_errors = detector_errors
        self.dark_count_alice = dark_count_alice
        self.dark_count_bob = dark_count_bob
        self.f_e = f_e
        self.rep = repetition_rate
        self.switch_loss = switch_loss


    def get_user_node_list(self):
        user_node_list = []
        for node_1 in self.g.nodes():
            if self.g.nodes()[node_1]["type"] == "S":
                for node_2 in self.g.nodes():
                    if self.g.nodes()[node_2]["type"] == "S":
                        if node_1 != node_2 and (node_2, node_1) not in user_node_list:
                            user_node_list.append((node_1, node_2))
        return user_node_list

    def get_paths_list(self, user_node_list):
        path_list = []
        for node in self.g.nodes():
            if self.g.nodes()[node]["type"] == "B":
                for node_1, node_2 in user_node_list:
                    path_list.append((node_1, node_2, node))
        return path_list

    def get_efficiency_channel(self, distance):
        eta_channel = np.power(10, - (self.loss_per_km * distance + self.switch_loss)/10)
        return eta_channel * self.eff_det

    def get_capacity(self, node_1, node_2, node_source):
        distance_1 =nx.shortest_path_length(self.g, node_1, node_source, weight = "distance")
        distance_1 = round(distance_1, 2)
        distance_2 = nx.shortest_path_length(self.g, node_2, node_source, weight = "distance")
        distance_2 = round(distance_2, 2)
        efficiency_alice = self.get_efficiency_channel(distance_1)
        efficiency_bob = self.get_efficiency_channel(distance_2)
        krc = KeyRateCalculator(detector_error=self.detector_errors, efficiency_alice=efficiency_alice, efficiency_bob=efficiency_bob,
                                dark_count_alice=self.dark_count_alice, dark_count_bob=self.dark_count_bob, f_e=self.f_e)
        return krc.get_current_rate() * self.rep


    def get_capacity_edge_list(self, path_list):
        capacity_edge_list = []
        for node_1, node_2, node_source in path_list:
            capacity = self.get_capacity(node_1, node_2, node_source)
            capacity_edge_list.append([node_1, node_2, node_source, capacity])
        return capacity_edge_list

    def get_capacity_df(self):
        user_node_list = self.get_user_node_list()
        path_list = self.get_paths_list(user_node_list)
        capacity_edge_list = self.get_capacity_edge_list(path_list)
        capacity_edge_df = pd.DataFrame(np.array(capacity_edge_list), columns=["source", "target", "eps_node", "capacity"])
        return capacity_edge_df

def save_csv_ent(csv_file_name, capacity_df, graph_id):
    store_dicts = []
    for index, row in capacity_df.iterrows():
        store_dicts.append({"ID": graph_id, "source": row["source"], "target": row["target"], "eps_node": row["eps_node"], "capacity": row["capacity"]})

    dictionary_fieldnames = ["ID", "source","target", "eps_node", "capacity"]
    if os.path.isfile(csv_file_name + ".csv"):
        with open(csv_file_name + ".csv", "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = dictionary_fieldnames)
            writer.writerows(store_dicts)
    else:
        with open(csv_file_name + ".csv", "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(store_dicts)

def save_csv_pnm(csv_file_name, capacity_df, graph_id):
    store_dicts = []
    for index, row in capacity_df.iterrows():
        store_dicts.append({"ID": graph_id, "source": row["source"], "target": row["target"], "capacity": row["capacity"]})

    dictionary_fieldnames = ["ID", "source","target", "capacity"]
    if os.path.isfile(csv_file_name + ".csv"):
        with open(csv_file_name + ".csv", "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = dictionary_fieldnames)
            writer.writerows(store_dicts)
    else:
        with open(csv_file_name + ".csv", "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(store_dicts)





if __name__ == "__main__":
    graphs = import_graphs(position_node_file="entanglement_mesh_graphs_positions_mesh_15", edge_file = "entanglement_mesh_graphs_edges_mesh_15")
    for i in graphs.keys():
        cpe = Capacity_Graph_Entanglement(graph = graphs[i], loss_per_km = 0.2, efficiency_detector = 0.15, detector_errors = 0.01, dark_count_alice = 1 * 10 ** -5, dark_count_bob = 1 * 10 ** -5, f_e = 1.2, repetition_rate = 100 * 10 ** 6, switch_loss=1)
        capacity_edge_df = cpe.get_capacity_df()
        save_csv_ent(csv_file_name="mesh_topology_entanglement_hot_rates_mesh_15", capacity_df=capacity_edge_df, graph_id = i)
        cpe = Capacity_Graph_Entanglement(graph = graphs[i], loss_per_km = 0.2, efficiency_detector = 0.8, detector_errors = 0.01, dark_count_alice = 3.5 * 10 ** -7, dark_count_bob = 3.5 * 10 ** -7, f_e = 1.2, repetition_rate = 100 * 10 ** 6, switch_loss=1)
        capacity_edge_df = cpe.get_capacity_df()
        save_csv_ent(csv_file_name="mesh_topology_entanglement_cold_rates_mesh_15", capacity_df=capacity_edge_df, graph_id = i)
        print("Finished Graph " + str(i))
    # capacity_edge_df.to_csv(path_or_buf = "trial.csv", index = False)

    dictionary = {}
    with open('rates_entanglement_spad.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary[round(float(row["L"]), 2)] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')


    dictionary_cold = {}
    with open('rates_entanglement_snspd.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_cold[round(float(row["L"]), 2)] = float(row['rate'])
            line_count += 1
        print(f'Processed {line_count} lines.')

    for i in graphs.keys():
        pnm = Capacity_Graph_BB84(graph = graphs[i], switch_loss=1)
        capacity_edge_df = pnm.get_capacity_graph(key_rates_dict=dictionary)
        save_csv_pnm(csv_file_name="mesh_topology_pm_hot_rates_mesh_15", capacity_df=capacity_edge_df, graph_id = i)
        capacity_edge_df = pnm.get_capacity_graph(key_rates_dict=dictionary_cold)
        save_csv_pnm(csv_file_name="mesh_topology_pm_cold_rates_mesh_15", capacity_df=capacity_edge_df, graph_id=i)
        print("Finished Graph " + str(i))
        # capacity_edge_df.to_csv(path_or_buf = "trial_pm.csv", index = False)