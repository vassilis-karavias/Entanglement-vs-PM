import os
import csv


def store_topology_for_hot_cold_bobs(graphs, node_data_store_location, edge_data_store_location, size = 100):
    """

    :param graphs: graphs to use
    :param size: Array of the size of the graphs
    :return:
    """
    for i in range(len(graphs)):
        store_position_graph(graphs[i], node_data_store_location, edge_data_store_location, graph_id = id)
        print("Finished graph " + str(i))
        id += 1



def store_position_graph(network, node_data_store_location, edge_data_store_location, graph_id = 0):
    edges = network.g.get_edges(eprops=[network.lengths_of_connections])
    dictionaries = []
    dictionary_fieldnames = ["ID", "source", "target", "distance"]
    for edge in range(len(edges)):
        source = edges[edge][0] + 1
        target = edges[edge][1] + 1
        distance = edges[edge][2]
        dictionaries.append(
            {"ID": graph_id, "source": source , "target": target, "distance": distance})
    nodes = network.g.get_vertices(vprops =[network.x_coord, network.y_coord])
    dictionary_fieldnames_nodes = ["ID", "node", "xcoord", "ycoord", "type"]
    dict_nodes = []
    for node in range(len(nodes)):
        node_label = nodes[node][0]
        xcoord = nodes[node][1]
        ycoord = nodes[node][2]
        type = network.vertex_type[node_label]
        dict_nodes.append({"ID": graph_id, "node": node_label+1, "xcoord": xcoord, "ycoord": ycoord, "type": type})

    if os.path.isfile(node_data_store_location + '.csv'):
        with open(node_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
            writer.writerows(dict_nodes)
    else:
        with open(node_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
            writer.writeheader()
            writer.writerows(dict_nodes)

    if os.path.isfile(edge_data_store_location + '.csv'):
        with open(edge_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writerows(dictionaries)
    else:
        with open(edge_data_store_location + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(dictionaries)