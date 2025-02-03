# Entanglement-vs-PM
Optimise the P&M and Entanglement QNs.

# Requirements
numpy: 1.20.1+  
graph-tool: 2.37+  
pandas: 1.5.2+  
scipy: 1.7.3+  
cplex: V20.1  
matplotlib: 3.6.2+  
networkx: 2.8.8+  
cplex V20.1  

# How to Use
## Generate Random Geometric Graphs
To generate random geometric graphs use the  
*entanglement_network_graph_generator.main_switched.py*  
file. Here you can specify the topology from one of  BUS, RING, STAR, MESH, HUBSPOKE, the size of the square containing the network in km (*box_size* parameter), the *radius* of the ring for a ring topology, and the number of average connections per node (*no_of_conns_av* parameter) for the mesh/hubspoke networks. Note that while the db_loss of the switch is a variable this does not set the db loss here. You can generate a list of these graphs by specifying n and no_of_bobs. n is the number of user nodes and no_of_bobs is the number of source nodes for entanglement. For each graph you want to generate, first set up a  
*stg = random_graph_generation.SpecifiedTopologyGraph()*  
class and use the appropriate *stg.generate_random_graph()* method as seen in *main_switched.py*. You can then store the graphs in a list of *graphs*. To create the .csv files of the graphs use  
*capacity_storage.store_topology_for_hot_cold_bobs(graphs,  node_data_store_location, edge_data_store_location, size)*  
where *graphs* is the array of graphs generated. *node_data_store_location* is the csv file path for storing the node data (minus the .csv). The node data will be stored in a csv file in the format of [ID, node, xcoord, ycoord, type] where ID is the graph ID of the current row, node is the node name (1,2 etc), xcoord is the xcoordinate value in km and ycoord is the y coordinate in km, finally type is the type of node, S is a user node and B is a bob node (source node). *edge_data_store_location* is the csv file path for storing the edge data (minus the .csv). The edge data will be stores in a csv file in the format of [ID, source, target, distance] where ID is the graph ID of the current row, source is the source node of the edge, target is the target node of the edge and distance is the overall distance of the edge in km.
## Generate Capacity Input Files
To take the random geometric graphs and transform them into capacity files use the 
*ent_preprocessing.graph_generator.py*  
file. First import the graphs using  
*graphs = graph_generator.import_graphs(position_node_file, edge_file)*  
where *position_node_file* is the .csv file of the node data and *edge_file* is the edge data generated in *store_topology_for_hot_cold_bobs*. For each graph, you can evaluate and store the entanglement key rate using:  
*cpe = Capacity_Graph_Entanglement(graph = graphs[i], loss_per_km, efficiency_detector, detector_errors, dark_count_alice, dark_count_bob, f_e, repetition_rate, switch_loss)*  
*capacity_edge_df = cpe.get_capacity_df()*  
*save_csv_ent(csv_file_name, capacity_df = capacity_edge_df, graph_id)*
The *graph* is the current graph, *loss_per_km* is the dB loss per km of the fibre link, *efficiency_detector* is the fractional detector efficiency (0.8 is 80% efficiency), *detector_errors* is the fractional error of detectors (0.01 is 1% error rate), *dark_count_alice* is the dark count probability at Alice's detector, *dark_count_bob* is the dark count probability at Bob's detector, *f_e* is the efficiency of the error correction method relative to the Shannon bound, *repetition_rate* is the repetition rate of the source in Hz and *switch_loss* is the dB loss of the switch. *csv_file_name* is the path name of the csv file for the output file of the key rate between users (minus the .csv) and *graph_id* is the current graph's ID value. The csv will be in the format [ID, source, target, detector, capacity, size] where ID is the graph ID of the current row, source is the node name of the source user node, target is the node name of the target user node, detector is the node name of the node on which the entanglement source is on, capacity is the bits/s secret key rate capacity of the link and size is the km size of the network box.  
For each graph, you can evaluate and store the PM key rate using:  
*pnm = Capacity_Graph_BB84(graph = graphs[i], switch_loss)*  
*capacity_edge_df = pnm.get_capacity_graph(key_rates_dict)*  
*save_csv_pnm(csv_file_name, capacity_df=capacity_edge_df, graph_id)*  
The *graph* is the current graph, *switch_losss* is the dB loss of the switch. The key rate vs length of link is given in the csv file stored in location *key_rates_dict*. We provide the key rates of the PM protocol for SPADS in *'rates_entanglement_spad.csv'* and for SNSPDS in *'rates_entanglement_snspd.csv'*. *csv_file_name* is the path name of the csv file for the output file of the key rate between users (minus the .csv) and *graph_id* is the current graph's ID value. The csv will be in the format [ID, source, target, capacity] where ID is the graph ID of the current row, source is the node name of the source user node, target is the node name of the target user node and capacity is the bits/s secret key rate capacity of the link. 
## Optimisation
To perform the optimisation use the  
*ent_optimisation.optimisation_investigations.py*  
file. Several different methods are provided for different investigations. In general, to import the graph data, use:  
*key_dict_hot, required_connections_hot = entanglement_utils.import_data_ent(data_file_location=ent_data_file_hot)*  
*key_dict_pm_hot = entanglement_utils.import_data_pm(data_file_location=pm_data_file_hot)*  
*key_dict_cold, required_connections_cold = entanglement_utils.import_data_ent(data_file_location=ent_data_file_cold)*  
*key_dict_pm_cold = entanglement_utils.import_data_pm(data_file_location=pm_data_file_cold)*  
where *data_file_loc* is the location of the .csv data file generated in the previous step. From there to optimise the entanglement network:  
*prob = cplex.Cplex()*  
*optimisation = entanglement_optimisation.Entanglement_Optimisation_Multiple_Dev_Types(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])*  
*sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=1e5, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost)*  
The first step is to create a Cplex class and then set up the problem. *required_connections,key_dict* is the same as the imported values, *id* is the current ID of the graph investigated. *cij* is the minimum transmission requirements between users in the network, this can be a float: all user pairs are assumed to have the same transmission requirements, or a dict of {(i,j): cij} where cij is the transmission requirement between users i,j. *time_limit* is the maximum time to solve the model, *Lambda* is the number of channels each entanglement source can support, *c_esource* is the cost of the entanglement source, *c_edet* is the cost of the SPAD detector, *c_edet_cold* is the cost of the cold SNSPD detector. *sol_dict* is the dictionary of {variable_name: optimal_solution_value} and *prob* is the Cplex class. To get the optimal objective value do: *objective_value = prob.solution.get_objective_value()*.  
To optimise the PM network:  
*cost = get_cost_for_prep_measure_mult_devs(key_dict_hot = key_dict_pm_hot[id], key_dict_cold = key_dict_pm_cold[id], cost_devs_hot = detector_cost + source_cost, cost_devs_cold = detector_cold_cost * detector_cost + source_cost, c_ij = cij)*  
*key_dict* is the same as the imported values, *id* is the current ID of the graph investigated. *cost_devs_hot* is the cost of the devices (source + detector pair) for a SPAD detector and *cost_devs_cold* is the cost of the devices (source + detector pair) for a SNSPD detector. *cij* is the minimum transmission requirements between users in the network, this can be a float: all user pairs are assumed to have the same transmission requirements, or a dict of {(i,j): cij} where cij is the transmission requirement between users i,j.


