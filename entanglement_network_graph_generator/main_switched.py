from capacity_storage import *
from random_graph_generation import SpecifiedTopologyGraph
import numpy as np
from enum import Enum, unique

@unique
class Topology(Enum):
    """

    """
    BUS = 0
    RING = 1
    STAR = 2
    MESH = 3
    HUBSPOKE = 4
### parameters to what kind of investigation this is

topology =  Topology(3).name
no_of_bobs = 10
no_of_bob_locations = 10
dbswitch = 0
box_size = 100
# for ring topology
radius = 50
# for star topology
central_node_is_detector = False
# for mesh and hub&spoke
no_of_conns_av = 3.5
mesh_composed_of_only_detectors = False
nodes_for_mesh = 3
mesh_in_centre = True



graphs = []
sizes = []
current_graphs = 0
for n in np.arange(start = 25, stop = 30, step = 5):
    for no_of_bobs in np.arange(start = 10, stop = 15, step =5):
        for i in range(40):
            graph_node = SpecifiedTopologyGraph()
            if topology == Topology(0).name:
                graph_node.generate_random_bus_graph(n, no_of_bobs, no_of_bob_locations, dbswitch, box_size)
            elif topology == Topology(1).name:
                graph_node.generate_random_ring_graph(n, no_of_bobs, no_of_bob_locations, radius, dbswitch)
            elif topology == Topology(2).name:
                graph_node.generate_random_star_graph(n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, central_node_is_detector)
            elif topology == Topology(3).name:
                try:
                    graph_node.generate_random_mesh_graph(n + no_of_bobs, no_of_bobs, no_of_bobs, dbswitch, box_size, no_of_conns_av, current_graphs)
                    current_graphs += 1
                    print("Current Graph: " + str(current_graphs))
                except ValueError:
                    print("Current Graph: " + str(current_graphs))
                    continue
            elif topology == Topology(4).name:
                if mesh_in_centre:
                    try:
                        graph_node.generate_hub_spoke_with_hub_in_centre(n, no_of_bobs, no_of_bobs, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av)
                    except ValueError:
                        continue
                else:
                    try:
                        graph_node.generate_random_hub_spoke_graph(n, no_of_bobs, no_of_bobs, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av)
                    except ValueError:
                        continue
            print("Finished Graph " + str(no_of_bobs) + "," + str(i))
            graph = graph_node.graph
            graphs.append(graph)
            sizes.append(20)




store_topology_for_hot_cold_bobs(graphs,  node_data_store_location="ent_mesh_graphs_positions",
                                   edge_data_store_location="ent_mesh_graphs_edges", size=sizes)
