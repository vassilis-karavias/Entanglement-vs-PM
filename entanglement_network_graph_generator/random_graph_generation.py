from generate_graph import *

class SpecifiedTopologyGraph:

    def __init__(self, graph = None):
        if graph != None:
            self.graph = graph


    def bus_topology_graph(self, xcoords, ycoords, nodetypes, label, dbswitch):
        self.graph = BusNetwork(xcoords, ycoords, nodetypes, label, dbswitch)

    def ring_topology_graph(self, radius, no_nodes, node_types, label, dbswitch):
        self.graph = RingNetwork(radius, no_nodes, node_types, label, dbswitch)

    def star_topology_graph(self, xcoords, ycoords, node_types, central_node, label, dbswitch):
        self.graph = StarNetwork(xcoords, ycoords, node_types, central_node, label, dbswitch)

    def mesh_topology_graph(self, xcoords, ycoords, node_types, no_of_conns_av, box_size, label, dbswitch):
        self.graph = MeshNetwork(xcoords, ycoords, node_types, no_of_conns_av, box_size, label, dbswitch)

    def hub_spoke_graph(self, xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre):
        self.graph = HubSpokeNetwork(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre)

    def lattice_graph(self, shape, nodetypes, box_size,  label, dbswitch):
        self.graph = LatticeNetwork(shape, nodetypes, box_size,  label, dbswitch)

    def make_standard_labels(self):
        node_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                      "11", "12", "13", "14", "15"]
        for i in range(16, 500):
            node_names.append(str(i))
        return node_names


    def generate_random_coordinates(self, n, box_size):
        xcoords = np.random.uniform(low=0.0, high=box_size, size=(n))
        ycoords = np.random.uniform(low=0.0, high=box_size, size=(n))
        self.xcoords, self.ycoords = xcoords, ycoords
        return xcoords, ycoords

    def generate_random_detector_perturbation(self, n, no_of_bobs, no_of_bob_locations):
        # get an array of the appropriate number of nodes, bob nodes, and bob locations
        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                 np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
        array_for_perturbation = np.concatenate(
            (array_for_perturbation, np.full(shape=n - no_of_bob_locations, fill_value=0)))
        # randomly permute this set of nodes to generate random graph
        node_types = np.random.permutation(array_for_perturbation)
        self.node_types = node_types
        return node_types

    def generate_random_bus_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size):
        label = self.make_standard_labels()
        xcoords, ycoords = self.generate_random_coordinates(n, box_size)
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        self.bus_topology_graph(xcoords, ycoords, node_types, label, dbswitch)

    def generate_random_ring_graph(self, n, no_of_bobs, no_of_bob_locations, radius, dbswitch):
        label = self.make_standard_labels()
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        self.ring_topology_graph(radius, n, node_types, label, dbswitch)

    def generate_random_star_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, central_node_is_detector):
        label = self.make_standard_labels()
        xcoords, ycoords = self.generate_random_coordinates(n, box_size)
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        central_node = 0
        if central_node_is_detector:
            for i in range(len(node_types)):
                if node_types[i] == 1 or node_types[i] == 2:
                    central_node = i
                    break
        else:
            for i in range(len(node_types)):
                if node_types[i] == 0:
                    central_node = i
                    break
        self.star_topology_graph(xcoords, ycoords, node_types, central_node, label, dbswitch)

    def generate_random_mesh_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, no_of_conns_av, current_graphs = 0):
        i = 0
        while True:
            try:
                label = self.make_standard_labels()
                xcoords, ycoords = self.generate_random_coordinates(n, box_size)
                node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
                self.mesh_topology_graph(xcoords, ycoords, node_types, no_of_conns_av, box_size, label, dbswitch)
            except:
                print("Current Graph: " + str(current_graphs))
                if i < 2000:
                    pass
                else:
                    i += 1
                    raise ValueError
            else:
                break

    def generate_random_hub_spoke_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av):
        i = 0
        while True:
            try:
                label = self.make_standard_labels()
                xcoords, ycoords = self.generate_random_coordinates(n, box_size)
                if mesh_composed_of_only_detectors:
                    if no_of_bob_locations < nodes_for_mesh:
                        print("For all mesh nodes to be detectors the number of detectors must be bigger than the number of nodes in the mesh grid.")
                        raise ValueError
                    else:
                        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                                 np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
                        array_for_perturbation = np.concatenate(
                            (array_for_perturbation, np.full(shape=n - no_of_bob_locations, fill_value=0)))
                        node_types = array_for_perturbation
                else:
                    node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
                self.hub_spoke_graph(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre = False)
            except ValueError:
                if i < 50:
                    pass
                else:
                    i += 1
                    raise ValueError
            else:
                break

    def generate_hub_spoke_with_hub_in_centre(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av):
        i = 0
        while True:
            try:
                label = self.make_standard_labels()
                xcoords_mesh, ycoords_mesh = self.generate_random_coordinates(nodes_for_mesh, box_size/3)
                xcoords_mesh = xcoords_mesh + box_size/3
                ycoords_mesh = ycoords_mesh + box_size/3 # add term to centre mesh parts
                xcoords_rest, ycoords_rest = self.generate_random_coordinates(n - nodes_for_mesh, box_size)
                xcoords = np.concatenate((xcoords_mesh, xcoords_rest))
                ycoords = np.concatenate((ycoords_mesh, ycoords_rest))
                self.xcoords, self.ycoords = xcoords, ycoords
                if mesh_composed_of_only_detectors:
                    if no_of_bob_locations < nodes_for_mesh:
                        print(
                            "For all mesh nodes to be detectors the number of detectors must be bigger than the number of nodes in the mesh grid.")
                        raise ValueError
                    else:
                        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                                 np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
                        array_for_perturbation = np.concatenate(
                            (array_for_perturbation, np.full(shape=n - no_of_bob_locations, fill_value=0)))
                        node_types = array_for_perturbation
                else:
                    node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
                self.hub_spoke_graph(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size, label, dbswitch, mesh_in_centre = True)
            except ValueError:
                if i < 50:
                    pass
                else:
                    i += 1
                    raise ValueError
            else:
                break


    def generate_random_lattice_graph(self, shape, no_of_bobs, no_of_bob_locations, box_size, dbswitch):
        label = self.make_standard_labels()
        n = math.prod(shape)
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        self.lattice_graph(shape, node_types, box_size, label, dbswitch)
        self.xcoords = (np.arange(self.graph.num_vertices()) % shape[0]) * box_size / (shape[0] - 1)
        self.ycoords = (np.arange(self.graph.num_vertices()) // shape[0]) * box_size / (shape[1] - 1)
