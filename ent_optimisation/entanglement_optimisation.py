from copy import deepcopy
import os
import pandas as pd
import cplex
import time
from entanglement_utils import import_data_ent

def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names=prob.variables.get_names()
    values=prob.solution.get_values()
    sol_dict={names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict


class Entanglement_Optimisation():

    def __init__(self, prob, required_connections, key_dict):
        ## required_connections: [(source, target) : source > target]
        ## key_dict: {(i,j,s): c_ijs} : i > j
        self.prob = prob
        self.required_connections = required_connections
        self.key_dict = key_dict
        self.sources = []
        for k in self.key_dict.keys():
            if k[2] not in self.sources:
                self.sources.append(k[2])

    def add_minimum_capacity_constraint(self, cij):
        q_terms = []
        for key in self.key_dict.keys():
            q_terms.append(f"q_{key[0]}_{key[1]}_{key[2]}")
        self.prob.variables.add(names=q_terms, types=[self.prob.variables.type.continuous] * len(q_terms))
        for i, j in self.required_connections:
            ind = []
            val = []
            for s in self.sources:
                ind.append(f"q_{i}_{j}_{s}")
                val.append(self.key_dict[(i,j,s)])
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            if isinstance(cij, float) or isinstance(cij, int):
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij])
            else:
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij[i,j]])

    def add_n_det_integer(self):
        n_terms = []
        for key in self.key_dict.keys():
            n_terms.append(f"n_{key[0]}_{key[1]}_{key[2]}")
        self.prob.variables.add(names=n_terms, types=[self.prob.variables.type.integer] * len(n_terms))
        for key in self.key_dict.keys():
            ind = [f"n_{key[0]}_{key[1]}_{key[2]}", f"q_{key[0]}_{key[1]}_{key[2]}"]
            val = [1,-1]
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[0])

    def add_num_sources_constraint(self, Lambda):

        n_source_terms = []
        for s in self.sources:
            n_source_terms.append(f"n_{s}")
        self.prob.variables.add(names=n_source_terms, types=[self.prob.variables.type.integer] * len(n_source_terms))
        for s in self.sources:
            ind = []
            val = []
            for i, j in self.required_connections:
                ind.append(f"n_{i}_{j}_{s}")
                val.append(1/Lambda)
            ind.append(f"n_{s}")
            val.append(-1)
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])

    def add_cost_minimisation(self, c_esource, c_edet):
        obj_vals = []
        for k in self.key_dict.keys():
            obj_vals.append((f"n_{k[0]}_{k[1]}_{k[2]}", 2 * c_edet))
        for s in self.sources:
            obj_vals.append((f"n_{s}", c_esource))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def initial_optimisation_cost_reduction(self, cij, time_limit=1e5, Lambda = 8, c_esource = 10, c_edet = 1):
        t_0 = time.time()
        print("Start Optimisation")
        self.add_minimum_capacity_constraint(cij)
        self.add_n_det_integer()
        self.add_num_sources_constraint(Lambda)
        self.add_cost_minimisation(c_esource, c_edet)
        # self.prob.write("entanglement_optimisation/test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        sol_dict = create_sol_dict(self.prob)
        return sol_dict, self.prob


class Entanglement_Optimisation_Multiple_Dev_Types():

    def __init__(self, prob, required_connections, key_dict_hot, key_dict_cold):
        ## required_connections: [(source, target) : source > target]
        ## key_dict: {(i,j,s): c_ijs} : i > j
        self.prob = prob
        self.required_connections = required_connections
        self.key_dict_hot = key_dict_hot
        self.key_dict_cold = key_dict_cold
        self.sources = []
        for k in self.key_dict_hot.keys():
            if k[2] not in self.sources:
                self.sources.append(k[2])
        for k in self.key_dict_cold.keys():
            if k[2] not in self.sources:
                self.sources.append(k[2])

    def add_minimum_capacity_constraint(self, cij):
        q_terms = []
        for key in self.key_dict_hot.keys():
            q_terms.append(f"q_{key[0]}_{key[1]}_{key[2]}_h")
        for key in self.key_dict_cold.keys():
            q_terms.append(f"q_{key[0]}_{key[1]}_{key[2]}_c")
        self.prob.variables.add(names=q_terms, types=[self.prob.variables.type.continuous] * len(q_terms))
        for i, j in self.required_connections:
            ind = []
            val = []
            for s in self.sources:
                if self.key_dict_hot[(i,j,s)] > 0.00001:
                    ind.append(f"q_{i}_{j}_{s}_h")
                    val.append(self.key_dict_hot[(i, j, s)])
                else:
                    ind_x = [f"q_{i}_{j}_{s}_h"]
                    val_x = [1]
                    lin_expressions = [cplex.SparsePair(ind=ind_x, val=val_x)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])
                if self.key_dict_cold[(i, j, s)] > 0.00001:
                    ind.append(f"q_{i}_{j}_{s}_c")
                    val.append(self.key_dict_cold[(i,j,s)])
                else:
                    ind_x = [f"q_{i}_{j}_{s}_c"]
                    val_x = [1]
                    lin_expressions = [cplex.SparsePair(ind=ind_x, val=val_x)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            if isinstance(cij, float) or isinstance(cij, int):
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij])
            else:
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij[i,j]])

    def add_n_det_integer(self):
        n_terms = []
        for key in self.key_dict_hot.keys():
            n_terms.append(f"n_{key[0]}_{key[1]}_{key[2]}_h")
        for key in self.key_dict_cold.keys():
            n_terms.append(f"n_{key[0]}_{key[1]}_{key[2]}_c")
        self.prob.variables.add(names=n_terms, types=[self.prob.variables.type.integer] * len(n_terms))
        for key in self.key_dict_hot.keys():
            ind = [f"n_{key[0]}_{key[1]}_{key[2]}_h", f"q_{key[0]}_{key[1]}_{key[2]}_h"]
            val = [1,-1]
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[0])
        for key in self.key_dict_cold.keys():
            ind = [f"n_{key[0]}_{key[1]}_{key[2]}_c", f"q_{key[0]}_{key[1]}_{key[2]}_c"]
            val = [1,-1]
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[0])

    def add_num_sources_constraint(self, Lambda):

        n_source_terms = []
        for s in self.sources:
            n_source_terms.append(f"n_{s}")
        self.prob.variables.add(names=n_source_terms, types=[self.prob.variables.type.integer] * len(n_source_terms))
        for s in self.sources:
            ind = []
            val = []
            for i, j in self.required_connections:
                ind.append(f"n_{i}_{j}_{s}_h")
                val.append(1/Lambda)
                ind.append(f"n_{i}_{j}_{s}_c")
                val.append(1 / Lambda)
            ind.append(f"n_{s}")
            val.append(-1)
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])

    def add_cost_minimisation(self, c_esource, c_edet, c_edet_cold):
        obj_vals = []
        for k in self.key_dict_hot.keys():
            obj_vals.append((f"n_{k[0]}_{k[1]}_{k[2]}_h", 2 * c_edet))
        for k in self.key_dict_hot.keys():
            obj_vals.append((f"n_{k[0]}_{k[1]}_{k[2]}_c", 2 * c_edet_cold))
        for s in self.sources:
            obj_vals.append((f"n_{s}", c_esource))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def initial_optimisation_cost_reduction(self, cij, time_limit=1e5, Lambda = 8, c_esource = 10, c_edet = 1, c_edet_cold = 3):
        t_0 = time.time()
        print("Start Optimisation")
        self.add_minimum_capacity_constraint(cij)
        self.add_n_det_integer()
        self.add_num_sources_constraint(Lambda)
        self.add_cost_minimisation(c_esource, c_edet, c_edet_cold)
        # self.prob.write("entanglement_optimisation/test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        sol_dict = create_sol_dict(self.prob)
        return sol_dict, self.prob






if __name__ == "__main__":
    key_dict, required_connections = import_data_ent(data_file_location="test.csv")
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location="test.csv")
    for key in key_dict.keys():
        prob = cplex.Cplex()
        optimisation = Entanglement_Optimisation_Multiple_Dev_Types(prob, required_connections[key],
                                                                    key_dict[key], key_dict_cold[key])
        sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij = 1500, time_limit=2e2, Lambda=1,
                                                                          c_esource=1, c_edet=2,
                                                                          c_edet_cold=2 * 3)
        objective_value = prob.solution.get_objective_value()
        print(prob.solution.get_objective_value())


