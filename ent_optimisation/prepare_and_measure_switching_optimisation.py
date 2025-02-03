from _ast import Lambda

import cplex
import time
from entanglement_utils import import_data_pm


def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names=prob.variables.get_names()
    values=prob.solution.get_values()
    sol_dict={names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict




class Prep_And_Measure_Switching_Optimisation(object):

    def __init__(self, prob, key_dict_hot, key_dict_cold):
        self.prob = prob
        self.key_dict_hot = key_dict_hot
        self.key_dict_cold = key_dict_cold
        self.nodes = []
        for i,j in self.key_dict_hot.keys():
            if i not in self.nodes:
                self.nodes.append(i)
            if j not in self.nodes:
                self.nodes.append(j)
        for i,j in self.key_dict_cold.keys():
            if i not in self.nodes:
                self.nodes.append(i)
            if j not in self.nodes:
                self.nodes.append(j)

    def add_capacity_requirement_constraint(self, cij):
        q_terms = []
        for key in self.key_dict_hot.keys():
            q_terms.append(f"q_{key[0]}_{key[1]}_h")
            q_terms.append(f"q_{key[1]}_{key[0]}_h")
        for key in self.key_dict_cold.keys():
            q_terms.append(f"q_{key[0]}_{key[1]}_c")
            q_terms.append(f"q_{key[1]}_{key[0]}_c")
        self.prob.variables.add(names=q_terms, types=[self.prob.variables.type.continuous] * len(q_terms))
        for key in self.key_dict_hot.keys():
            if key in self.key_dict_cold.keys():
                ind = [f"q_{key[0]}_{key[1]}_h", f"q_{key[1]}_{key[0]}_h", f"q_{key[0]}_{key[1]}_c", f"q_{key[1]}_{key[0]}_c"]
                val = [self.key_dict_hot[key], self.key_dict_hot[key], self.key_dict_cold[key],self.key_dict_cold[key]]
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                if isinstance(cij, float) or isinstance(cij, int):
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij])
                else:
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij[key[0], key[1]]])
            else:
                ind = [f"q_{key[0]}_{key[1]}_h", f"q_{key[1]}_{key[0]}_h"]
                val = [self.key_dict_hot[key], self.key_dict_hot[key]]
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                if isinstance(cij, float) or isinstance(cij, int):
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij])
                else:
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij[key[0], key[1]]])
        for key in self.key_dict_cold.keys():
            if key not in self.key_dict_hot.keys():
                ind = [f"q_{key[0]}_{key[1]}_c", f"q_{key[1]}_{key[0]}_c"]
                val = [self.key_dict_cold[key], self.key_dict_cold[key]]
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                if isinstance(cij, float) or isinstance(cij, int):
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij])
                else:
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[cij[key[0], key[1]]])

    def add_sources_on_node_constraint(self, Lambda, f_switch):
        source_terms = []
        for source in self.nodes:
            source_terms.append(f"n_{source}")
        self.prob.variables.add(names=source_terms, types=[self.prob.variables.type.integer] * len(source_terms))
        for source in self.nodes:
            ind = []
            val = []
            for key in self.key_dict_hot.keys():
                if key[0]== source:
                    ind.append(f"q_{key[0]}_{key[1]}_h")
                    val.append(1/Lambda)
                elif key[1] == source:
                    ind.append(f"q_{key[1]}_{key[0]}_h")
                    val.append(1 / Lambda)
            for key in self.key_dict_cold.keys():
                if key[0]== source:
                    ind.append(f"q_{key[0]}_{key[1]}_c")
                    val.append(1/Lambda)
                elif key[1] == source:
                    ind.append(f"q_{key[1]}_{key[0]}_c")
                    val.append(1 / Lambda)
            ind.append(f"n_{source}")
            val.append(-(1-f_switch))
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])


    def add_detector_on_node_constraint(self, f_switch):
        detector_terms = []
        for detector in self.nodes:
            detector_terms.append(f"n_{detector}_h")
            detector_terms.append(f"n_{detector}_c")
        self.prob.variables.add(names=detector_terms, types=[self.prob.variables.type.integer] * len(detector_terms))
        for detector in self.nodes:
            ind = []
            val = []
            for key in self.key_dict_hot.keys():
                if key[1] == detector:
                    ind.append(f"q_{key[0]}_{key[1]}_h")
                    val.append(1)
                elif key[0] == detector:
                    ind.append(f"q_{key[1]}_{key[0]}_h")
                    val.append(1)
            ind.append(f"n_{detector}_h")
            val.append(-(1-f_switch))
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])
            ind = []
            val = []
            for key in self.key_dict_hot.keys():
                if key[1] == detector:
                    ind.append(f"q_{key[0]}_{key[1]}_c")
                    val.append(1)
                elif key[0] == detector:
                    ind.append(f"q_{key[1]}_{key[0]}_c")
                    val.append(1)
            ind.append(f"n_{detector}_c")
            val.append(-(1 - f_switch))
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])

    def minimise_cost_devices(self, cost_source,c_det, c_det_cold):
        obj_vals = []
        for node in self.nodes:
            obj_vals.append((f"n_{node}",cost_source))
            obj_vals.append((f"n_{node}_h",c_det))
            obj_vals.append((f"n_{node}_c",c_det_cold))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def initial_optimisation_cost_reduction(self, cij, time_limit=1e5, Lambda = 8, cost_source = 10, c_det = 1, c_det_cold =5, f_switch = 0.1):
        t_0 = time.time()
        print("Start Optimisation")
        self.add_capacity_requirement_constraint(cij)
        self.add_sources_on_node_constraint(Lambda, f_switch)
        self.add_detector_on_node_constraint(f_switch)
        self.minimise_cost_devices(cost_source, c_det, c_det_cold)
        self.prob.write("test_1.lp")
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
    key_dict_hot = import_data_pm(data_file_location="test_pm.csv")
    key_dict_cold= import_data_pm(data_file_location="test_pm.csv")
    prob = cplex.Cplex()
    prob = cplex.Cplex()
    optimisation = Prep_And_Measure_Switching_Optimisation(prob, key_dict_hot[0],
                                                           key_dict_cold[0])
    sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij = 200, time_limit=2e2, Lambda=1, cost_source=1,
                                                                      c_det=2,
                                                                      c_det_cold=3 * 2,
                                                                      f_switch=0.0)
    objective_value = prob.solution.get_objective_value()