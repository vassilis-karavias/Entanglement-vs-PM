import cplex
from entanglement_utils import import_data_ent
from entanglement_optimisation import Entanglement_Optimisation_Multiple_Dev_Types, create_sol_dict
import time


class Entanglement_With_Switching_Opt(Entanglement_Optimisation_Multiple_Dev_Types):

    def __init__(self,prob, required_connections, key_dict_hot, key_dict_cold):
        super().__init__(prob, required_connections, key_dict_hot, key_dict_cold)

    def add_num_sources_constraint(self, Lambda, f_switch):
        n_source_terms = []
        for s in self.sources:
            n_source_terms.append(f"n_{s}")
        self.prob.variables.add(names=n_source_terms, types=[self.prob.variables.type.integer] * len(n_source_terms))
        for s in self.sources:
            ind = []
            val = []
            for key in self.key_dict_hot:
                if key[2] == s:
                    ind.append(f"q_{key[0]}_{key[1]}_{key[2]}_h")
                    val.append(1/Lambda)
            for key in self.key_dict_cold:
                if key[2] == s:
                    ind.append(f"q_{key[0]}_{key[1]}_{key[2]}_c")
                    val.append(1/Lambda)
            ind.append(f"n_{s}")
            val.append(-(1-f_switch))
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])

    def get_all_unique_user_nodes(self):
        user_nodes = []
        for i,j in self.required_connections:
            if i not in user_nodes:
                user_nodes.append(i)
            if j not in user_nodes:
                user_nodes.append(j)
        return user_nodes


    def add_num_detectors_constraint(self, f_switch):
        n_terms = []
        user_nodes = self.get_all_unique_user_nodes()
        for node in user_nodes:
            n_terms.append(f"nd_{node}_h")
            n_terms.append(f"nd_{node}_c")
        self.prob.variables.add(names=n_terms, types=[self.prob.variables.type.integer] * len(n_terms))

        for i in user_nodes:
            ind = []
            val = []
            for key in self.key_dict_hot:
                if key[0] == i:
                    ind.append(f"q_{key[0]}_{key[1]}_{key[2]}_h")
                    val.append(1)
                elif key[1] == i:
                    ind.append(f"q_{key[0]}_{key[1]}_{key[2]}_h")
                    val.append(1)
            ind.append(f"nd_{i}_h")
            val.append(-(1-f_switch))
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])
            ind = []
            val = []
            for key in self.key_dict_cold:
                if key[0] == i:
                    ind.append(f"q_{key[0]}_{key[1]}_{key[2]}_c")
                    val.append(1)
                elif key[1] == i:
                    ind.append(f"q_{key[0]}_{key[1]}_{key[2]}_c")
                    val.append(1)
            ind.append(f"nd_{i}_c")
            val.append(-(1 - f_switch))
            lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])

    def add_cost_minimisation(self, c_esource, c_edet, c_edet_cold):
        obj_vals = []
        user_nodes = self.get_all_unique_user_nodes()
        for node in user_nodes:
            obj_vals.append((f"nd_{node}_h", c_edet))
            obj_vals.append((f"nd_{node}_c", c_edet_cold))
        for s in self.sources:
            obj_vals.append((f"n_{s}", c_esource))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def initial_optimisation_cost_reduction(self, cij, time_limit=1e5, Lambda=8, c_esource=10, c_edet=1, c_edet_cold=3, f_switch = 0.1):
        t_0 = time.time()
        print("Start Optimisation")
        self.add_minimum_capacity_constraint(cij)
        self.add_num_detectors_constraint(f_switch)
        self.add_num_sources_constraint(Lambda, f_switch)
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
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location="test.csv")
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location="test.csv")
    prob = cplex.Cplex()
    optimisation = Entanglement_With_Switching_Opt(prob, required_connections_hot[0], key_dict_hot[0],
                                                   key_dict_cold[0])
    sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij=200, time_limit=2e2, Lambda=1, c_esource=1,
                                                                      c_edet=2,
                                                                      c_edet_cold=0.5 * 2,
                                                                      f_switch=0.0)
    objective_value = prob.solution.get_objective_value()