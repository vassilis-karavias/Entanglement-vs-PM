import numpy as np
from entanglement_utils import import_data_pm
import cplex
import scipy.optimize as opt


def get_cost_for_prep_measure(key_dict, c_det_pair, c_source, c_ij):
    cost = 0
    for key in key_dict.keys():
        if key_dict[key] > 0.000001:
            if isinstance(c_ij, float) or isinstance(c_ij, int):
                number_devices = np.ceil(c_ij / key_dict[key])
            else:
                number_devices = np.ceil(c_ij[key] / key_dict[key])
        else:
            return np.infty
        cost += (c_det_pair + c_source) * number_devices
    return cost


def add_minimum(prob, cost_devs_hot, cost_devs_cold):
    obj_vals = []
    obj_vals.append((f"N_h", cost_devs_hot))
    obj_vals.append((f"N_c", cost_devs_cold))
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)

def add_capacity_constraint(prob, rate_hot, rate_cold, c_ij):
    n_terms = ["N_h", "N_c"]
    prob.variables.add(names=n_terms, types=[prob.variables.type.integer] * len(n_terms))
    if rate_hot > 0.000001:
        ind = [f"N_h"]
        val = [rate_hot]
    else:
        ind = [f"N_h"]
        val = [1]
        lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])
        ind = None
        val = None
    if rate_cold > 0.000001:
        if ind == None:
            ind = [f"N_c"]
            val = [rate_cold]
        else:
            ind.append(f"N_c")
            val.append(rate_cold)
        lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[c_ij])
    else:
        ind = [f"N_c"]
        val = [1]
        lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])
        ind = [f"N_h", f"N_c"]
        val = [rate_hot, rate_cold]
        lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[c_ij])


def get_cost_for_prep_measure_mult_devs(key_dict_hot, key_dict_cold, cost_devs_hot, cost_devs_cold, c_ij):
    cost = 0
    cost_terms = {}
    i = 0
    for key in key_dict_hot.keys():
        if isinstance(c_ij, float) or isinstance(c_ij, int):
            try:
                prob = cplex.Cplex()
                add_capacity_constraint(prob, rate_hot=key_dict_hot[key], rate_cold=key_dict_cold[key], c_ij=c_ij)
                add_minimum(prob, cost_devs_hot, cost_devs_cold)
                prob.parameters.mip.limits.cutpasses.set(1)
                prob.parameters.mip.strategy.probe.set(-1)
                prob.parameters.mip.strategy.variableselect.set(4)
                prob.parameters.mip.strategy.kappastats.set(1)
                prob.parameters.mip.tolerances.mipgap.set(float(0.01))
                prob.solve()
                cost += prob.solution.get_objective_value()
                cost_terms[key] = prob.solution.get_objective_value()
            except:
                return None
        else:
            try:
                prob = cplex.Cplex()
                add_capacity_constraint(prob, rate_hot=key_dict_hot[key], rate_cold=key_dict_cold[key], c_ij=c_ij[key])
                add_minimum(prob, cost_devs_hot, cost_devs_cold)
                prob.parameters.mip.limits.cutpasses.set(1)
                prob.parameters.mip.strategy.probe.set(-1)
                prob.parameters.mip.strategy.variableselect.set(4)
                prob.parameters.mip.strategy.kappastats.set(1)
                prob.parameters.mip.tolerances.mipgap.set(float(0.01))
                prob.solve()
                cost += prob.solution.get_objective_value()
                cost_terms[key] = prob.solution.get_objective_value()
                i += 1
            except:
                return None
    return cost



if __name__ == "__main__":
    key_dict = import_data_pm(data_file_location = "trial_mult_graphs_pm.csv")
    cost = get_cost_for_prep_measure_mult_devs(key_dict_hot = key_dict[0], key_dict_cold = key_dict[0], cost_devs_hot = 2, cost_devs_cold = 5, c_ij = 1000.0)
    # cost = get_cost_for_prep_measure(key_dict, c_source= 0.9, c_det_pair=10, c_ij=10000.0)
    print(cost)
