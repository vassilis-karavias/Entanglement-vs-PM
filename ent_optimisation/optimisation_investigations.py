import matplotlib.colors

from prep_and_measure_optimal import get_cost_for_prep_measure_mult_devs
from entanglement_utils import import_data_pm, import_data_ent
from prepare_and_measure_switching_optimisation import Prep_And_Measure_Switching_Optimisation
from entanglement_optimisation import Entanglement_Optimisation_Multiple_Dev_Types
from entanglement_switching_optimisation import Entanglement_With_Switching_Opt
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator
import os
import csv
import pandas as pd
import numpy as np
import cplex


###### NOTE: USE Switchednetworkoptimisation/main_switched.py to get the network data and then use Entanglement Investigation/graph_generator
###### to get the datafiles.

def split_sol_dict_pm(sol_dict):
    n_devs = {}
    q_terms = {}
    for key in sol_dict.keys():
        if key[0] == "n":
            n_devs[key] = sol_dict[key]
        else:
            q_terms[key] = sol_dict[key]
    return n_devs, q_terms




def compare_for_different_detector_costs(ent_data_file_hot, pm_data_file_hot,ent_data_file_cold, pm_data_file_cold, Lambda, cij, min_det_cost, max_det_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None):
    # key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    # key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost = last_row_explored["Detector_Cost"].iloc[0]
            current_node = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost = min_det_cost
            current_node = None
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost = min_det_cost
        current_node = None
    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost_pm = last_row_explored["Detector_Cost"].iloc[0]
        else:
            current_detector_cost_pm = min_det_cost
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost_pm = min_det_cost

    objective_list = {}
    for detector_cost in np.arange(current_detector_cost, max_det_cost + 0.05, 0.05):
        for id in key_dict_hot.keys():
            if current_node != None:
                if id != current_node:
                    continue
                else:
                    current_node = None
                    continue
            try:
                prob = cplex.Cplex()
                optimisation = Entanglement_Optimisation_Multiple_Dev_Types(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=1e5, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost)
                objective_value = prob.solution.get_objective_value()
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
    
            except:
                continue
    objective_list_pm = {}
    for detector_cost in np.arange(current_detector_cost_pm, max_det_cost + 0.05, 0.05):
        for id in key_dict_pm_hot.keys():
            try:
                cost = get_cost_for_prep_measure_mult_devs(key_dict_hot = key_dict_pm_hot[id], key_dict_cold = key_dict_pm_cold[id], cost_devs_hot = detector_cost + 1, cost_devs_cold = detector_cold_cost * detector_cost + 1, c_ij = cij)
                if cost != None:
                    if data_storage_location_keep_each_loop_pm != None:
                        dictionary = [
                            {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": cost}]
                        dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
            except:
                continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list.keys():
                objective_list[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list[row["Detector_Cost"]].keys():
                objective_list[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    if data_storage_location_keep_each_loop_pm != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list_pm.keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list_pm[row["Detector_Cost"]].keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for det_cost in objective_list.keys():
        for key in objective_list[det_cost].keys():
            if det_cost not in objective_values.keys():
                objective_values[det_cost] = [(objective_list[det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key]]
            else:
                objective_values[det_cost].append((objective_list[det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0, label = "Fractional Cost Difference")
    plt.xlabel("Relative Detector Cost", fontsize=10)
    plt.ylabel("Fractional Difference Between Network Costs", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("network_cost_diff.png")
    plt.show()



def compare_for_different_detector_costs_and_lambda(ent_data_file_hot, pm_data_file_hot,ent_data_file_cold, pm_data_file_cold, Lambda_max, cij, min_det_cost, max_det_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None):
    # key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    # key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_lambda = last_row_explored["Lambda"].iloc[0]
            current_detector_cost = last_row_explored["Detector_Cost"].iloc[0]
            current_node = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost = min_det_cost
            current_lambda = 1
            current_node = None
            dictionary_fieldnames = ["Lambda", "Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost = min_det_cost
        current_node = None
        current_lambda = 1
    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost_pm = last_row_explored["Detector_Cost"].iloc[0]
            current_node_pm = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost_pm = min_det_cost
            current_node_pm = None
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost_pm = min_det_cost
        current_node_pm = None
    objective_list = {}
    for Lambda in [32]:
        for detector_cost in np.arange(min_det_cost, max_det_cost + 0.05, 0.05):
            if abs(current_detector_cost - min_det_cost) > 0.01:
                if abs(current_detector_cost - detector_cost) > 0.01:
                    continue
                else:
                    current_detector_cost = min_det_cost
                    continue
            for id in key_dict_hot.keys():
                if current_node != None:
                    if id != current_node:
                        continue
                    else:
                        current_node = None
                        continue
                try:
                    prob = cplex.Cplex()
                    optimisation = Entanglement_Optimisation_Multiple_Dev_Types(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                    sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=2e2, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost)
                    objective_value = prob.solution.get_objective_value()
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"Lambda": Lambda,"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                        dictionary_fieldnames = ["Lambda", "Detector_Cost","Graph_ID", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
    
                except:
                    continue
    objective_list_pm = {}
    for detector_cost in np.arange(current_detector_cost_pm, max_det_cost + 0.05, 0.05):
        for id in key_dict_pm_hot.keys():
            if current_node_pm != None:
                if id != current_node_pm:
                    continue
                else:
                    current_node_pm = None
                    continue
            try:
                cost = get_cost_for_prep_measure_mult_devs(key_dict_hot = key_dict_pm_hot[id], key_dict_cold = key_dict_pm_cold[id], cost_devs_hot = detector_cost, cost_devs_cold = detector_cold_cost * detector_cost, c_ij = cij)
                if cost != None:
                    if data_storage_location_keep_each_loop_pm != None:
                        dictionary = [
                            {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": cost}]
                        dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
            except:
                continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Lambda"] not in objective_list.keys():
                objective_list[row["Lambda"]] = {row["Detector_Cost"]: {row["Graph_ID"]: row["objective_value"]}}
            elif row["Detector_Cost"] not in objective_list[row["Lambda"]].keys():
                objective_list[row["Lambda"]][row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list[row["Lambda"]][row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    if data_storage_location_keep_each_loop_pm != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list_pm.keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list_pm[row["Detector_Cost"]].keys():
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
            else:
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for Lambda in objective_list.keys():
        for det_cost in objective_list[Lambda].keys():
            for key in objective_list[Lambda][det_cost].keys():
                if key in objective_list_pm[det_cost].keys():
                    if Lambda not in objective_values.keys():
                        objective_values[Lambda] = {det_cost: [(objective_list[Lambda][det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key]]}
                    elif det_cost not in objective_values[Lambda].keys():
                        objective_values[Lambda][det_cost] = [(objective_list[Lambda][det_cost][key] - objective_list_pm[det_cost][key])/ objective_list_pm[det_cost][key]]
                    else:
                        objective_values[Lambda][det_cost].append(
                            (objective_list[Lambda][det_cost][key] - objective_list_pm[det_cost][key]) /
                            objective_list_pm[det_cost][key])
    fig = plt.figure()
    ax = plt.axes()
    mean_objectives = {}
    std_objectives = {}
    for Lambda in objective_values.keys():
        for key in objective_values[Lambda].keys():
            mean_objectives[key] = np.mean(objective_values[Lambda][key])
            std_objectives[key] = np.std(objective_values[Lambda][key])
        mean_differences = []
        std_differences = []
        # topologies
        x = []
        for key in mean_objectives.keys():
            mean_differences.append(mean_objectives[key])
            std_differences.append(std_objectives[key])
            x.append(key)
        ax.errorbar(x, mean_differences, yerr=std_differences, capsize= 0, label = f"Number of Channels: {Lambda}")
    ax.axhline(y=0, linestyle='--', color='black')
    # ax.set_yscale("log")
    # ax.set_ylim([-10, 1])
    ax.set_xlabel("Relative Detector Cost", fontsize=14)
    ax.set_ylabel("Fractional Difference Between Network Costs", fontsize=14)
    ax.legend(loc='best', fontsize='medium')
    plt.savefig("network_cost_diff_lambda_with_cmin_1000_zero_source_pm_cost.png")
    plt.show()






def compare_for_different_source_costs_and_lambda(pm_data_file_hot, pm_data_file_cold, cij, min_source_cost, max_source_cost, min_detector_cost, max_detector_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None):
    # key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    # key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_lambda = last_row_explored["Lambda"].iloc[0]
            current_detector_cost = last_row_explored["Detector_Cost"].iloc[0]
            current_node = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost = 1
            current_lambda = 1
            current_node = None
            dictionary_fieldnames = ["Lambda", "Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost = 1
        current_node = None
        current_lambda = 1
    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_source_cost_pm = last_row_explored["Source_Cost"].iloc[0]
            current_detector_cost_pm = last_row_explored["Detector_Cost"].iloc[0]
            current_node_pm = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_source_cost_pm = min_source_cost
            current_node_pm = None
            current_detector_cost_pm = min_detector_cost
            dictionary_fieldnames = ["Source_Cost", "Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_source_cost_pm = min_source_cost
        current_detector_cost_pm = min_detector_cost
        current_node_pm = None
    objective_list = {}
    for Lambda in [32]:
        for detector_cost in np.arange(min_det_cost, max_det_cost + 0.05, 0.05):
            if abs(current_detector_cost - min_det_cost) > 0.01:
                if abs(current_detector_cost - detector_cost) > 0.01:
                    continue
                else:
                    current_detector_cost = min_det_cost
                    continue
            for id in key_dict_hot.keys():
                if current_node != None:
                    if id != current_node:
                        continue
                    else:
                        current_node = None
                        continue
                try:
                    prob = cplex.Cplex()
                    optimisation = Entanglement_Optimisation_Multiple_Dev_Types(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                    sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=2e2, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost)
                    objective_value = prob.solution.get_objective_value()
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"Lambda": Lambda,"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                        dictionary_fieldnames = ["Lambda", "Detector_Cost","Graph_ID", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
    
                except:
                    continue
    objective_list_pm = {}
    no_sol_list = []
    for detector_cost in np.arange(current_detector_cost_pm, max_detector_cost + 0.05, 0.05):
        for source_cost in np.arange(current_source_cost_pm, max_source_cost + 0.05, 0.05):
            for id in key_dict_pm_hot.keys():
                if id not in no_sol_list:
                    if current_node_pm != None:
                        if id != current_node_pm:
                            continue
                        else:
                            current_node_pm = None
                            continue
                    try:
                        cost = get_cost_for_prep_measure_mult_devs(key_dict_hot = key_dict_pm_hot[id], key_dict_cold = key_dict_pm_cold[id], cost_devs_hot = detector_cost + source_cost, cost_devs_cold = detector_cold_cost * detector_cost + source_cost, c_ij = cij)
                        if cost != None:
                            if data_storage_location_keep_each_loop_pm != None:
                                dictionary = [
                                    {"Source_Cost": source_cost, "Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": cost}]
                                dictionary_fieldnames = ["Source_Cost","Detector_Cost", "Graph_ID", "objective_value"]
                                if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                                    with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                        writer.writerows(dictionary)
                                else:
                                    with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                        writer.writeheader()
                                        writer.writerows(dictionary)
                    except:
                        no_sol_list.append(id)
                        continue
        current_source_cost_pm = min_source_cost

    current_source_cost_pm = min_source_cost
    current_detector_cost_pm = min_detector_cost
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Lambda"] not in objective_list.keys():
                objective_list[row["Lambda"]] = {round(row["Detector_Cost"],2): {row["Graph_ID"]: row["objective_value"]}}
            elif round(row["Detector_Cost"],2) not in objective_list[row["Lambda"]].keys():
                objective_list[row["Lambda"]][round(row["Detector_Cost"],2)] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list[row["Lambda"]][round(row["Detector_Cost"],2)][row["Graph_ID"]] = row["objective_value"]
        if data_storage_location_keep_each_loop_pm != None:
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            for index, row in plot_information.iterrows():
                if round(row["Source_Cost"],2) not in objective_list_pm.keys():
                    objective_list_pm[round(row["Source_Cost"],2)] = {round(row["Detector_Cost"],2): {row["Graph_ID"]: row["objective_value"]}}
                elif round(row["Detector_Cost"],2) not in objective_list_pm[round(row["Source_Cost"],2)].keys():
                    objective_list_pm[round(row["Source_Cost"],2)][round(row["Detector_Cost"],2)] = {row["Graph_ID"]: row["objective_value"]}
                else:
                    objective_list_pm[round(row["Source_Cost"],2)][round(row["Detector_Cost"],2)][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for Lambda in objective_list.keys():
        for det_cost in objective_list[Lambda].keys():
            for s_cost in objective_list_pm.keys():
                for key in objective_list[Lambda][det_cost].keys():
                    if det_cost in objective_list_pm[s_cost].keys():
                        if key in objective_list_pm[s_cost][det_cost].keys():
                            if Lambda not in objective_values.keys():
                                objective_values[Lambda] = {s_cost: {det_cost: {key : (objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][det_cost][key] )/ objective_list_pm[s_cost][det_cost][key]}}}
                            elif s_cost not in objective_values[Lambda].keys():
                                objective_values[Lambda][s_cost] = {det_cost: {key : (objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][det_cost][key] )/ objective_list_pm[s_cost][det_cost][key]}}
                            elif det_cost not in objective_values[Lambda][s_cost].keys():
                                objective_values[Lambda][s_cost][det_cost] =  {key : (objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][det_cost][key] )/ objective_list_pm[s_cost][det_cost][key]}
                            else:
                                objective_values[Lambda][s_cost][det_cost][key] = (objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][det_cost][key] )/ objective_list_pm[s_cost][det_cost][key]

    for Lambda in objective_values.keys():
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.arange(current_source_cost_pm, max_source_cost + 0.05, 0.05)
        Y = np.arange(current_detector_cost_pm, max_detector_cost + 0.05, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = np.empty((len(X), len(X[0])))
        for i in range(len(X)):
            for j in range(len(X[i])):
                Z[i][j] = np.mean(list(objective_values[Lambda][round(X[i][j],2)][round(Y[i][j],2)].values()))

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, alpha = 0.5, antialiased=False)
        a = 0 * X
        surf_2 = ax.plot_surface(X, Y, a, alpha = 0.75, color = "green")
        intersection_mask = np.isclose(Z, a, atol=0.001)
        ax.contour3D(X, Y, Z, levels=[0], colors='black', alpha = 1, linewidths=4)

        zmin, zmax = Z.min(), Z.max()
        sig_fig_min = int(np.floor(np.log10(abs(zmin))))
        sig_fig_max = int(np.floor(np.log10(abs(zmax))))

        ax.contour3D(X, Y, Z, levels=[0], offset= np.floor(zmin / sig_fig_min) * sig_fig_min, colors='black', alpha=1, linewidths=4)
        if sig_fig_max != 0:
            ax.set_zlim(np.floor(zmin / sig_fig_min) * sig_fig_min, np.ceil(zmax / sig_fig_max) * sig_fig_max)
        else:
            ax.set_zlim(-np.floor(-zmin / sig_fig_min) * sig_fig_min, np.ceil(zmax))
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set(
            xlabel='Relative Source Cost',
            ylabel='Relative Detector Cost',
            zlabel='Fractional Difference Between Network Costs',
        )
        ax.view_init(20,50, 0)
        plt.savefig(f"3d_plot_source_det_var_1000bits_{Lambda}.png")
        plt.show()

    # all in one
    current_source_cost_pm = 0.2
    current_detector_cost_pm = 0.2
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(current_source_cost_pm, max_source_cost + 0.05, 0.05)
    Y = np.arange(current_detector_cost_pm, max_detector_cost + 0.05, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = np.empty((len(X), len(X[0])))
    for i in range(len(X)):
        for j in range(len(X[i])):
            Z[i][j] = np.mean(list(objective_values[1.0][round(X[i][j], 2)][round(Y[i][j], 2)].values()))
    vmax = Z.max()
    for i in range(len(X)):
        for j in range(len(X[i])):
            Z[i][j] = np.mean(list(objective_values[32.0][round(X[i][j], 2)][round(Y[i][j], 2)].values()))
    vmin = Z.min()
    sig_fig_min = int(np.floor(np.log10(abs(vmin))))
    sig_fig_max = int(np.floor(np.log10(abs(vmax))))
    color_map = cm.coolwarm
    norm = colors.Normalize(vmin = vmin, vmax = vmax)
    color_map_norm = cm.ScalarMappable(norm = norm, cmap = color_map)
    k = 1
    for Lambda in objective_values.keys():

        for i in range(len(X)):
            for j in range(len(X[i])):
                Z[i][j] = np.mean(list(objective_values[Lambda][round(X[i][j], 2)][round(Y[i][j], 2)].values()))

        surf = ax.plot_surface(X, Y, Z, facecolors=color_map(norm(Z)), linewidth=0, alpha=k/10, antialiased=False, label = f"No Channels: {Lambda}")
        a = 0 * X

        intersection_mask = np.isclose(Z, a, atol=0.001)
        ax.contour3D(X, Y, Z, levels=[0], colors='black', alpha=k/10, linewidths=4)
        ax.contour3D(X, Y, Z, levels=[0], offset=np.floor(vmin / sig_fig_min) * sig_fig_min, colors='black', alpha=k/10, linewidths=4)
        k += 1
    surf_2 = ax.plot_surface(X, Y, a, alpha=0.75, color="green")
    if sig_fig_max != 0:
        ax.set_zlim(np.floor(vmin / sig_fig_min) * sig_fig_min, np.ceil(vmax / sig_fig_max) * sig_fig_max)
    else:
        ax.set_zlim(-np.floor(-vmin / sig_fig_min) * sig_fig_min, np.ceil(vmax))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(color_map_norm, shrink=0.5, aspect=5)
    ax.set(
        xlabel='Relative Source Cost',
        ylabel='Relative Detector Cost',
        zlabel='Fractional Difference Between Network Costs',
    )
    # ax.legend()
    ax.view_init(5, 30, 0)
    plt.savefig(f"3d_plot_source_det_var_1000bits_all_lambda.png")
    plt.show()



    # if data_storage_location_keep_each_loop != None:
    #     plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
    #     for index, row in plot_information.iterrows():
    #         if row["Lambda"] not in objective_list.keys():
    #             objective_list[row["Lambda"]] = {row["Detector_Cost"]: {row["Graph_ID"]: row["objective_value"]}}
    #         elif row["Detector_Cost"] not in objective_list[row["Lambda"]].keys():
    #             objective_list[row["Lambda"]][row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
    #         else:
    #             objective_list[row["Lambda"]][row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    # if data_storage_location_keep_each_loop_pm != None:
    #     plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
    #     for index, row in plot_information.iterrows():
    #         if row["Source_Cost"] not in objective_list_pm.keys():
    #             objective_list_pm[row["Source_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
    #         elif row["Graph_ID"] not in objective_list_pm[row["Source_Cost"]].keys():
    #             objective_list_pm[row["Source_Cost"]][row["Graph_ID"]] = row["objective_value"]
    #         else:
    #             objective_list_pm[row["Source_Cost"]][row["Graph_ID"]] = row["objective_value"]
    # objective_values = {}
    # for Lambda in objective_list.keys():
    #     for det_cost in objective_list[Lambda].keys():
    #         if abs(det_cost - detector_cost) < 0.001:
    #             for s_cost in objective_list_pm.keys():
    #                 for key in objective_list[Lambda][det_cost].keys():
    #                     if key in objective_list_pm[s_cost].keys():
    #                         if Lambda not in objective_values.keys():
    #                             objective_values[Lambda] = {s_cost: [(objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][key] )/ objective_list_pm[s_cost][key]]}
    #                         elif s_cost not in objective_values[Lambda].keys():
    #                             objective_values[Lambda][s_cost] = [(objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][key])/ objective_list_pm[s_cost][key]]
    #                         else:
    #                             objective_values[Lambda][s_cost].append(
    #                                 (objective_list[Lambda][det_cost][key] - objective_list_pm[s_cost][key]) /
    #                                 objective_list_pm[s_cost][key])
    # fig = plt.figure()
    # ax = plt.axes()
    # mean_objectives = {}
    # std_objectives = {}
    # for Lambda in objective_values.keys():
    #     for key in objective_values[Lambda].keys():
    #         mean_objectives[key] = np.mean(objective_values[Lambda][key])
    #         std_objectives[key] = np.std(objective_values[Lambda][key])
    #     mean_differences = []
    #     std_differences = []
    #     # topologies
    #     x = []
    #     for key in mean_objectives.keys():
    #         mean_differences.append(mean_objectives[key])
    #         std_differences.append(std_objectives[key])
    #         x.append(key)
    #     ax.errorbar(x, mean_differences, yerr=std_differences, capsize= 0, label = f"Number of Channels: {Lambda}")
    # ax.axhline(y=0, linestyle='--', color='black')
    # # ax.set_yscale("log")
    # # ax.set_ylim([-10, 1])
    # ax.set_xlabel("Relative Source Cost", fontsize=14)
    # ax.set_ylabel("Fractional Difference Between Network Costs", fontsize=14)
    # ax.legend(loc='best', fontsize='medium')
    # plt.savefig("network_cost_diff_lambda_with_cmin_10_source_pm_cost_var_det_cost_1.png")
    # plt.show()






def compare_switch_for_different_detector_costs(ent_data_file_hot, pm_data_file_hot,ent_data_file_cold, pm_data_file_cold, Lambda, cij, fswitch, min_det_cost, max_det_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None):
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost = last_row_explored["Detector_Cost"].iloc[0]
        else:
            current_detector_cost = min_det_cost
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost = min_det_cost

    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost_pm = last_row_explored["Detector_Cost"].iloc[0]
        else:
            current_detector_cost_pm = min_det_cost
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost_pm = min_det_cost
    objective_list = {}
    for detector_cost in np.arange(current_detector_cost, max_det_cost + 0.05, 0.05):
        for id in key_dict_hot.keys():
            try:
                prob = cplex.Cplex()
                optimisation = Entanglement_With_Switching_Opt(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=2e2, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost, f_switch=fswitch)
                objective_value = prob.solution.get_objective_value()
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)

            except:
                continue
    objective_list_pm = {}
    for detector_cost in np.arange(current_detector_cost_pm, max_det_cost + 0.05, 0.05):
        for id in key_dict_pm_hot.keys():
            try:
                prob = cplex.Cplex()
                optimisation = Prep_And_Measure_Switching_Optimisation(prob, key_dict_pm_hot[id],
                                                               key_dict_pm_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=2e2, Lambda = Lambda, cost_source = 1, c_det = detector_cost, c_det_cold =detector_cold_cost * detector_cost, f_switch = fswitch)
                objective_value = prob.solution.get_objective_value()
                if data_storage_location_keep_each_loop_pm != None:
                    dictionary = [
                        {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                        with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
            except:
                continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list.keys():
                objective_list[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list[row["Detector_Cost"]].keys():
                objective_list[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    if data_storage_location_keep_each_loop_pm != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list_pm.keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list_pm[row["Detector_Cost"]].keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for det_cost in objective_list.keys():
        for key in objective_list[det_cost].keys():
            if det_cost not in objective_values.keys():
                objective_values[det_cost] = [(objective_list[det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key]]
            else:
                objective_values[det_cost].append((objective_list[det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0, label = "Fractional Cost Difference")
    plt.xlabel("Relative Detector Cost", fontsize=10)
    plt.ylabel("Fractional Difference Between Network Costs", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("network_cost_diff_switch.png")
    plt.show()



def compare_switch_for_different_detector_costs_lambda(ent_data_file_hot, pm_data_file_hot,ent_data_file_cold, pm_data_file_cold, Lambda_max, cij, fswitch, min_det_cost, max_det_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None):
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_lambda = last_row_explored["Lambda"].iloc[0]
            current_detector_cost = last_row_explored["Detector_Cost"].iloc[0]
            current_node = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost = min_det_cost
            current_lambda = 1
            current_node = None
            dictionary_fieldnames = ["Lambda", "Detector_Cost", "Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost = min_det_cost
        current_node = None
        current_lambda = 1
    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost_pm = last_row_explored["Detector_Cost"].iloc[0]
            current_node_pm = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost_pm = min_det_cost
            current_node_pm = None
            dictionary_fieldnames = ["Detector_Cost", "Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost_pm = min_det_cost
        current_node_pm = None
    objective_list = {}
    for Lambda in [16,32]:
        if current_detector_cost == None:
            current_detector_cost = min_det_cost
        for detector_cost in np.arange(min_det_cost, max_det_cost + 0.05, 0.05):
            # if current_detector_cost != None:
            #     if abs(current_detector_cost - detector_cost) > 0.01:
            #         continue
            #     else:
            #         current_detector_cost = None
            for id in key_dict_hot.keys():
                # if current_node != None:
                #     if id != current_node:
                #         continue
                #     else:
                #         current_node = None
                #         continue
                if id in [4,6,12,14,15,16,18,20,22,25,26,27,30,31,32,33,34,39]:
                    try:
                        prob = cplex.Cplex()
                        optimisation = Entanglement_With_Switching_Opt(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                        sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=2e2, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost, f_switch=fswitch)
                        objective_value = prob.solution.get_objective_value()
                        if data_storage_location_keep_each_loop != None:
                            dictionary = [
                                {"Lambda": Lambda, "Detector_Cost": detector_cost, "Graph_ID": id,
                                 "objective_value": objective_value}]
                            dictionary_fieldnames = ["Lambda", "Detector_Cost", "Graph_ID", "objective_value"]
                            if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writerows(dictionary)
                            else:
                                with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                    writer.writeheader()
                                    writer.writerows(dictionary)
    
                    except:
                        continue
    objective_list_pm = {}
    for detector_cost in np.arange(current_detector_cost_pm, max_det_cost + 0.05, 0.05):
        for id in key_dict_pm_hot.keys():
            if current_node_pm != None:
                if id != current_node_pm:
                    continue
                else:
                    current_node_pm = None
                    continue
            try:
                prob = cplex.Cplex()
                optimisation = Prep_And_Measure_Switching_Optimisation(prob, key_dict_pm_hot[id],
                                                               key_dict_pm_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=1e2, Lambda = 1, cost_source = 1, c_det = detector_cost, c_det_cold =detector_cold_cost * detector_cost, f_switch = fswitch)
                objective_value = prob.solution.get_objective_value()
                # n_terms, q_terms = split_sol_dict_pm(sol_dict)
                if data_storage_location_keep_each_loop_pm != None:
                    dictionary = [
                        {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                        with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
            except:
                continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Lambda"] not in objective_list.keys():
                objective_list[row["Lambda"]] = {row["Detector_Cost"]: {row["Graph_ID"]: row["objective_value"]}}
            elif row["Detector_Cost"] not in objective_list[row["Lambda"]].keys():
                objective_list[row["Lambda"]][row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list[row["Lambda"]][row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    if data_storage_location_keep_each_loop_pm != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list_pm.keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list_pm[row["Detector_Cost"]].keys():
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
            else:
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for Lambda in objective_list.keys():
        for det_cost in objective_list[Lambda].keys():
            for key in objective_list[Lambda][det_cost].keys():
                if key in objective_list_pm[det_cost].keys():
                    if Lambda not in objective_values.keys():
                        objective_values[Lambda] = {det_cost: [
                            (objective_list[Lambda][det_cost][key] - objective_list_pm[det_cost][key]) /
                            objective_list_pm[det_cost][key]]}
                    elif det_cost not in objective_values[Lambda].keys():
                        objective_values[Lambda][det_cost] = [
                            (objective_list[Lambda][det_cost][key] - objective_list_pm[det_cost][key]) /
                            objective_list_pm[det_cost][key]]
                    else:
                        objective_values[Lambda][det_cost].append(
                            (objective_list[Lambda][det_cost][key] - objective_list_pm[det_cost][key]) /
                            objective_list_pm[det_cost][key])
    fig = plt.figure()
    mean_objectives = {}
    std_objectives = {}
    for Lambda in objective_values.keys():
        for key in objective_values[Lambda].keys():
            mean_objectives[key] = np.mean(objective_values[Lambda][key])
            std_objectives[key] = np.std(objective_values[Lambda][key])
        mean_differences = []
        std_differences = []
        # topologies
        x = []
        for key in mean_objectives.keys():
            mean_differences.append(mean_objectives[key])
            std_differences.append(std_objectives[key])
            x.append(key)
        plt.errorbar(x, mean_differences, yerr=std_differences, capsize=0,
                     label=f"Number of Channels: {Lambda}")
    plt.axhline(y=0, linestyle='--', color='black')
    plt.xlabel("Relative Detector Cost", fontsize=14)
    plt.ylabel("Fractional Difference Between Network Costs", fontsize=14)
    plt.legend(loc='best', fontsize='medium')
    plt.savefig("network_cost_diff_switch_lambda_with_cmin_10_2.png")
    plt.show()



def compare_switch_f_switch(ent_data_file_hot, pm_data_file_hot,ent_data_file_cold, pm_data_file_cold, Lambda, cij, min_fswitch, max_fswitch, detector_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None):
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_f_switch = last_row_explored["f_switch"].iloc[0]
        else:
            current_f_switch = min_fswitch
            dictionary_fieldnames = ["f_switch","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_f_switch = min_fswitch

    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_f_switch_pm = last_row_explored["f_switch"].iloc[0]
        else:
            current_f_switch_pm = min_fswitch
            dictionary_fieldnames = ["f_switch","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_f_switch_pm = min_fswitch
    objective_list = {}
    for f_switch in np.arange(current_f_switch, max_fswitch + 0.05, 0.05):
        for id in key_dict_hot.keys():
            try:
                prob = cplex.Cplex()
                optimisation = Entanglement_With_Switching_Opt(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=1e5, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost, f_switch=f_switch)
                objective_value = prob.solution.get_objective_value()
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"f_switch": f_switch,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["f_switch","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)

            except:
                continue
    objective_list_pm = {}
    for f_switch in np.arange(current_f_switch_pm, max_fswitch + 0.05, 0.05):
        for id in key_dict_pm_hot.keys():
            try:
                prob = cplex.Cplex()
                optimisation = Prep_And_Measure_Switching_Optimisation(prob, key_dict_pm_hot[id],
                                                               key_dict_pm_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=1e5, Lambda = Lambda, cost_source = 1, c_det = detector_cost, c_det_cold =detector_cold_cost, f_switch = f_switch)
                objective_value = prob.solution.get_objective_value()
                if data_storage_location_keep_each_loop_pm != None:
                    dictionary = [
                        {"f_switch": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["f_switch","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                        with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
            except:
                continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["f_switch"] not in objective_list.keys():
                objective_list[row["f_switch"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list[row["Detector_Cost"]].keys():
                objective_list[row["f_switch"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list[row["f_switch"]][row["Graph_ID"]] = row["objective_value"]
    if data_storage_location_keep_each_loop_pm != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
        for index, row in plot_information.iterrows():
            if row["f_switch"] not in objective_list_pm.keys():
                objective_list_pm[row["f_switch"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list_pm[row["Detector_Cost"]].keys():
                objective_list_pm[row["f_switch"]] = {row["Graph_ID"]: row["objective_value"]}
            else:
                objective_list_pm[row["f_switch"]][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for f_switch_val in objective_list.keys():
        for key in objective_list[f_switch_val].keys():
            if f_switch_val not in objective_values.keys():
                objective_values[f_switch_val] = [(objective_list[f_switch_val][key] - objective_list_pm[f_switch_val][key] )/ objective_list_pm[f_switch_val][key]]
            else:
                objective_values[f_switch_val].append((objective_list[f_switch_val][key] - objective_list_pm[f_switch_val][key] )/ objective_list_pm[f_switch_val][key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0, label = "Fractional Cost Difference")
    plt.xlabel("f_switch", fontsize=14)
    plt.ylabel("Fractional Difference Between Network Costs", fontsize=14)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("network_cost_diff_f_switch_var.png")
    plt.show()

def get_values_at_which_ent_equals_pm(ent_data_file_hot, pm_data_file_hot,ent_data_file_cold, pm_data_file_cold, Lambda, cij, min_det_cost, max_det_cost, detector_cold_cost, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None, data_store_place = None, current_connectivity = 3.5):
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost = last_row_explored["Detector_Cost"].iloc[0]
            current_node = last_row_explored["Graph_ID"].iloc[0]
        else:
            current_detector_cost = min_det_cost
            current_node = None
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost = min_det_cost
        current_node = None
    if data_storage_location_keep_each_loop_pm != None:
        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_detector_cost_pm = last_row_explored["Detector_Cost"].iloc[0]
        else:
            current_detector_cost_pm = min_det_cost
            dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_detector_cost_pm = min_det_cost

    if data_store_place != None:
        if not os.path.isfile(data_store_place + '.csv'):
            current_detector_cost_pm = min_det_cost
            dictionary_fieldnames = ["Average Connectivity","detector_cost_at_zero"]
            with open(data_store_place + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    objective_list_pm = {}
    for detector_cost in np.arange(current_detector_cost_pm, max_det_cost + 0.05, 0.05):
        for id in key_dict_pm_hot.keys():
            try:
                cost = get_cost_for_prep_measure_mult_devs(key_dict_hot=key_dict_pm_hot[id],
                                                           key_dict_cold=key_dict_pm_cold[id],
                                                           cost_devs_hot=detector_cost + 1,
                                                           cost_devs_cold=detector_cold_cost * detector_cost + 1,
                                                           c_ij=cij)
                if cost != None:
                    if data_storage_location_keep_each_loop_pm != None:
                        dictionary = [
                            {"Detector_Cost": detector_cost, "Graph_ID": id, "objective_value": cost}]
                        dictionary_fieldnames = ["Detector_Cost", "Graph_ID", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop_pm + '.csv'):
                            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop_pm + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
            except:
                continue




    objective_list = {}
    for detector_cost in np.arange(current_detector_cost, max_det_cost + 0.05, 0.05):
        for id in key_dict_hot.keys():
            if current_node != None:
                if id != current_node:
                    continue
                else:
                    current_node = None
                    continue
            try:
                prob = cplex.Cplex()
                optimisation = Entanglement_Optimisation_Multiple_Dev_Types(prob, required_connections_hot[id], key_dict_hot[id], key_dict_cold[id])
                sol_dict, prob = optimisation.initial_optimisation_cost_reduction(cij, time_limit=1e5, Lambda = Lambda, c_esource = 1, c_edet = detector_cost, c_edet_cold=detector_cold_cost * detector_cost)
                objective_value = prob.solution.get_objective_value()
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"Detector_Cost": detector_cost,"Graph_ID": id, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Detector_Cost","Graph_ID", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)

            except:
                continue


        if data_storage_location_keep_each_loop != None:
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            for index, row in plot_information.iterrows():
                if round(row["Detector_Cost"],2) not in objective_list.keys():
                    objective_list[round(row["Detector_Cost"],2)] = {row["Graph_ID"]: row["objective_value"]}
                elif row["Graph_ID"] not in objective_list[round(row["Detector_Cost"],2)].keys():
                    objective_list[round(row["Detector_Cost"],2)][row["Graph_ID"]] = row["objective_value"]
                else:
                    objective_list[round(row["Detector_Cost"],2)][row["Graph_ID"]] = row["objective_value"]
        if data_storage_location_keep_each_loop_pm != None:
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
            for index, row in plot_information.iterrows():
                if round(row["Detector_Cost"],2) not in objective_list_pm.keys():
                    objective_list_pm[round(row["Detector_Cost"],2)] = {row["Graph_ID"]: row["objective_value"]}
                elif row["Graph_ID"] not in objective_list_pm[round(row["Detector_Cost"],2)].keys():
                    objective_list_pm[round(row["Detector_Cost"],2)][row["Graph_ID"]] =  row["objective_value"]
                else:
                    objective_list_pm[round(row["Detector_Cost"],2)][row["Graph_ID"]] = row["objective_value"]
        objective_values = {}
        for det_cost in objective_list.keys():
            for key in objective_list[det_cost].keys():
                if det_cost not in objective_values.keys() and key in objective_list_pm[det_cost].keys():
                    objective_values[det_cost] = [(objective_list[det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key]]
                elif key in objective_list_pm[det_cost].keys():
                    objective_values[det_cost].append((objective_list[det_cost][key] - objective_list_pm[det_cost][key] )/ objective_list_pm[det_cost][key])
        mean_objectives = {}
        std_objectives = {}
        for key in objective_values.keys():
            mean_objectives[key] = np.mean(objective_values[key])
            std_objectives[key] = np.std(objective_values[key])
            if mean_objectives[key] -std_objectives[key] > 0.0:
                if data_store_place != None:
                    dictionary = [
                        {"Average Connectivity": current_connectivity,"detector_cost_at_zero": key}]
                    dictionary_fieldnames = ["Average Connectivity", "detector_cost_at_zero"]
                    if os.path.isfile(data_store_place + '.csv'):
                        with open(data_store_place + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_store_place + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
                print("Detector Cost: " + str(key))
                return 0


def get_where_mean_is_zero(Lambda = 8, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_pm = None, data_store_place = None, current_connectivity = 3.5):
    objective_list = {}
    objective_list_pm = {}
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list.keys():
                objective_list[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list[row["Detector_Cost"]].keys():
                objective_list[row["Detector_Cost"]][row["Graph_ID"]] =  row["objective_value"]
            else:
                objective_list[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    if data_storage_location_keep_each_loop_pm != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_pm + ".csv")
        for index, row in plot_information.iterrows():
            if row["Detector_Cost"] not in objective_list_pm.keys():
                objective_list_pm[row["Detector_Cost"]] = {row["Graph_ID"]: row["objective_value"]}
            elif row["Graph_ID"] not in objective_list_pm[row["Detector_Cost"]].keys():
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
            else:
                objective_list_pm[row["Detector_Cost"]][row["Graph_ID"]] = row["objective_value"]
    objective_values = {}
    for det_cost in objective_list.keys():
        for key in objective_list[det_cost].keys():

            if det_cost not in objective_values.keys() and key in objective_list_pm[det_cost].keys():
                objective_values[det_cost] = [
                    (objective_list[det_cost][key] - objective_list_pm[det_cost][key]) / objective_list_pm[det_cost][
                        key]]
            elif key in objective_list_pm[det_cost].keys():
                objective_values[det_cost].append(
                    (objective_list[det_cost][key] - objective_list_pm[det_cost][key]) / objective_list_pm[det_cost][
                        key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
        if mean_objectives[key]  > 0.0:
            if data_store_place != None:
                dictionary = [
                    {"Average Connectivity": current_connectivity, "detector_cost_at_zero": key}]
                dictionary_fieldnames = ["Average Connectivity", "detector_cost_at_zero"]
                if os.path.isfile(data_store_place + '.csv'):
                    with open(data_store_place + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_store_place + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)
            print("Detector Cost: " + str(key))
            return 0


def calculate_average_key_rates(ent_data_file_hot, pm_data_file_hot, ent_data_file_cold, pm_data_file_cold, conn):
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    for id in key_dict_pm_hot.keys():
        print(f"Hot Ent Rates conn {conn}, id {id}: " + str(round(np.mean(list(key_dict_hot[id].values())), 4)))
        print(f"Cold Ent Rates conn {conn}, id {id}: " + str(round(np.mean(list(key_dict_cold[id].values())), 4)))
        print(f"Hot PM Rates conn {conn}, id {id}: " + str(round(np.mean(list(key_dict_pm_hot[id].values())), 4)))
        print(f"Cold PM Rates conn {conn}, id {id}: " + str(round(np.mean(list(key_dict_pm_cold[id].values())), 4)))

def number_con(key_dict_pm_hot):
    i = 0
    for key in key_dict_pm_hot.keys():
        if key_dict_pm_hot[key] < 10:
            i += 1
    return i

def number_con_ent(key_dict_hot):
    i = 0
    is_lower = []
    for key in key_dict_hot.keys():
        if key[0] < key[1]:
            if (key[0],key[1]) not in is_lower:
                is_lower.append((key[0],key[1]))
            else:
                is_lower.append((key[1],key[0]))
    for key in key_dict_hot.keys():
        if key_dict_hot[key] > 10:
            if key[0] < key[1] and (key[0], key[1]) in is_lower:
                is_lower.remove((key[0], key[1]))
            elif (key[1],key[0]) in is_lower:
                is_lower.remove((key[1], key[0]))
    return len(is_lower)


def calculate_min_key_rates(ent_data_file_hot, pm_data_file_hot, ent_data_file_cold, pm_data_file_cold, conn):
    key_dict_hot, required_connections_hot = import_data_ent(data_file_location=ent_data_file_hot)
    # key_dict_pm_hot = import_data_pm(data_file_location=pm_data_file_hot)
    # key_dict_cold, required_connections_cold = import_data_ent(data_file_location=ent_data_file_cold)
    key_dict_pm_cold = import_data_pm(data_file_location=pm_data_file_cold)
    for id in key_dict_hot.keys():
        if round(min(list(key_dict_pm_cold[id].values())), 4) > 0.00001:
            # print(f"Hot Ent Rates conn {conn}, id {id}: " + str(round(min(list(key_dict_hot[id].values())), 4)))
            # print(f"Cold Ent Rates conn {conn}, id {id}: " + str(round(min(list(key_dict_cold[id].values())), 4)))
            i = number_con_ent(key_dict_hot[id])
            print(f"Number Hot Edges less than 10 conn {conn}, id {id}: " + str(i))

            # print(f"Hot PM Rates conn {conn}, id {id}: " + str(round(min(list(key_dict_pm_hot[id].values())), 4)))
            # print(f"Cold PM Rates conn {conn}, id {id}: " + str(round(min(list(key_dict_pm_cold[id].values())), 4)))


if __name__ == "__main__":
    # get_where_mean_is_zero(data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_10", data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_10",
    #                        data_store_place="results_mesh_topologies_cross_zero", current_connectivity=10)
    #
    #
    # get_where_mean_is_zero(data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_15", data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_15",
    #                        data_store_place="results_mesh_topologies_cross_zero", current_connectivity=15)
    #
    # get_where_mean_is_zero(
    #     data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_10",
    #     data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_10",
    #     data_store_place="results_mesh_topologies_cross_zero_cmin_1000", current_connectivity=10)
    #
    # get_where_mean_is_zero(
    #     data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_15",
    #     data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_15",
    #     data_store_place="results_mesh_topologies_cross_zero_cmin_1000", current_connectivity=15)
    # calculate_min_key_rates(ent_data_file_hot = "mesh_topology_entanglement_hot_rates_mesh_3.csv", pm_data_file_hot =  "mesh_topology_pm_hot_rates_mesh_3.csv",
    #                                    ent_data_file_cold = "mesh_topology_entanglement_cold_rates_mesh_3.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates_mesh_3.csv", conn = 3)
    #
    # calculate_min_key_rates(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_10.csv",
    #                             pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_10.csv",
    #                             ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_10.csv",
    #                             pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_10.csv", conn = 10)

    # get_where_mean_is_zero(
    #     data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_5",
    #     data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_5",
    #     data_store_place="results_mesh_topologies_cross_zero_std_cmin_1000", current_connectivity=5)
    #
    # get_where_mean_is_zero(
    #     data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_6",
    #     data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_6",
    #     data_store_place="results_mesh_topologies_cross_zero_std_cmin_1000", current_connectivity=6)
    #
    #     data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_8",
    #     data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_8",
    #     data_store_place="results_mesh_topologies_cross_zero_std_cmin_1000", current_connectivity=8)

    # compare_for_different_detector_costs(ent_data_file_hot = "mesh_topology_entanglement_hot_rates.csv", pm_data_file_hot = "mesh_topology_pm_hot_rates.csv",
    #                                      ent_data_file_cold = "mesh_topology_entanglement_cold_rates.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates.csv",
    #                                      Lambda = 1, cij = 10, min_det_cost = 0.05, max_det_cost= 5, detector_cold_cost = 3,
    #                                      data_storage_location_keep_each_loop="results_mesh_trial_entanglement",
    #                                      data_storage_location_keep_each_loop_pm="results_mesh_trial_pm_2")
    #
    # compare_for_different_detector_costs_and_lambda(ent_data_file_hot = "mesh_topology_entanglement_hot_rates.csv", pm_data_file_hot = "mesh_topology_pm_hot_rates.csv",
    #                                      ent_data_file_cold = "mesh_topology_entanglement_cold_rates.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates.csv",
    #                                                 Lambda_max = 8, cij = 1000, min_det_cost = 0.05, max_det_cost = 3,
    #                                                 detector_cold_cost = 3, data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement",
    #                                                 data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_zero_pm_source_cost")

    # compare_for_different_source_costs_and_lambda(pm_data_file_hot = "mesh_topology_pm_hot_rates.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates.csv",
    #                                               cij =10, min_source_cost = 0.0,
    #                                               max_source_cost = 2, detector_cost = 1, detector_cold_cost = 3,
    #                                               data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement",
    #                                               data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_source_cost_var_det_cost_1")

    compare_for_different_source_costs_and_lambda(pm_data_file_hot = "mesh_topology_pm_hot_rates.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates.csv",
                                                  cij =1000, min_source_cost = 0.0,
                                                  max_source_cost = 2, min_detector_cost = 0.05, max_detector_cost = 3,
                                                  detector_cold_cost = 3, data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement",
                                                  data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_3d_source_and_det_var")


    # get_values_at_which_ent_equals_pm(ent_data_file_hot = "mesh_topology_entanglement_hot_rates_mesh_3.csv", pm_data_file_hot =  "mesh_topology_pm_hot_rates_mesh_3.csv",
    #                                   ent_data_file_cold = "mesh_topology_entanglement_cold_rates_mesh_3.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates_mesh_3.csv",
    #                                   Lambda = 8, cij = 10, min_det_cost = 0.05, max_det_cost = 3, detector_cold_cost = 3,
    #                                   data_storage_location_keep_each_loop = "results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_3",
    #                                     data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_3", data_store_place= "results_mesh_topologies_cross_zero_std_2_cmin_10",
    #                                   current_connectivity=3)
    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_10.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_10.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_10.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_10.csv",
    #                                   Lambda=8, cij=10, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_10",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_10",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2",
    #                                   current_connectivity=10)
    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_15.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_15.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_15.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_15.csv",
    #                                   Lambda=8, cij=10, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_15",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_15",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2",
    #                                   current_connectivity=15)
    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_10.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_10.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_10.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_10.csv",
    #                                   Lambda=8, cij=1000, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_10",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_10",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2_cmin_1000",
    #                                   current_connectivity=10)
    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_15.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_15.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_15.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_15.csv",
    #                                   Lambda=8, cij=1000, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_1000_entanglement_mesh_15",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_1000_pm_mesh_15",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2_cmin_1000",
    #                                   current_connectivity=15)


    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_5.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_5.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_5.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_5.csv",
    #                                   Lambda=8, cij=10, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_5",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_5",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2",
    #                                   current_connectivity=5)
    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_6.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_6.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_6.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_6.csv",
    #                                   Lambda=8, cij=10, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_6",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_6",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2",
    #                                   current_connectivity=6)
    #
    # get_values_at_which_ent_equals_pm(ent_data_file_hot="mesh_topology_entanglement_hot_rates_mesh_8.csv",
    #                                   pm_data_file_hot="mesh_topology_pm_hot_rates_mesh_8.csv",
    #                                   ent_data_file_cold="mesh_topology_entanglement_cold_rates_mesh_8.csv",
    #                                   pm_data_file_cold="mesh_topology_pm_cold_rates_mesh_8.csv",
    #                                   Lambda=8, cij=10, min_det_cost=0.05, max_det_cost=1.5, detector_cold_cost=3,
    #                                   data_storage_location_keep_each_loop="results_mesh_lambda_and_det_cost_at_cmin_10_entanglement_mesh_8",
    #                                   data_storage_location_keep_each_loop_pm="results_mesh_lambda_and_det_cost_at_cmin_10_pm_mesh_8",
    #                                   data_store_place="results_mesh_topologies_cross_zero_std_2",
    #                                   current_connectivity=8)

    #     # compare_switch_for_different_detector_costs_lambda(
    #     #     ent_data_file_hot="test.csv",
    #     #     pm_data_file_hot="test_pm.csv",
    #     #     ent_data_file_cold="test.csv",
    #     pm_data_file_cold="test_pm.csv",
    #     Lambda_max=8, cij=1000, fswitch=0.1, min_det_cost=0.05,
    #     max_det_cost=3, detector_cold_cost=3,
    #     data_storage_location_keep_each_loop="test_ent_2",
    #     data_storage_location_keep_each_loop_pm="test_pm_2")
    # compare_switch_for_different_detector_costs_lambda(ent_data_file_hot = "mesh_topology_entanglement_hot_rates_switch.csv", pm_data_file_hot = "mesh_topology_pm_hot_rates_switch.csv",
    #                                      ent_data_file_cold = "mesh_topology_entanglement_cold_rates_switch.csv", pm_data_file_cold = "mesh_topology_pm_cold_rates_switch.csv",
    #                                                    Lambda_max = 8, cij = 10, fswitch = 0.1, min_det_cost = 0.05,
    #                                                    max_det_cost = 3, detector_cold_cost = 3,
    #                                                    data_storage_location_keep_each_loop="results_switched_mesh_lambda_and_det_cost_at_cmin_10_entanglement",
    #                                                    data_storage_location_keep_each_loop_pm="results_switched_mesh_lambda_and_det_cost_at_cmin_10_pm")
    # compare_switch_for_different_detector_costs(ent_data_file_hot, pm_data_file_hot, ent_data_file_cold,
    #                                             pm_data_file_cold, Lambda = 1, cij = 10, fswitch =0.1, min_det_cost = 0.05, max_det_cost = 5,
    #                                             detector_cold_cost = 3, data_storage_location_keep_each_loop=None,
    #                                             data_storage_location_keep_each_loop_pm=None)
