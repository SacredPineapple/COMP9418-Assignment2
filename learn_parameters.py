from smart_building import SmartBuilding
from example_test import SmartBuildingSimulatorExample

simulator = SmartBuildingSimulatorExample()

# Gets the cost associated with a transition matrix, somehow?? W.I.P
def cost_t_matrix(simulator, transition_matrix):
    total_cost = 0
    for i in range(len(simulator.data)):
        sensor_data = simulator.timestep()
        actions_dict = get_action(sensor_data)   
        total_cost += simulator.cost_timestep(actions_dict)

