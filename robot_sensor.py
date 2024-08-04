import ast

from gaussian_factor import GaussianFactor

class RobotSensor:
    # Create a RobotSensor
    def __init__(self):
        # From data, it looks like robots are perfect??
        self.bias = 0
        self.vars = 0.0001
        self.idx_to_name = ['outside'] + ['r' + str(i) for i in range(1, 35)] + ['c1', 'c2']
        self.name_to_idx = {'outside': 0} | {'r' + str(i): i for i in range(1, 35)} | {'c1': 35, 'c2': 36}
    
    def update(self, data):
        if data == None:
            self.count = None
            return
        
        data = ast.literal_eval(data)
        self.room = self.name_to_idx[data[0]]
        self.count = data[1]

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars, t_m):
        if self.count == None:
            return means, vars
        # Create a temporary mini-Bayesian network:
        # [ ppl_count ] -> [ obs_ppl_count ]
        #
        # We are seeking to find the updated distribution:
        # ppl_count | obs_ppl_count = self.count
        prior = GaussianFactor(('num_ppl',), mu=means[self.room], sigma=vars[self.room])
        robot = GaussianFactor(('obs_num_ppl', 'num_ppl',), beta=[1], b_mean=self.bias, b_var=self.vars)
        joint = prior * robot
        final = joint.evidence(obs_num_ppl=self.count)

        means[self.room] = (final.mean()).reshape(1)[0]
        vars[self.room] = (final.covariance()).reshape(1)[0]

        return means, vars