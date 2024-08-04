from gaussian_factor import GaussianFactor

class CameraSensor:
    # Create a CameraSensor active in specified area
    def __init__(self, area):
        # Values taken directly from datasets
        self.bias = -0.030364583333333334
        self.vars = 0.029439021584357853
        self.area = area

    def update(self, data):
        if data == None:
            self.count = None
            return
        self.count = data

    # Apply evidence on the room distributions given the currently stored evidence.
    # Takes in the current means and vars, and returns the new means and vars.
    def apply_evidence(self, means, vars):
        if self.count == None:
            return means, vars
        
        # Create a temporary mini-Bayesian network:
        # [ ppl_count ] -> [ obs_ppl_count ]
        #
        # We are seeking to find the updated distribution:
        # ppl_count | obs_ppl_count = self.count
        prior = GaussianFactor(('num_ppl',), mu=means[self.area], sigma=vars[self.area])
        camera = GaussianFactor(('obs_num_ppl', 'num_ppl',), beta=[1], b_mean=self.bias, b_var=self.vars)
        joint = prior * camera
        final = joint.evidence(obs_num_ppl=self.count)

        # Error checking - we should have eliminated all but 1 variable now, so mean and variance should be 1d
        # If this assertion passes, extract the sole element as a constant
        means[self.area] = (final.mean()).reshape(1)[0]
        vars[self.area] = (final.covariance()).reshape(1)[0]

        return means, vars