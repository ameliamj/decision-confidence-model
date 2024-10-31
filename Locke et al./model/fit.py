import numpy as np
from scipy.stats import norm

# fits the observational noise (sdz)
# train_trails: list of trials to train on
# start_sd: the starting guess for the observational noise
# iters: number of iterations to estimate the observational noise
class Fit:
    def __init__(self, train_trials, start_val, iters):
        self.train_trials = train_trials
        self.start_val = start_val
        self.iters = iters

    # returns an estimation of the observational noise (here the start sd)
    def fit_sdz(self):
        return self.start_val

# fits the observational noise (sdz) analytically (this is only feasible for trials with
# a uniform distribution of prior probability and value over choices)
# train_trails: list of trials to train on
# start_sd: the starting guess for the observational noise
# iters: number of iterations to estimate the observational noise
# alpha: the learning rate
class AnalyticalFit(Fit):
    def __init__(self, train_trials, start_sd, iters, alpha):
        super().__init__(train_trials, start_sd, iters)
        self.alpha = alpha

    # fits the observational noise by moving the sdz in the direction of the
    # gradient of the likelihood of the accuracy for a trial given an observation
    # for math/derivations, see notebook page w/ bottom sticky note
    def fit_sdz(self):
        sd = self.start_val
        alpha = self.alpha
        for i in range(0, self.iters):
            grad = 0
            for trial in self.train_trials:
                if ((trial.stimulus == trial.reaction and trial.stimulus == 1) or
                        (trial.stimulus != trial.reaction and trial.stimulus == -1)):  # correct right or incorrect left
                    grad += (1 / norm.cdf(trial.stimulus / sd)) * (norm.pdf(trial.stimulus / sd)) * (
                                -trial.stimulus * sd ** -2)
                else:  # incorrect right or correct left
                    grad += (1 / norm.cdf(-trial.stimulus / sd)) * (norm.pdf(-trial.stimulus / sd)) * (
                                trial.stimulus * sd ** -2)
            sd = sd + grad * alpha
            alpha = alpha * .99
            if grad * grad < 10 ** -4:
                break  # means it has converged
            if sd < 0.1:
                sd = 0.1
            if sd > 5:
                sd = 5
        return sd

# fits the observational noise (sdz) using hierarchical grid search
# train_trails: list of trials to train on
# start_param: the starting guess for the parameter we are fitting
# iters: number of iterations to estimate the parameter
class GridFit(Fit):
    def __init__(self, train_trials, start_val, iters, like, param):
        super().__init__(train_trials, start_val, iters)
        self.like = like
        self.param = param

    # fits the observational noise by doing a hierarchical grid search around the start sd
    # fits it the number of iterations times and then averages across these to TRY to get better fit
    def fit(self):
        new_param = self.start_val
        if self.param == 'sd':
            intervals = [2.5, 1, .5, .1]
            steps = [.25, .1, .05, .01]
        elif self.param == 'conf':
            intervals = [.5, .1, .05, .01]
            steps = [.1, .05, .01, 0.005]
        else:  # param == 'beta'
            intervals = [1, .1, .05, .01]
            steps = [.1, .05, .01, 0.005]
        for i in range(len(intervals)):
            low_param = new_param - intervals[i]
            high_param = new_param + intervals[i]
            if low_param < 0:
                low_param = .01
            pot_param = np.arange(low_param, high_param, steps[i])
            probs = np.empty(len(pot_param))
            for j, param in enumerate(pot_param):
                probs[j] = self.like.lnprob(param, self.train_trials)
            new_param = pot_param[np.argmax(probs)]
        return new_param

    def sdz_fit(self, sigz):
        new_param = self.start_val
        intervals = [2.5, 1, .5, .1]
        steps = [.25, .1, .05, .01]
        for i in range(len(intervals)):
            low_param = new_param - intervals[i]
            high_param = new_param + intervals[i]
            if low_param < 0:
                low_param = .01
            pot_param = np.arange(low_param, high_param, steps[i])
            probs = np.empty(len(pot_param))
            for j, param in enumerate(pot_param):
                probs[j] = self.like.lnprob(sigz, param, self.train_trials)
            new_param = pot_param[np.argmax(probs)]
        return new_param

    def bias_fit(self, sigz, sigz_sub):
        new_param = self.start_val
        intervals = [.5, .25, .125, .625]
        steps = [.1, .05, .01, 0.005]
        for i in range(len(intervals)):
            low_param = new_param - intervals[i]
            high_param = new_param + intervals[i]
            pot_param = np.arange(low_param, high_param, steps[i])
            probs = np.empty(len(pot_param))
            for j, param in enumerate(pot_param):
                probs[j] = self.like.lnprob(sigz, sigz_sub, param, self.train_trials)
            new_param = pot_param[np.argmax(probs)]
        return new_param
