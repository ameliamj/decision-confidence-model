import numpy as np
from .environment import Environment
from .data import Data

# finds the likelihood of a given value for a parameter theta
# given the model and the data
class Likelihood:
    def lnlike(self, theta, trials):
        return 1

    # makes sure the given theta is within its prior limits (non-negative)
    # returns 0 if within limits, returns negative infinity if not
    def lnprior(self, theta):
        if theta >= 0:
            return 0.0
        else:
            return -np.inf

    # calculates the probability of a given theta given the model and the data
    # by combining the probabilities of the likelihood and prior
    def lnprob(self, theta, trials):
        lp = self.lnprior(theta)
        if lp == -np.inf:
            return -np.inf
        return lp + self.lnlike(theta, trials)

# finds the likelihood of a given value for the observational noise
# given the model and the data
class SdzLike(Likelihood):
    def lnlike(self, sigz, sigz_sub, trials):
        trials_split = Data.split_session_direction(trials)

        LnLike = 0
        for session in trials_split.values():
            # compares accuracy on average
            if len(session) != 0:
                pred_acc = Environment.get_pred_acc(session, sigz, sigz_sub)
                real_acc = Environment.get_real_acc(session)

                if pred_acc == 1:
                    pred_acc = 1 - (10 ** -10)
                if pred_acc == 0:
                    pred_acc = 0 + (10 ** -10)

                LnLike += len(session) * (real_acc * np.log(pred_acc) + (1 - real_acc) * (np.log(1 - pred_acc)))
        return LnLike

    # makes sure the given observational noise is within its prior limits (non-negative)
    # returns 0 if within limits, returns negative infinity if not
    def lnprior(self, sig):
        if sig >= 0:
            return 0.0
        else:
            return -np.inf

    # calculates the probability of a given observation noise given the model and the data
    # by combining the probabilities of the likelihood and prior
    def lnprob(self, sigz, sigz_sub, trials):
        lp = self.lnprior(sigz_sub)
        if lp == -np.inf:
            return -np.inf
        return lp + self.lnlike(sigz, sigz_sub, trials)

# finds the likelihood of a given value for the right bias
# given the model and the data
class BiasLike(Likelihood):
    def lnlike(self, sigz, sigz_sub, bias, trials):
        trials_split = Data.split_session_direction(trials)

        LnLike = 0
        for session in trials_split.values():
            # compares accuracy on average
            if len(session) != 0:
                # pred_acc = Environment.get_pred_acc_bias(session, sigz, sigz_sub, bias)
                pred_acc = Environment.get_pred_acc(session, sigz, sigz_sub, bias)
                real_acc = Environment.get_real_acc(session)

                if pred_acc == 1:
                    pred_acc = 1 - (10 ** -10)
                if pred_acc == 0:
                    pred_acc = 0 + (10 ** -10)

                LnLike += len(session) * (real_acc * np.log(pred_acc) + (1 - real_acc) * (np.log(1 - pred_acc)))
        return LnLike

    # makes sure the given bias is within its prior limits
    # returns 0 if within limits, returns negative infinity if not
    def lnprior(self, sig):
        return 0

    # calculates the probability of a given bias given the model and the data
    # by combining the probabilities of the likelihood and prior
    def lnprob(self, sigz, sigz_sub, bias, trials):
        lp = self.lnprior(sigz_sub)
        if lp == -np.inf:
            return -np.inf
        return lp + self.lnlike(sigz, sigz_sub, bias, trials)

# finds the likelihood of a given value for the confidence cutoff
# given the model and the data
class WeightedConf(Likelihood):
    def __init__(self, sigz, sdz, beta, conf_type, bias=0):
        self.sdz = sdz
        self.beta = beta
        self.conf_type = conf_type
        self.sigz = sigz
        self.bias = bias
        # conf types:
        # 0 --> planning as inference
        # 1 --> max probability of observation
        # 2 --> max posterior probability
        # 3 --> max posterior expected value

    def lnlike(self, conf_cutoff, train_trials):
        # gives us a dictionary with 14 keys (one for each session and direction combinations)
        trials_split = Data.split_session_direction(train_trials)

        LnLike = 0
        for session in trials_split.values():
            pred_conf_rate, real_conf_rate = self.conf_condition(conf_cutoff, session, self.sigz, self.sdz, self.beta, self.bias)
            if pred_conf_rate == 1:
                pred_conf_rate = pred_conf_rate - (10**-10)
            if pred_conf_rate == 0:
                pred_conf_rate = pred_conf_rate + (10 ** -10)
            LnLike += len(session) * ((real_conf_rate * np.log(pred_conf_rate)) + ((1-real_conf_rate) * np.log(1 - pred_conf_rate)))
        return LnLike

    # makes sure the given confidence cutoff is within its prior limits (between 0.1 and 1)
    # returns 0 if within limits, returns negative infinity if not
    def lnprior(self, theta):
        conf_cutoff = theta
        if .1 <= conf_cutoff <= 1:
            return 0.0
        else:
            return -np.inf

    # calculates the probability of a given confidence cutoff given the model and the data
    # by combining the probabilities of the likelihood and prior
    def lnprob(self, theta, trials):
        lp = self.lnprior(theta)
        if lp == -np.inf:
            return -np.inf
        return lp + self.lnlike(theta, trials)

    # gets the predicted rate of confidence and the real rate of confidence for different
    # confidence models
    def conf_condition(self, conf_cutoff, trials_cond, sigz, sdz, beta, bias):
        if self.conf_type == 0:
            pred_conf_rate = Environment.get_pred_conf(trials_cond, conf_cutoff, sigz, sdz, beta, bias)
        else:
            pred_conf_rate = Environment.get_pred_conf_other(trials_cond, conf_cutoff, sigz, sdz, self.conf_type, bias)
        real_conf_rate = Environment.get_real_conf(trials_cond)
        return pred_conf_rate, real_conf_rate

