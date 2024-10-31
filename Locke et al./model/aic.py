from .environment import Environment
import numpy as np
from .data import Data
from tqdm import tqdm

# finds the aic value for the given parameters given the data
class Aic:
    # calculates the log likelihood for choice/accuracy predictions
    @staticmethod
    def calc_ll_acc(sigzs, sdzs, trials_sub):
        ll_avg = 0
        ll_subs = np.zeros(len(trials_sub.keys()))
        for subject in tqdm(trials_sub, desc="calculating acc aic"):
            trials_split = Data.split_session_direction(trials_sub[subject])
            ll_sub = 0
            for session in trials_split.values():
                # compares accuracy on average
                if len(session) != 0:
                    pred_acc = Environment.get_pred_acc(session, sigzs[int(subject) - 1], sdzs[int(subject) - 1])
                    real_acc = Environment.get_real_acc(session)

                    if pred_acc == 1:
                        pred_acc = 1 - (10 ** -10)
                    if pred_acc == 0:
                        pred_acc = 0 + (10 ** -10)
                    ll_temp = len(session) * (real_acc * np.log(pred_acc) + (1 - real_acc) * (np.log(1 - pred_acc)))
                    ll_avg += ll_temp
                    ll_sub += ll_temp
            ll_subs[int(subject) - 1] = ll_sub
        return ll_avg, ll_subs

    # calculates the log likelihood for choice/accuracy predictions w/ bias
    @staticmethod
    def calc_ll_acc_bias(sigzs, sdzs, biases, trials_sub):
        ll_avg = 0
        ll_subs = np.zeros(len(trials_sub.keys()))
        for subject in tqdm(trials_sub, desc="calculating acc aic"):
            trials_split = Data.split_session_direction(trials_sub[subject])
            ll_sub = 0
            for session in trials_split.values():
                # compares accuracy on average
                if len(session) != 0:
                    pred_acc = Environment.get_pred_acc(session, sigzs[int(subject) - 1], sdzs[int(subject) - 1],
                                                             biases[int(subject) - 1])
                    real_acc = Environment.get_real_acc(session)

                    if pred_acc == 1:
                        pred_acc = 1 - (10 ** -10)
                    if pred_acc == 0:
                        pred_acc = 0 + (10 ** -10)
                    ll_temp = len(session) * (real_acc * np.log(pred_acc) + (1 - real_acc) * (np.log(1 - pred_acc)))
                    ll_avg += ll_temp
                    ll_sub += ll_temp
            ll_subs[int(subject) - 1] = ll_sub
        return ll_avg, ll_subs

    # calculates the log likelihood for confidence predictions
    @staticmethod
    def calc_ll_conf(sigzs, sdzs, conf_cutoffs, betas, trials_sub, conf_type):
        ll_avg = 0
        ll_subs = np.zeros(len(trials_sub.keys()))
        for subject in trials_sub:
            trials_split = Data.split_session(trials_sub[subject])
            ll_sub = 0
            for cond in [3,4]: # ONLY TESTING ON THE VALUE TRIALS
                session = trials_split[int(cond)]
                # compares accuracy on average
                if len(session) != 0:
                    if conf_type == 0:
                        pred_conf_rate = Environment.get_pred_conf(session, conf_cutoffs[int(subject) - 1], sigzs[int(subject) - 1], sdzs[int(subject) - 1], betas[int(subject) - 1])
                    else:
                        pred_conf_rate = Environment.get_pred_conf_other(session, conf_cutoffs[int(subject) - 1], sigzs[int(subject) - 1], sdzs[int(subject) - 1], conf_type)
                    real_conf_rate = Environment.get_real_conf(session)
                    if pred_conf_rate == 1:
                        pred_conf_rate = 1 - (10 ** -10)
                    if pred_conf_rate == 0:
                        pred_conf_rate = 0 + (10 ** -10)

                    ll_temp = len(session) * (real_conf_rate * np.log(pred_conf_rate) +
                                              (1 - real_conf_rate) * (np.log(1 - pred_conf_rate)))
                    ll_sub += ll_temp
                    ll_avg += ll_temp
            ll_subs[int(subject) - 1] = ll_sub
        return ll_avg, ll_subs

    # calculates the log likelihood for confidence predictions w/ bias
    @staticmethod
    def calc_ll_conf_bias(sigzs, sdzs, conf_cutoffs, betas, biases, trials_sub, conf_type):
        ll_avg = 0
        ll_subs = np.zeros(len(trials_sub.keys()))
        for subject in trials_sub:
            trials_split = Data.split_session(trials_sub[subject])
            ll_sub = 0
            for cond in [3, 4]:  # ONLY TESTING ON THE VALUE TRIALS
                session = trials_split[int(cond)]
                # compares accuracy on average
                if len(session) != 0:
                    if conf_type == 0:
                        pred_conf_rate = Environment.get_pred_conf(session, conf_cutoffs[int(subject) - 1],
                                                                   sigzs[int(subject) - 1], sdzs[int(subject) - 1],
                                                                   betas[int(subject) - 1], biases[int(subject) - 1])
                    else:
                        pred_conf_rate = Environment.get_pred_conf_other(session, conf_cutoffs[int(subject) - 1],
                                                                         sigzs[int(subject) - 1],
                                                                         sdzs[int(subject) - 1], conf_type, biases[int(subject) - 1])
                    real_conf_rate = Environment.get_real_conf(session)
                    if pred_conf_rate == 1:
                        pred_conf_rate = 1 - (10 ** -10)
                    if pred_conf_rate == 0:
                        pred_conf_rate = 0 + (10 ** -10)

                    ll_temp = len(session) * (real_conf_rate * np.log(pred_conf_rate) +
                                              (1 - real_conf_rate) * (np.log(1 - pred_conf_rate)))
                    ll_sub += ll_temp
                    ll_avg += ll_temp
            ll_subs[int(subject) - 1] = ll_sub
        return ll_avg, ll_subs

    # calculates the aic for choice/accuracy predictions
    @staticmethod
    def calc_aic_acc(num_params, sigzs, sdzs, trials_sub):
        ll_avg, ll_subs = Aic.calc_ll_acc(sigzs, sdzs, trials_sub)
        aic_subs = []
        for ll in ll_subs:
            aic_subs.append(2 * (num_params - ll))
        return 2 * (num_params - ll_avg), aic_subs

    # calculates the aic for choice/accuracy predictions w/ bias
    @staticmethod
    def calc_aic_acc_bias(num_params, sigzs, sdzs, biases, trials_sub):
        ll_avg, ll_subs = Aic.calc_ll_acc_bias(sigzs, sdzs, biases, trials_sub)
        aic_subs = []
        for ll in ll_subs:
            aic_subs.append(2 * (num_params - ll))
        return 2 * (num_params - ll_avg), aic_subs

    # calculates the aic for confidence predictions
    @staticmethod
    def calc_aic_conf(num_params, sigzs, sdzs, conf_cutoffs, betas, trials_sub, conf_type):
        ll_avg, ll_subs = Aic.calc_ll_conf(sigzs, sdzs, conf_cutoffs, betas, trials_sub, conf_type)
        aic_subs = []
        for ll in ll_subs:
            aic_subs.append(2 * (num_params - ll))
        return (2 * (num_params - ll_avg)), aic_subs

    # calculates the aic for confidence predictions w/ bias
    @staticmethod
    def calc_aic_conf_bias(num_params, sigzs, sdzs, conf_cutoffs, betas, biases, trials_sub, conf_type):
        ll_avg, ll_subs = Aic.calc_ll_conf_bias(sigzs, sdzs, conf_cutoffs, betas, biases, trials_sub, conf_type)
        aic_subs = []
        for ll in ll_subs:
            aic_subs.append(2 * (num_params - ll))
        return (2 * (num_params - ll_avg)), aic_subs
