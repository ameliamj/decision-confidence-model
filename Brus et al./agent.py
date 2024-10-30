import numpy as np
import pandas as pd


class Agent:

    @staticmethod
    def pred_conf(data, conf_type, sampling=False, beta=1):
        if conf_type == 'pai':
            return Agent.pred_conf_pai(data, sampling, beta)
        else:  # conf_type == 'ev'
            return Agent.pred_conf_ev(data, sampling)

    # reward for correct is 1 and reward for incorrect is 0
    @staticmethod
    def pred_conf_pai(data, sampling=False, beta=1):
        r_max = 2
        r_correct = 1
        r_incorrect = 0
        avg_ratings = pd.concat([data['mrUp'], data['mrDown']], axis=1)
        pred_conf = np.exp(
            beta * avg_ratings.max(axis='columns') * (r_correct - r_max) + avg_ratings.min(axis='columns') * (
                        r_incorrect - r_max))
        pred_non_conf = np.exp(
            beta * avg_ratings.min(axis='columns') * (r_correct - r_max) + avg_ratings.max(
                axis='columns') * (r_incorrect - r_max))
        pred_conf = pred_conf / (pred_non_conf + pred_conf)

        # OPTIONAL
        if sampling:
            samples = np.random.random(size=data.shape[0])
            explore_inds = np.where(samples - pred_conf > 0)[0]
            pred_conf[explore_inds] = 1 - pred_conf[explore_inds]
        return pred_conf

    @staticmethod
    def pred_conf_ev(data, sampling=False):
        avg_ratings = pd.concat([data['mrUp'], data['mrDown']], axis=1)
        pred_conf = avg_ratings.max(axis='columns')
        pred_non_conf = avg_ratings.min(axis='columns')
        pred_conf = pred_conf / (pred_non_conf + pred_conf)

        # OPTIONAL
        if sampling:
            samples = np.random.random(size=data.shape[0])
            explore_inds = np.where(samples - pred_conf > 0)[0]
            pred_conf[explore_inds] = 1 - pred_conf[explore_inds]
        return pred_conf