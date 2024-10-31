from scipy.stats import norm
import numpy as np

# the agent makes predictions on the direction of the stimulus
# based off of observations
# these observations equate to the agents choices when the value distribution
# is uniform
# the agent has observational noise (sdz) and potentially a bias
class Agent:
    def __init__(self, sdz, bias=0):
        self.sdz = sdz
        self.num_stab = 10**-30
        self.bias = bias

    # predicts the stimulus direction based on the observations and the given priors
    # obsvs: a list of observations of the stimulus
    # priors: a list of prior distributions, one for every observation given
    # values: a list of value distributions, one for every observation given
    # returns a list of predictions, where 1 means a right prediction and -1 means a left prediction

    # helper method to get the expected values, posterior probabilities, and observation likelihoods for each trial
    def predict_choice_with_details(self, obsvs, priors, values, beta=1):
        priors[:, 0] += self.bias
        priors[:, 1] -= self.bias

        obs_ll = np.empty([len(obsvs), 2])
        obs_ll[:, 0] = norm.pdf(obsvs, 1, self.sdz)
        obs_ll[:, 1] = norm.pdf(obsvs, -1, self.sdz)
        posts = priors * obs_ll
        posts = posts / (10 ** -30 + posts.sum(axis=1)[:, None])

        expected_val = posts * values
        ### explore exploit
        r_correct = np.max(6 * values, axis=1)
        r_max = 5
        r_incorrect = 0
        conf = np.exp(
            np.longdouble(beta * (np.max(posts, axis=1) * (r_correct - r_max) + np.min(posts, axis=1) * (r_incorrect - r_max))))
        non_conf = np.exp(
            beta * (np.min(posts, axis=1) * (r_correct - r_max) + np.max(posts, axis=1) * (r_incorrect - r_max)))
        conf = conf / (conf + non_conf)
        explore_inds = []
        return explore_inds, expected_val, posts, obs_ll

    # returns the predicted choice for each observation as 1 for a right choice and -1 for a left choice
    def predict_choice(self, obsvs, priors, values):
        explore_inds, expected_val, posts, obs_ll = self.predict_choice_with_details(obsvs, priors, values)
        inds = np.argmax(expected_val, axis=1)  ## right -> 0; left -> 1, but we want right -> 1; left-> -1
        inds[explore_inds] = 1 - inds[
            explore_inds]  #### choose the non-max prob if sample is more than conf (since conf is higher than .5)
        return -(inds ** 2) + (1 - inds)  # (-x^2 + 1 - x) turns 0s to 1 and 1s to -1

    # returns the predicted confidence for each observation as a list of 0s (low confidence) and 1s (high confidence)
    # predicts the confidence using the planning as inference method
    def predict_conf(self, obsvs, priors, values, cutoff, beta_conf):
        explore_inds, expected_val, posts, obs_ll = self.predict_choice_with_details(obsvs, priors, values)

        r_correct = np.max(6 * values, axis=1)
        r_max = 5
        r_incorrect = 0
        conf = np.exp(
            beta_conf * (np.max(posts, axis=1) * (r_correct - r_max) + np.min(posts, axis=1) * (r_incorrect - r_max)))
        non_conf = np.exp(
            beta_conf * (np.min(posts, axis=1) * (r_correct - r_max) + np.max(posts, axis=1) * (r_incorrect - r_max)))
        conf = conf / (conf + non_conf)
        conf[explore_inds] = 1 - conf[
            explore_inds]  #### choose the non-max prob if sample is more than conf (since conf is higher than .5)

        preds_conf = np.zeros(len(obsvs))
        preds_conf[np.where(conf > cutoff)[0]] = 1
        return preds_conf

    # predicts confidence as the max of likelihood of observations
    def predict_conf_simple(self, obsvs, priors, values, cutoff):
        explore_inds, expected_val, posts, obs_ll = self.predict_choice_with_details(obsvs, priors, values)
        obs_ll = obs_ll / (obs_ll.sum(axis=1)[:, None] + self.num_stab)  # numerical stability added
        inds = np.argmax(expected_val, axis=1)
        inds[explore_inds] = 1 - inds[explore_inds]
        conf = -10 * np.ones(inds.size)
        for i, choice_index in enumerate(inds):
            conf[i] = obs_ll[i][choice_index]
        # conf = obs_ll[inds, inds]
        preds_conf = np.zeros(len(obsvs))
        preds_conf[np.where(conf > cutoff)[0]] = 1
        return preds_conf

    # predicts confidence as the max of the posterior probability
    def predict_conf_prior(self, obsvs, priors, values, cutoff):
        explore_inds, expected_val, posts, obs_ll = self.predict_choice_with_details(obsvs, priors, values)
        inds = np.argmax(expected_val, axis=1)
        inds[explore_inds] = 1 - inds[explore_inds]
        conf = -10 * np.ones(inds.size)
        for i, choice_index in enumerate(inds):
            conf[i] = posts[i][choice_index]
        # conf = posts[inds, inds]
        preds_conf = np.zeros(len(obsvs))
        preds_conf[np.where(conf > cutoff)[0]] = 1
        return preds_conf

    # predicts confidence as the entropy of the posterior probability
    def predict_conf_entropy(self, obsvs, priors, values, cutoff):
        explore_inds, expected_val, posts, obs_ll = self.predict_choice_with_details(obsvs, priors, values)
        inds = np.argmax(expected_val, axis=1)
        inds[explore_inds] = 1 - inds[explore_inds]
        conf = -10 * np.ones(inds.size)
        for i, choice_index in enumerate(inds):
            conf[i] = -(posts[i][choice_index] * np.log2(posts[i][choice_index] + self.num_stab) + ((1 - posts[i][choice_index]) * np.log2((1 - posts[i][choice_index]) + self.num_stab)))
        # conf = posts[inds, inds]
        # print(np.histogram(conf))
        preds_conf = np.zeros(len(obsvs))
        preds_conf[np.where(conf > cutoff)[0]] = 1
        return preds_conf

    # predicts confidence as the max of the posterior expected reward
    def predict_conf_value(self, obsvs, priors, values, cutoff):
        explore_inds, expected_val, posts, obs_ll = self.predict_choice_with_details(obsvs, priors, values)
        expected_val = expected_val / (expected_val.sum(axis=1)[:, None] + self.num_stab)  # numerical stability added
        conf = np.max(expected_val, axis=1)
        # print (explore_inds.size, expected_val.shape, conf.shape)
        conf[explore_inds] = 1 - conf[explore_inds]
        preds_conf = np.zeros(len(obsvs))
        preds_conf[np.where(conf > cutoff)[0]] = 1
        return preds_conf
