import numpy as np

class Environment:
    @staticmethod
    def get_aic(real_conf, pred_conf, num_params=0):
        pred_conf = pred_conf.where(pred_conf != 1, 1 - (10 ** -10))
        pred_conf = pred_conf.where(pred_conf != 0, 0 + (10 ** -10))
        ll = (real_conf * np.log(pred_conf) + (1 - real_conf) * (np.log(1 - pred_conf)))
        ll = ll.sum()
        return 2 * (num_params - ll)

    @staticmethod
    def get_absolute_error(real_conf, pred_conf):
        denom = pred_conf.shape[0]
        return (np.sum(np.abs(pred_conf - real_conf))) / denom
