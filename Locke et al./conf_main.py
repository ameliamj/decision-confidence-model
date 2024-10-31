from model.data import Data
from model.fit import AnalyticalFit, GridFit
from model.likelihood import WeightedConf, SdzLike
import random
import numpy as np
from tqdm import tqdm
import time
from model.aic import Aic
from model.param import ConfParam
import pickle

# compares confidence fits of 4 different bayesian based models

# conf_cutoffs fit on prior trials:
    # conf_vales.txt
# or on a mixture of prior and symm trials:
    # conf_vales_combo.txt
# sigz fit on symm trials and sigz_sub fit on prior trials
if __name__ == '__main__':
    start = time.time()
    reps = 10

    # loads and cleans data
    data = Data('data/rawChoiceData.txt')
    trials_sub = data.split_subject(data.original_trials)
    num_sub = len(trials_sub.keys())

    # keeping track of observational noise and confidence criteria for each subject
    sigzs = np.empty((reps, num_sub))  # external noise
    sigz_subs = np.empty((reps, num_sub)) # internal noise

    conf_cutoffs1 = np.empty((reps, num_sub))  # planning as inference
    conf_cutoffs2 = np.empty((reps, num_sub))  # simple observations
    conf_cutoffs3 = np.empty((reps, num_sub)) # bayesian posterior
    conf_cutoffs4 = np.empty((reps, num_sub))  # normalized expected value
    conf_cutoffs5 = np.empty((reps, num_sub))
    betas = np.ones(num_sub)

    # starting values and options
    start_sd = 2.1  # average analytical fit for sd
    start_conf = .75  # works best for current grid search
    symm_iters = 300
    iters = 1000
    alpha = .1
    beta = 1
    num_trials_symm = 600
    num_trials = 1200

    aic_pai = np.zeros((reps, num_sub))
    aic_obsv = np.zeros((reps, num_sub))
    aic_bayes = np.zeros((reps, num_sub))
    aic_ev = np.zeros((reps, num_sub))
    run_nums = np.zeros(reps)

    for i in range(reps):
        run_num = random.randint(0, 1000)
        run_nums[i] = run_num
        random.seed(run_num)

        for subject in tqdm(trials_sub, desc="fitting for each subject"):
            split_trials = data.split_session(trials_sub[subject])
            symm_trials = split_trials[0]
            prior_trials = np.concatenate((split_trials[1], split_trials[2]))
            symm_prior = np.concatenate((symm_trials, prior_trials)) # fitting conf on symm and prior right now


            random.shuffle(trials_sub[subject])
            random.shuffle(prior_trials)
            random.shuffle(split_trials)
            random.shuffle(symm_prior)

            # sig_z fit analytically on SYMMETRIC trials, sdz fit through grid search on PRIOR trials
            sigz_fit = AnalyticalFit(symm_trials[:num_trials_symm], start_sd, symm_iters, alpha)
            sigz = sigz_fit.fit_sdz()
            sigzs[i, int(subject) - 1] = sigz

            sigz_sub_like = SdzLike()
            sigz_sub_fit = GridFit(prior_trials[:num_trials], sigz, iters, sigz_sub_like,
                                   'sd')  # start from sigz since the ideal observer uses the same value
            sigz_sub = sigz_sub_fit.sdz_fit(sigz)
            sigz_subs[i, int(subject) - 1] = sigz_sub

            """
            # fit confidence criterion with planning as inference
            conf1_type = 0
            conf1_like = WeightedConf(sigz, sigz_sub, beta, conf1_type)
            conf1_fit = GridFit(symm_prior[:num_trials], start_conf, iters, conf1_like,
                                'conf')
            conf1_cutoff = conf1_fit.fit()
            conf_cutoffs1[i, int(subject) - 1] = conf1_cutoff

            # fit confidence criterion with simple observations
            conf2_type = 1
            conf2_like = WeightedConf(sigz, sigz_sub, beta, conf2_type)
            conf2_fit = GridFit(symm_prior[:num_trials], start_conf, iters, conf2_like,
                                'conf')
            conf2_cutoff = conf2_fit.fit()
            conf_cutoffs2[i, int(subject) - 1] = conf2_cutoff

            # fit confidence criterion with bayesian confidence
            conf3_type = 2
            conf3_like = WeightedConf(sigz, sigz_sub, beta, conf3_type)
            conf3_fit = GridFit(symm_prior[:num_trials], start_conf, iters, conf3_like,
                                'conf')
            conf3_cutoff = conf3_fit.fit()
            conf_cutoffs3[i, int(subject) - 1] = conf3_cutoff

            # fit confidence criterion with normalized expected value
            conf4_type = 3
            conf4_like = WeightedConf(sigz, sigz_sub, beta, conf4_type)
            conf4_fit = GridFit(symm_prior[:num_trials], start_conf, iters, conf4_like,
                                'conf')
            conf4_cutoff = conf4_fit.fit()
            conf_cutoffs4[i, int(subject) - 1] = conf4_cutoff
            """

            # fit confidence criterion with normalized expected value
            conf5_type = 4
            conf5_like = WeightedConf(sigz, sigz_sub, beta, conf5_type)
            conf5_fit = GridFit(symm_prior[:num_trials], start_conf, iters, conf5_like,
                                'conf')
            conf5_cutoff = conf5_fit.fit()
            conf_cutoffs5[i, int(subject) - 1] = conf5_cutoff

        print(f'observational noise for each subject: {sigzs[i]}')
        print(f'internal observational noise for each subject: {sigz_subs[i]}')
        # print(f'aic for accuracy fit: {Aic.calc_aic_acc(2, sigzs, sdzs, trials_sub)}')
        """print(f'confidence cutoffs for each subject (pai): {conf_cutoffs1[i]}')
        print(f'confidence cutoffs for each subject (observations: {conf_cutoffs2[i]}')
        print(f'confidence cutoffs for each subject (bayesian): {conf_cutoffs3[i]}')
        print(f'confidence cutoffs for each subject (expected value): {conf_cutoffs4[i]}')"""
        print(f'confidence cutoffs for each subject (expected value): {conf_cutoffs5[i]}')

        num_params = 3

        """pai_aic, pai_aic_sub = Aic.calc_aic_conf(num_params, sigzs[i], sigz_subs[i], conf_cutoffs1[i], betas, trials_sub, conf_type=0)
        aic_pai[i, :] = pai_aic_sub
        print(f'total aic for pai conf fit: {pai_aic}')
        print(f'aics for pai conf fit: {pai_aic_sub}')

        obsv_aic, obsv_aic_sub = Aic.calc_aic_conf(num_params, sigzs[i], sigz_subs[i], conf_cutoffs2[i], betas, trials_sub, conf_type=1)
        aic_obsv[i, :] = obsv_aic_sub
        print(f'total aic for obsv conf fit: {obsv_aic}')
        print(f'aics for obsv conf fit: {obsv_aic_sub}')

        bayes_aic, bayes_aic_sub = Aic.calc_aic_conf(num_params, sigzs[i], sigz_subs[i], conf_cutoffs3[i], betas, trials_sub, conf_type=2)
        aic_bayes[i, :] = bayes_aic_sub
        print(f'total aic for bayesian conf fit: {bayes_aic}')
        print(f'aics for bayesian conf fit: {bayes_aic_sub}')

        ev_aic, ev_aic_sub = Aic.calc_aic_conf(num_params, sigzs[i], sigz_subs[i], conf_cutoffs4[i], betas, trials_sub, conf_type=3)
        aic_ev[i, :] = ev_aic_sub
        print(f'total aic for ev conf fit: {ev_aic}')
        print(f'aics for ev conf fit: {ev_aic_sub}')"""

        entropy_aic, entropy_aic_sub = Aic.calc_aic_conf(num_params, sigzs[i], sigz_subs[i], conf_cutoffs4[i], betas, trials_sub,
                                               conf_type=4)
        # aic_ev[i, :] = ev_aic_sub
        print(f'total aic for ev conf fit: {entropy_aic}')
        print(f'aics for ev conf fit: {entropy_aic_sub}')

    param = ConfParam(sigzs, sigz_subs, run_nums, conf_cutoffs1, conf_cutoffs2, conf_cutoffs3, conf_cutoffs4)

    with open('results/conf/params/conf_params_combo.pkl', 'wb') as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_aics = [aic_pai, aic_obsv, aic_bayes, aic_ev]
    aic_names = ['pai', 'obsv', 'bayes', 'ev']

    with open('results/conf/vales (pkl)/ conf_vales_combo.pkl', 'wb') as handle:
        pickle.dump(all_aics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for idx, aics in enumerate(all_aics):
        avg_aic = np.mean(aics, axis=0)
        max_aic = np.max(aics, axis=0)
        min_aic = np.min(aics, axis=0)
        total_aic = np.sum(avg_aic)

        diff = np.maximum(max_aic - avg_aic, avg_aic - min_aic)

        result_str = "["
        for jdx, aic in enumerate(avg_aic):
            result_str += (str(aic) + " +/- " + str(diff[jdx]) + " ")
        result_str += "]"
        with open('results/conf/vales (txt)/ conf_vales_combo.txt', 'a') as f:
                f.write(f'total {aic_names[idx]} aic over {reps} runs \n')
                f.write(str(total_aic))
                f.write(f'\naverage {aic_names[idx]} aic over {reps} runs \n')
                f.write(result_str)
                f.write('\n \n')

    end = time.time()
    print(f'total elapsed time: {(end - start) / 60}')




