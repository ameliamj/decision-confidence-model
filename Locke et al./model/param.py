# used to store parameters that are fit during training so that they can be saved
# params needed to predict choices/accuracy of subjects
# external noise --> sigz; internal noise --> sigz_subs; random seed for shuffling data --> run_nums
class Param:
    def __init__(self, sigzs, sigz_subs, run_nums):
        self.sigzs = sigzs
        self.sigzs_subs = sigz_subs
        self.run_nums = run_nums

# params needed to predict choices/accuracy of subjects w/ bias
# degree of right bias --> biases
class BiasParam(Param):
    def __init__(self, sigzs, sigz_subs, run_nums, biases):
        super().__init__(sigzs, sigz_subs, run_nums)
        self.biases = biases

# params needed to predict confidence of subjects
# confidence cutoff for...:
#   planning as inference --> cc_pai
#   observation --> cc_obsv
#   perception --> cc_bayes
#   expected value --> cc_ev
class ConfParam(Param):
    def __init__(self, sigzs, sigz_subs, run_nums, cc_pai, cc_obsv, cc_bayes, cc_ev):
        super().__init__(sigzs, sigz_subs, run_nums)
        self.cc_pai = cc_pai
        self.cc_obsv = cc_obsv
        self.cc_bayes = cc_bayes
        self.cc_ev = cc_ev

# params needed to predict confidence of subjects w/ bias
class BiasConfParam(BiasParam):
    def __init__(self, sigzs, sigz_subs, run_nums, biases, cc_pai, cc_obsv, cc_bayes, cc_ev):
        super().__init__(sigzs, sigz_subs, run_nums, biases)
        self.cc_pai = cc_pai
        self.cc_obsv = cc_obsv
        self.cc_bayes = cc_bayes
        self.cc_ev = cc_ev