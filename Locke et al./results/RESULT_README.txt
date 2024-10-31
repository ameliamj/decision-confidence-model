Results description:
acc_vales: aic values for accuracy where the external noise (sigz) is fit on symmetric trials
and the internal noise (sigz_sub) is fit on prior trials

acc_ee_vales: acc_vales but explore/exploit is used

acc_bias_vales: acc_vales but bias is used and is fit on symmetric trials

acc_bias_combo_vales: acc_vales but bias is used and is fit on a mixture of symmetric and prior trials

conf_vales: aic values for confidence where the external noise (sigz) is fit on symmetric trials,
the internal noise (sigz_sub) is fit on prior trials, and conf_cutoffs are fit on prior trials

conf_combo_vales: conf_vales but conf_cutoffs are fit on a mixture of symmetric and prior trials

conf_bias_vales: conf_vales but bias is used and fit on symmetric trials

conf_bias_combo_vales: conf_combo_vales but bias is used and fit on a mixture of symmetric and prior trials

The parameters that are fit to produce the above results are saved in the corresponding params files.