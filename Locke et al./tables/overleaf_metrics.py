import numpy as np

if __name__ == '__main__':
    precision = 2
    high_confs = np.load("../results/other/ overall_high_conf.pkl", allow_pickle=True)
    right_biases = np.load("../results/other/ overall_right_bias.pkl", allow_pickle=True)

    table_str = "high confidence"
    for conf in high_confs:
        table_str += " & " + str(round(conf * 100, precision))
    table_str += " \\\ \n"

    table_str += "right bias"
    for right_bias in right_biases:
        table_str += " & " + str(round(conf * 100, precision))
    table_str += " \\\ \n"

    print(table_str)
