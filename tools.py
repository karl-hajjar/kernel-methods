import csv
import numpy as np


def submitResults(filename, y_pred_final):
    y = np.concatenate([y_pred_final[i] for i in [0,1,2]])
    with open("{}.csv".format(filename), 'w') as f:
        string = "Id,Bound\n"
        for j in range(0,3000):
            string += str(j)+','+str(y[j][0])+'\n'
        f.write(string)
    print("----------------- Prediction written on {}.csv ---------------".format(filename))
