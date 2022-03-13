
import numpy as np

def read_dataset(filename):
    with open(filename,'r') as file:
        lines = file.readlines()
    lines = np.array( [line.strip().split(',') for line in lines] , dtype = 'float64')
    return lines

if __name__ == '__main__':
    X_train = read_dataset("X_train.csv")   #5000 * 784
    X_test = read_dataset("X_test.csv")     #2500 * 784
    Y_train = read_dataset("Y_train.csv")   #5000 * 1
    Y_test = read_dataset("Y_test.csv")     #2500 * 1
