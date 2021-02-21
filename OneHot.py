import numpy as np
import re


def one_hot(sequence):
    str_len = len(sequence)
    matrix = np.zeros((4, str_len))
    A_location = [x.start() for x in re.finditer('A', sequence)]
    T_location = [x.start() for x in re.finditer('T', sequence)]
    C_location = [x.start() for x in re.finditer('C', sequence)]
    G_location = [x.start() for x in re.finditer('G', sequence)]
    if A_location:
        matrix[0][A_location] = 1
    if T_location:
        matrix[1][T_location] = 1
    if C_location:
        matrix[2][C_location] = 1
    if G_location:
        matrix[3][G_location] = 1
    return matrix

