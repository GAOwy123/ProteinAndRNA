import OneHot
import pandas as pd
import numpy as np
from keras.utils import to_categorical


def data_processing(file, file01, file02, seed=0, test_frac=0.1):
    # data processing
    data = pd.read_table(file, header=None)
    data01 = pd.read_table(file01, header=None)
    data02 = pd.read_table(file02, header=None)
    # get sequence length
    a = max(list(map(len, data[3])))
    b = max(list(map(len, data01[data01[8] == 0][3])))
    c = max(list(map(len, data02[data02[8] == 0][3])))
    seq_len = max([a, b, c])

    # positive data
    pos_seq = pd.DataFrame({"seq": [x.center(seq_len, "N").upper() for x in data[3]]})
    # pos_y = pd.DataFrame({"y": [1]*pos_seq.shape[0]})
    pos_data = pd.DataFrame({"seq": [x.center(seq_len, "N").upper() for x in data[3]], "y": [1]*pos_seq.shape[0]})
    # negtive data
    if pos_data.shape[0] > data01[data01[8] == 0].append(data02[data02[8] == 0]).shape[0]:
        neg_data = data01[data01[8] == 0].append(data02[data02[8] == 0]).sample(n=pos_data.shape[0], replace=True,
                                                                                random_state=seed)
        neg_data = pd.DataFrame({"seq": [x.center(seq_len, "N").upper() for x in neg_data[3]], "y": [0]*pos_seq.shape[0]})
    else:
        neg_data = data01[data01[8] == 0].append(data02[data02[8] == 0]).sample(n=pos_data.shape[0], replace=False,
                                                                                random_state=seed)
        neg_data = pd.DataFrame(
            {"seq": [x.center(seq_len, "N").upper() for x in neg_data[3]], "y": [0] * pos_seq.shape[0]})
    data = pos_data.append(neg_data)
    data.index = range(data.shape[0])

    test_data = data.sample(frac=test_frac, replace=False,  random_state=seed)
    train_validation_index = data.index.difference(test_data.index)
    train_validation_data = data.iloc[train_validation_index]

    test_x = []
    for seq in test_data['seq']:
        test_x.append(OneHot.one_hot(seq))

    train_validation_x = []
    for seq in train_validation_data['seq']:
        train_validation_x.append(OneHot.one_hot(seq))

    train_validation_x = np.array(train_validation_x)
    test_x = np.array(test_x)

    test_x = [np.transpose(mat) for mat in test_x]  # 4 215 -->  215 4  转置矩阵
    test_x = list(map(lambda x: x.reshape(test_x[0].shape[0], test_x[0].shape[1], 1), test_x))
    test_x = np.array(test_x)  # test_x.shape (2088, 215, 4, 1)

    train_validation_x = [np.transpose(mat) for mat in train_validation_x]
    train_validation_x = list(map(lambda x: x.reshape(test_x[0].shape[0], test_x[0].shape[1], 1), train_validation_x))
    train_validation_x = np.array(train_validation_x)  # train_voladation_x.shape (51492, 215, 4, 1)

    test_y = test_data['y']
    test_y = np.array(test_y)
    train_validation_y = train_validation_data['y']
    train_validation_y = np.array(train_validation_y)

    train_validation_y_encoded = to_categorical(train_validation_y)
    test_y_encoded = to_categorical(test_y)
    # test_y_encoded = np.zeros([len(test_y), 2])
    # for i in range(len(test_y)):
    #         test_y_encoded[i, test_y[i]] = 1
    #
    # train_validation_y_encoded = np.zeros([len(train_validation_y), 2])
    # for i in range(len(train_validation_y)):
    #         train_validation_y_encoded[i, train_validation_y[i]] = 1
    return seq_len, train_validation_x, train_validation_y, train_validation_y_encoded, test_x, test_y, test_y_encoded


