from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.initializers import he_normal
from keras.layers import Dense, Concatenate, Input
from keras.models import Model
from keras import regularizers
from AUC import auc


def create_classify_model(learning_rate=0.01, num_filter=16, filter_size=(24, 4),
                          input_shape=(215, 4, 1), strides=2, num_hidden1=300,
                          num_hidden2=2, padding='same', dropout=0.5,
                          pool_size=(2, 2),  # num_epoch=100,
                          activation1='relu', activation2='softmax',
                          weight_decay=0.0005,  # 权值衰减,目的是防止过拟合
                          momentum=0.9,
                          #decay=0.0
                          ):
    sgd = SGD(lr=learning_rate, decay=learning_rate/1000, momentum=momentum, nesterov=True)
    # lr: float >= 0.学习率。
    # decay: float >= 0.每次参数更新后学习率衰减值。
    # momentum: float >= 0.参数，用于加速SGD在相关方向上前进，并抑制震荡。
    # nesterov: boolean.是否使用Nesterov动量。
    model1_in = Input(shape=input_shape)
    # print(model1_in.shape)  # (?, 215, 4, 1)
    model1_out = Conv2D(filters=num_filter, kernel_size=filter_size, strides=strides, padding=padding,
                        input_shape=input_shape,
                        kernel_initializer=he_normal(seed=100),
                        bias_initializer=he_normal(seed=200),
                        kernel_regularizer=regularizers.l2(weight_decay))(model1_in)
    # print(model1_out.shape)  # (?, 108, 2, 16)
    model1_out = Activation(activation=activation1)(model1_out)
    # print(model1_out.shape)  # (?, 108, 2, 16)
    maxPooling_out = MaxPooling2D(pool_size=pool_size, strides=strides)(model1_out)
    # print(maxPooling_out.shape)  # (?, 54, 1, 16)
    avePooling_out = AveragePooling2D(pool_size=pool_size, strides=strides)(model1_out)
    # print(avePooling_out.shape)  # (?, 54, 1, 16)
    concatenated = Concatenate()([maxPooling_out, avePooling_out])
    model_out = BatchNormalization()(concatenated)

    model_out = Flatten()(model_out)

    model_out = Dense(num_hidden1, activation=activation1, kernel_regularizer=regularizers.l2(weight_decay))(model_out)
    # print(model_out.shape)  # (?, 300)
    model_out = Dropout(dropout)(model_out)
    # print(model_out.shape)  # (?, 300)
    model_out = Dense(num_hidden1, activation=activation1, kernel_regularizer=regularizers.l2(weight_decay))(model_out)
    # print(model_out.shape)
    model_out = Dense(num_hidden2, activation=activation2)(model_out)  # softmax
    # print(model_out.shape)  # (?, 300)
    model = Model(inputs=[model1_in], outputs=model_out)
    print(model.summary())
    # print(model.output_shape)  # (None, 300)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[auc])  # "accuracy" f1
    # 取某一层的输出为输出新建为model，采用函数模型
    # relu_layer_model = Model(inputs=model1_in, outputs=model.layers[2].output)
    return model

# create_classify_model(input_shape=(243,4,1),padding='valid',pool_size=(2, 1))