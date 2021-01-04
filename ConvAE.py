from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.datasets import mnist
from argslist import *

KERNEL_SIZE = [5, 3]

def CAE(input_shape=(DAYS_PERIOD, 1, 1), filters=[32, 64, 128, 10]):
    model = Sequential()

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    if len(input_shape) == 2:
        input_shape = tuple(list(input_shape) + [1])

    model.add(Conv2D(filters[0], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='conv1',
                     input_shape=input_shape))
    model.add(Conv2D(filters[1], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='conv2'))
    model.add(
        Conv2D(filters[2], KERNEL_SIZE[1], strides=2, padding=pad3, activation='relu',
               name='conv3'))  # todo : check pad3

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2] * 30, activation='relu'))  # 128*3*3  # todo : why does it divide 8?

    model.add(Reshape((-1, 1, filters[2])))

    model.add(
        Conv2DTranspose(filters[1], KERNEL_SIZE[1], strides=(2, 1), padding=pad3, activation='relu', name='deconv3'))
    model.add(
        Conv2DTranspose(filters[0], KERNEL_SIZE[0], strides=(2, 1), padding='same', activation='relu', name='deconv2'))
    model.add(Conv2DTranspose(input_shape[2], KERNEL_SIZE[0], strides=(2, 1), padding='same', name='deconv1'))
    model.summary()
    # model.add(Conv2D(filters[0], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    # model.add(Conv2D(filters[1], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='conv2'))
    # model.add(Conv2D(filters[2], KERNEL_SIZE[1], strides=2, padding=pad3, activation='relu', name='conv3'))  # todo : check pad3
    #
    # model.add(Flatten())
    # model.add(Dense(units=filters[3], name='embedding'))
    # model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))  # 128*3*3  # todo : why does it divide 8?
    #
    # model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    #
    # model.add(Conv2DTranspose(filters[1], KERNEL_SIZE[1], strides=2, padding=pad3, activation='relu', name='deconv3'))
    # model.add(Conv2DTranspose(filters[0], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='deconv2'))
    # model.add(Conv2DTranspose(input_shape[2], KERNEL_SIZE[0], strides=2, padding='same', name='deconv1'))
    # model.summary()
    return model


if __name__ == '__main__':
    from time import time

    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'usps'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps
    if args.dataset == 'mnist':
        (x, y), (_, _) = mnist.load_data()

    elif args.dataset == 'usps':
        raise Exception('no such a dataset')

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
    # plot_model(model, to_file=args.save_dir + f"/{args.dataset}-pretrain-model.png", show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger(args.save_dir + f"/{args.dataset}-pretrain-log.csv")

    # begin training
    t0 = time()
    x = np.expand_dims(x, axis=-1)
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + f"/{args.epochs}-pretrain-model-{args.epochs}.h5")

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    from . import metrics
    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
