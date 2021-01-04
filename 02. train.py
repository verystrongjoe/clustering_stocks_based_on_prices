"""
코스피, 코스닥 수익율 데이터 가지고 time window로 짤라 데이터 만들기
"""
import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd
import numpy as np
from argslist import *
import os
from DCEC import DCEC
import metrics

if __name__ == '__main__':
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/temp')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    x_dataset = np.load('data/x_dataset.npy')
    y_dataset = np.load('data/y_dataset.npy')
    n_stocks = len(x_dataset)

    # x_dataset, y_dataset : (samples, DAYS_PERIOD, 1), (samples, 2)
    x = x_dataset.reshape((-1, DAYS_PERIOD, 1))

    """
    여기 y 데이터는 종목코드와 1년 샘플링 시작 년월 인덱스가 넘어오므로 학습용 y가 아니다
    Then set γ = 0.1 and update CAE’s weights, cluster centers and target distribution P as follows.
    그래서 논문에서처럼 별도로 y를 gamma를 0으로 하고 학습된 target distribution을 ground truth soft label로 사용해야 함!!
    """
    y = y_dataset.reshape((-1, 2))

    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=args.n_clusters)
    dcec.model.summary()

    # begin clustering
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
    dcec.fit(x, tol=args.tol, max_iter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights)
    y_pred = dcec.y_pred



    # print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))


