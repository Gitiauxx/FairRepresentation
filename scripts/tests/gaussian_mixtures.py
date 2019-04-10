import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

from lafr import models, train, dataset, test, results_logging

def test1(n, resdirname, unbalance=0.5, sigma_noise=0.1, tag=None, auditor_coeff=0.25):

    # create 8 Gaussians
    cov = [[0.01, 0], [0, 0.01]]
    X = np.zeros((n, 2, 8))

    X[:, :, 0] = np.random.multivariate_normal([1, 0], cov=cov, size=n)
    X[:, :, 1] = np.random.multivariate_normal([0.75, 0.75], cov=cov, size=n)
    X[:, :, 2] = np.random.multivariate_normal([1, 0], cov=cov, size=n)
    X[:, :, 3] = np.random.multivariate_normal([0.75, -0.75], cov=cov, size=n)
    X[:, :, 4] = np.random.multivariate_normal([-1, 0], cov=cov, size=n)
    X[:, :, 5] = np.random.multivariate_normal([-0.75, -0.75], cov=cov, size=n)
    X[:, :, 6] = np.random.multivariate_normal([-1, 0], cov=cov, size=n)
    X[:, :, 7] = np.random.multivariate_normal([-0.75, 0.75], cov=cov, size=n)

    # weight by sensitive attributes
    w0 = np.ones(8)
    w1 = np.ones(8)
    w1[[1, 3, 5, 7]] = 1 - unbalance
    w1[[0, 2, 4, 6]] = 1 + unbalance
    
    # create Gaussian mixture
    X1X0 = np.zeros((2*n, 2))
    for i in range(n):
        j0 = np.random.choice(np.arange(8), 1, p=w0/w0.sum())
        j1 = np.random.choice(np.arange(8), 1, p=w1/w1.sum())
        X1X0[i, :] = X[i, :, j0]
        X1X0[n + i, :] = X[i, :, j1]

    # simulate a synthethic data with n covariates
    data = pd.DataFrame(index=np.arange(2 *n))
    data['x1'] = X1X0[:, 0]
    data['x2'] = X1X0[:, 1]
    data['sensitive'] = 0
    data.loc[data.index >= n, 'sensitive'] = 1
    
    data1 = data.loc[np.random.choice(data[data.sensitive == 1].index, 5000, replace=True), :]
    data2 = data.loc[np.random.choice(data[data.sensitive == 0].index, 5000, replace=True), :]
    data = pd.concat([data1, data2])
    data = data.loc[np.random.choice(data.index, len(data), replace=False)]
    data.to_csv('test_lafr.csv')

    # create model
    md = models.DPGanLafr(xdim=2,
                          zdim=2,
                          auditor_coeff=auditor_coeff)

    # get dataset
    ds = dataset.Dataset('test_lafr.csv', ['x1', 'x2'], 'sensitive')

    # logging file
    reslogger = results_logging.ResultLogger(resdirname, tag=tag)

    with tf.Session() as sess:

        #create Trainer
        trainer = train.Train(md, ds, sess=sess)
        tester = test.Test(md, ds, sess, reslogger)

        # training
        trainer.fit(40, 1)

        # atacker 
        trainer.fit_attack(40)

        print(trainer.train_L)

        # evaluation
        tester.evaluate(32)

def main(dirname, **kwargs):

    if 'unbalance' in kwargs:
        unbalance = kwargs['unbalance']
    if 'tag' in kwargs:
        tag = kwargs['tag']
    if 'sigma_noise' in kwargs:
        sigma_noise = kwargs['sigma_noise']
    if 'auditor_coeff' in kwargs:
        auditor_coeff = kwargs['auditor_coeff']
    
    test1(20000, dirname, 
                auditor_coeff = auditor_coeff,
                tag=tag)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname')
    parser.add_argument('--fairness')
    parser.add_argument('--tag')
    args = parser.parse_args()

    kwargs_dict = {'auditor_coeff': float(args.fairness), 'tag':args.tag}
    
    main(args.dirname, **kwargs_dict)



