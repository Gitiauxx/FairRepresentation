import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

from lafr import models, train, dataset, test, results_logging

def test1(n, resdirname, unbalance=1, sigma_noise=0.1, tag=None, auditor_coeff=0.25):

    # simulate a synthethic data with n covariates
    data = pd.DataFrame(index=np.arange(n))
    data['x1'] = np.random.normal(size=n)
    data['x2'] = data['x1'] + 0.2 * np.random.normal(size=n)
    data['x3'] = data['x2'] * data['x1'] + np.random.normal(size=n)

    data['noise'] = np.random.normal(scale=sigma_noise, size=n)

    # create senstivie attribute
    data['w'] = np.exp(unbalance * (data['x2'] + data['x1'] + data['x3'] + data['noise']) ** 2)
    data['w'] = data['w'] / (1 + data['w'])
  
    # draw sensitive attr accrding to w
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u < data.w, 'sensitive'] = 1
    data.loc[data.u >= data.w, 'sensitive'] = 0

    data1 = data.loc[np.random.choice(data[data.sensitive == 1].index, 5000, replace=True), :]
    data2 = data.loc[np.random.choice(data[data.sensitive == 0].index, 5000, replace=True), :]
    data = pd.concat([data1, data2])
    data.to_csv('test_lafr.csv')

    # create model
    md = models.DPGanLafr(xdim=3,
                          zdim=3,
                          auditor_coeff=auditor_coeff)

    # get dataset
    ds = dataset.Dataset('test_lafr.csv', ['x1', 'x2', 'x3'], 'sensitive')

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



