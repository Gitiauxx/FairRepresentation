import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import argparse

from lafr import models, train, dataset, test, logging

def test1(n, resdirname, unbalance=1, sigma_noise=0.0, tag=None):

    # simulate a synthethic data with n covariates
    data = pd.DataFrame(index=np.arange(n))
    data['x1'] = np.random.normal(size=n)
    data['x2'] = np.random.normal(size=n)

    data['noise'] = np.random.normal(scale=sigma_noise, size=n)

    # create senstivie attribute
    data['w'] = np.exp(unbalance * (data['x2'] + data['x1'] + data['noise']) ** 2)
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
    md = models.DPGanLafr(xdim=2,
                          zdim=2,
                          auditor_coeff=0.5)

    # get dataset
    ds = dataset.Dataset('test_lafr.csv', ['x1', 'x2'], 'sensitive')

    # logging file
    reslogger = logging.ResultLogger(resdirname, tag=tag)

    with tf.Session() as sess:

        #create Trainer
        trainer = train.Train(md, ds, sess=sess)
        tester = test.Test(md, ds, sess, reslogger)

        # training
        trainer.fit(100, 1)

        # evaluation
        tester.evaluate(32)

def main(**kwargs):

    unbalance = 0
    if kwargs['unbalance']:
        unbalance = kwargs['unbalance']
    if kwargs['tag']:
        tag = kwargs['tag']
    
    test1(20000, 'C:\\Users\\Xavier\\fair_representation\\FairRepresentation\\data\\tests', 
                unbalance = unbalance,
                tag=tag)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--unbalance')
    parser.add_argument('--tag')
    args = parser.parse_args()

    kwargs_dict = {'unbalance': float(args.unbalance), 'tag':args.tag}
    
    main(**kwargs_dict)



