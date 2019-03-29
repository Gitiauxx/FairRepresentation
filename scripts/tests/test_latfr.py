import pandas as pd
import numpy as np
import tensorflow as tf

from lafr import models, train, dataset

def test1(n, unbalance=0, sigma_noise=0.0):
    

    # simulate a synthethic data with n covariates
    data = pd.DataFrame(index=np.arange(n))
    data['x1'] = np.random.normal(size=n)
    data['x2'] = np.random.normal(size=n)

    data['noise'] = np.random.normal(scale=sigma_noise, size=n)

    # create senstivie attribute
    data['w'] = np.exp(unbalance * (data['x2'] + data['x1']) ** 2)
    data['w'] = data['w'] / (1 + data['w'])
  
    # draw sensitive attr accrding to w
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u < data.w, 'sensitive'] = 1
    data.loc[data.u >= data.w, 'sensitive'] = 0
    data.to_csv('test_lafr.csv')

    # create model
    md = models.DPGanLafr(xdim=2,
                          zdim=2)

    # get dataset
    ds = dataset.Dataset('test_lafr.csv', ['x1', 'x2'], 'sensitive')

    with tf.Session as sess:

        #create Trainer
        trainer = Train(md, ds, sess=sess)

        # training
        trainer.fit(10, 1)

if __name__ == "__main__":
    test1(1000)



