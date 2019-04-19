import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import os

from lafr import models, train, dataset, test, results_logging

def model(resdirname, tag=None, auditor_coeff=0.25):

    # simulate a synthethic data with n covariates
    features = ['age',
                'workclass',
                'education',
                'education-num',
                'marital-status',
                'occupation',
                'relationship',
                'capital-gain',
                'capital-loss',
                'hours-per-week']

    # create model
    md = models.DPGanLafr(xdim=len(features),
                          zdim=len(features),
                          auditor_coeff=auditor_coeff)

    
    # get dataset
    ds = dataset.Dataset(os.path.join(resdirname,'adult_all_clean.txt'), features, 'attr')

    # logging file
    reslogger = results_logging.ResultLogger(resdirname, tag=tag)

    with tf.Session() as sess:

        #create Trainer
        trainer = train.Train(md, ds, sess=sess, learning_rate=0.0001, batch_size=64)
        tester = test.Test(md, ds, sess, reslogger)

        # training
        trainer.fit(100, 1)

        # atacker 
        trainer.fit_attack(100)

        # evaluation
        tester.evaluate(32)

def main(dirname, **kwargs):

    if 'auditor_coeff' in kwargs:
        auditor_coeff = kwargs['auditor_coeff']

    if 'tag' in kwargs:
        tag = kwargs['tag']
    
    model(dirname, 
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



