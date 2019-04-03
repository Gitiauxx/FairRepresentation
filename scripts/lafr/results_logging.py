import numpy as np
import pandas as pd
import os
import tensorflow as tf
import time
import yaml


class ResultLogger(object):
    def __init__(self, dname, tag=None):
        self.dname = dname
        
        if not os.path.exists(dname):
            os.mkdir(dname)
        
        self.ckptdir = os.path.join(self.dname, 'checkpoints')
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        if tag is None:
            t = int(time.time())
        else:
            t = tag

        self.testcsv_name = os.path.join(self.dname, 'test_metrics_{}.csv'.format(t))
        

    def save_metrics(self, D):
        self.testcsv = open(self.testcsv_name, 'w')
        """save D (a dictionary of metrics: string to float) as csv"""
        for k in D:
            s = '{},{:.7f}\n'.format(k, D[k])
            self.testcsv.write(s)
        self.testcsv.close()
        print('Metrics saved to {}'.format(self.testcsv_name))

    


