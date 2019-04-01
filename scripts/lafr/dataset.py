import numpy as np
import pandas as pd

class Dataset(object):

    def __init__(self, 
                datafile,
                features,
                sensitive_attribute,
                seed=0):
        
        self.datafile = datafile
        self.seed = seed
        self.features = features
        self.sensitive_attribute = sensitive_attribute

        self.loaded = False
        self.load()

    def load(self):
        if self.loaded == False:
            data = pd.read_csv(self.datafile)
            data.set_index(np.arange(len(data)), inplace=True)
            train = data.loc[np.random.choice(data.index, int(0.7 * len(data)), replace=False), :]
            test = data.drop(train.index)

            self.x_train = np.array(train[self.features])
            self.x_test = np.array(test[self.features])

            self.sensitive_train = np.array(train[self.sensitive_attribute])
            self.sensitive_train = self.sensitive_train[:, np.newaxis]
            self.sensitive_test = np.array(test[self.sensitive_attribute])
            self.sensitive_test = self.sensitive_test[:, np.newaxis]
            
            self.loaded = True

    def get_batch_iterator(self, phase, b_size):
        
        if phase == 'train':
            x = self.x_train
            s = self.sensitive_train
        
        elif phase == 'test':
            x = self.x_test
            s = self.sensitive_test
        
        else:
            raise Exception("invalid phase name")

        n_size = x.shape[0]
        batch_inds = self.make_batch_inds(n_size, b_size, self.seed, phase)
        
        for ind in batch_inds:
            yield x[ind, :], s[ind]

    def make_batch_inds(self, n, mb_size, seed=0, phase='train'):
        np.random.seed(seed)
        if phase == 'train':
            shuf = np.random.permutation(n)
        else:
            shuf = np.arange(n)
        
        start = 0
        mbs = []
        while start < n:
            end = min(start + mb_size, n)
            mb_i = shuf[start:end]
            mbs.append(mb_i)
            start = end
        return mbs
        
       
