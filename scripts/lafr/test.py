import tensorflow as tf
import numpy as np

class Test(object):

    def __init__(self,
                model,
                data,
                sess):

        self.data = data
        self.model = model
        self.sess = sess

    def evaluate(self, batch_size):

        self.batches_seen = 0
        S_aud = np.empty((0, 1))
        S_att = np.empty((0, 1))

        test_iter = self.data.get_batch_iterator('test', batch_size)
        test_L = {'auditor_acc': 0, 'recon_err': 0., 'attacker_acc': 0.}

        for x, s in test_iter:
            # encoder - decoder training
            feed_dict = {self.model.X: x, self.model.S: s}
            enc_dec_op = self.model.decoder_loss
            dec_enc_loss = self.sess.run(enc_dec_op, feed_dict)
            test_L['recon_err'] += dec_enc_loss

            # auditor training
            auditor_op = [self.model.S_aud, self.model.auditor_accuracy]
            S, auditor_acc = self.sess.run(auditor_op, feed_dict)
            test_L['auditor_acc'] += auditor_acc
            S_aud = np.concatenate((S_aud, S))

            # attacker training
            attacker_op = [self.model.S_att, self.model.attacker_accuracy]
            S, attacker_acc = self.sess.run(attacker_op, feed_dict)
            test_L['attacker_acc'] += attacker_acc
            S_att = np.concatenate((S_att, S))

            self.batches_seen += 1

        for k in test_L:
                test_L[k] /= self.batches_seen
        
        print("Predicting")
        print(test_L)
