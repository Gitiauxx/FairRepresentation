import tensorflow as tf
import numpy as np

class Test(object):

    def __init__(self,
                model,
                data,
                sess,
                reslogger):

        self.data = data
        self.model = model
        self.sess = sess
        self.reslogger = reslogger

    def evaluate(self, batch_size):

        self.batches_seen = 0
        S_aud = np.empty((0, 1))
        S_att = np.empty((0, 1))

        test_iter = self.data.get_batch_iterator('test', batch_size)
        test_L = {'auditor_acc': 0., 'auditor2_acc': 0., 'recon_err': 0., 'attacker_acc': 0., 'attacker_direct_acc': 0.}

        for x, s in test_iter:
            # encoder - decoder -- computing
            feed_dict = {self.model.X: x, self.model.S: s}
            enc_dec_op = self.model.decoder_loss
            dec_enc_loss = self.sess.run(enc_dec_op, feed_dict)
            test_L['recon_err'] += dec_enc_loss

            # auditor -- computing
            auditor_op = [self.model.S_aud, self.model.auditor_accuracy]
            S, auditor_acc = self.sess.run(auditor_op, feed_dict)
            test_L['auditor_acc'] += auditor_acc
            S_aud = np.concatenate((S_aud, S))

            # auditor2 -- computing
            auditor2_op = self.model.auditor2_accuracy
            auditor2_acc = self.sess.run(auditor2_op, feed_dict)
            test_L['auditor2_acc'] += auditor2_acc

            # attacker -- computing
            attacker_op = [self.model.S_att, self.model.attacker_accuracy]
            S, attacker_acc = self.sess.run(attacker_op, feed_dict)
            test_L['attacker_acc'] += attacker_acc
            S_att = np.concatenate((S_att, S))

            # attacker using inputs directly -- computing
            attacker_direct_op = self.model.attacker_direct_accuracy
            attacker_direct_acc = self.sess.run(attacker_direct_op, feed_dict)
            test_L['attacker_direct_acc'] += attacker_direct_acc

            self.batches_seen += 1

        for k in test_L:
                test_L[k] /= self.batches_seen
        
        self.reslogger.save_metrics(test_L)
        
        
        