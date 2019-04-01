import tensorflow as tf

# defaults
BATCH_SIZE = 32

class Train(object):

    def __init__(self,
                model,
                data,
                sess=None,
                learning_rate=0.001,
                batch_size=32):

        self.data = data
        self.model = model
        self.batch_size = batch_size

        # auditor train op
        self.opt_aud = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.aud_op = self.opt_aud.minimize(
            self.model.auditor_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/auditor')
        )

        # attacker train op
        self.opt_att = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.att_op = self.opt_att.minimize(
            self.model.attacker_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/attacker')
        )

        # encoder-decoder op
        self.opt_enc_dec = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.enc_dec_op = self.opt_enc_dec.minimize(
            self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/enc_dec')
        )

        # initialize session
        self.sess = sess or tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # for saving model checkpoints

    
    def fit(self, n_epochs, patience, **kwargs):

        for epoch in range(n_epochs):
            self.batches_seen = 0
            print('starting Epoch {:d}'.format(epoch))

            train_iter = self.data.get_batch_iterator('train', self.batch_size)
            train_L = {'auditor_err': 0, 'recon_err': 0., 'attacker_err': 0.}

            for x, s in train_iter:
                # encoder - decoder training
                feed_dict = {self.model.X: x, self.model.S: s}
                enc_dec_op = [self.enc_dec_op, self.model.decoder_loss]
                _, dec_enc_loss = self.sess.run(enc_dec_op, feed_dict)
                train_L['recon_err'] += dec_enc_loss

                # auditor training
                auditor_op = [self.aud_op, self.model.auditor_loss]
                _, auditor_loss = self.sess.run(auditor_op, feed_dict)
                train_L['auditor_err'] += auditor_loss

                # attacker training
                attacker_op = [self.att_op, self.model.attacker_accuracy]
                _, attacker_acc = self.sess.run(attacker_op, feed_dict)
                train_L['attacker_err'] += attacker_acc

                self.batches_seen += 1

            for k in train_L:
                train_L[k] /= self.batches_seen
            
            print(train_L)


