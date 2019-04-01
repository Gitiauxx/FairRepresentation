import tensorflow as tf
from lafr.mlp import MLP
from abc import ABC, abstractmethod

# defaults
EPS = 1e-8
HIDDEN_LAYER_SPECS = {  # format as dict that maps network to list of hidden layer widths
    'enc': [8, 2],
    'cla': [1],
    'dec': [8, 2],
    'aud': [8, 2],
    'att': [8, 2]
}
CLASS_COEFF = 1.
AUDITOR_COEFF = 0.
RECON_COEFF = 1
XDIM = 61
YDIM = 1
ZDIM = 10
SDIM = 1
S_WTS = [1., 1.]
Y_WTS = [1., 1.]
AY_WTS = [[1., 1.], [1., 1.]]
SEED = 0
ACTIV = 'leakyrelu'
HINGE = 0.


class AbstractBaseNet(ABC):

    def __init__(self,
                 recon_coeff=RECON_COEFF,
                 class_coeff=CLASS_COEFF,
                 auditor_coeff=AUDITOR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 zdim=ZDIM,
                 sdim=SDIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 hinge=HINGE,
                 **kwargs):
        self.recon_coeff = recon_coeff
        self.class_coeff = class_coeff
        self.auditor_coeff = auditor_coeff
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.sdim = sdim
        self.hidden_layer_specs = hidden_layer_specs
        self.seed = seed
        self.hinge = hinge
        tf.set_random_seed(self.seed)

        self._define_vars()
        self.Z = self._encode(self.X)
        self.X_hat = self._decode(self.Z)
        self.S_hat_logits = self._auditor(self.Z)
        self.S_hat_attack_logits= self._attacker(self.Z)
        self.S_hat = self._get_sensitive_from_logits(self.S_hat_logits)
        self.S_hat_attack= self._get_sensitive_from_logits(self.S_hat_attack_logits)
        self.S_att = self._get_prediction_from_logits(self.S_hat_attack_logits)
        self.S_aud = self._get_prediction_from_logits(self.S_hat_logits)

        self.decoder_loss = self._get_dec_loss(self.X, self.X_hat)
        self.auditor_loss = self._get_aud_loss(self.S, self.S_hat)
        self.attacker_loss = self._get_att_loss(self.S, self.S_hat_attack)
        self.attacker_accuracy = self._get_acc(self.S, self.S_att)
        self.auditor_accuracy = self._get_acc(self.S, self.S_aud)
        self.loss = self._get_loss()

    
    @abstractmethod
    def _define_vars(self):  # declare tensorflow variables and placeholders
        pass

    @abstractmethod
    def _encode(self, inputs, scope_name='model/enc_dec', reuse=False): # encode inputs to latent
        pass

    @abstractmethod
    def _decode(self, latents, scope_name='model/enc_dec', reuse=False): # decode latents to inputs
        pass
    
    @abstractmethod
    def _auditor(self, latents, scope_name='model/auditor', reuse=False): # auditor network from latents to sensitive attribute
        pass

    @abstractmethod
    def _attacker(self, latent, scope_name='model/attacker', reuse=False): # attacker to find sub-population where it can predict sensitive attributes
        pass
    
    @abstractmethod
    def _get_dec_loss(self): # decoder loss
        pass

    @abstractmethod
    def _get_sensitive_from_logits(self, S_hat_logits): # transform logit from auditor into sensitive attribute
        pass

    @abstractmethod
    def _get_aud_loss(self, S, S_hat): # auditor loss
        pass

    @abstractmethod
    def _get_att_loss(self, S, S_hat): # attacker loss
        pass

    @abstractmethod
    def _get_loss(self):  # compute decoder complete loss (including opposite of auditor loss)
        pass

    @abstractmethod
    def _get_acc(self): # compute attacker accuracy
        pass


class DPGanLafr(AbstractBaseNet):
    def _define_vars(self):
        
        assert(
            isinstance(self.hidden_layer_specs, dict) and
            all([net_name in self.hidden_layer_specs for net_name in ['enc', 'cla', 'aud', 'dec', 'att']])
        )
        self.X = tf.placeholder("float", [None, self.xdim], name='X')
        self.Y = tf.placeholder("float", [None, self.ydim], name='Y')
        self.S = tf.placeholder("float", [None, self.sdim], name='S')
        self.epoch = tf.placeholder("float", [1], name='epoch')
        return

    def _encode(self, inputs, scope_name='model/enc_dec', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='inputs_to_latents',
                      shapes=[self.xdim] + self.hidden_layer_specs['enc'] + [self.zdim],
                      activ=ACTIV)
            return mlp.forward(inputs)

    def _decode(self, latents, scope_name='model/enc_dec', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='latents_to_inputs',
                      shapes=[self.zdim + 1] + self.hidden_layer_specs['dec'] + [self.xdim],
                      activ=ACTIV)
            Z_and_S = tf.concat([latents, self.S], axis=1)
            return mlp.forward(Z_and_S)

    def _auditor(self, latents, scope_name='model/auditor', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='latents_to_sensitive',
                      shapes=[self.zdim] + self.hidden_layer_specs['aud'] + [self.sdim],
                      activ=ACTIV)
            return mlp.forward(latents)

    def _attacker(self, latents, scope_name='model/attacker', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='latents_to_sensitive_attack',
                      shapes=[self.zdim] + self.hidden_layer_specs['att'] + [self.sdim],
                      activ=ACTIV)
            return mlp.forward(latents)

    def _get_sensitive_from_logits(self, S_hat_logits):
        return tf.nn.sigmoid(S_hat_logits)
    
    def _get_prediction_from_logits(self, logits):
        return tf.round(tf.nn.sigmoid(logits))

    def _get_dec_loss(self, X, X_hat):
        return tf.reduce_mean(tf.square(X - X_hat))

    def _get_aud_loss(self, S, S_hat, eps=EPS):
        n_1 = tf.reduce_sum(S)
        n_2 = tf.reduce_sum(1 - S)
        return tf.reduce_sum(1 / n_1 * S * (1 - S_hat) + 1 / n_2 * (1 - S) * S_hat)
        #return tf.reduce_mean(S * (1 - S_hat) +  (1 - S) * S_hat)
        #return - tf.reduce_mean( S * tf.log(S_hat + eps) + (1 - S) * tf.log(1 - S_hat + eps))

    def _get_att_loss(self, S, S_hat, eps=EPS):
        n_1 = tf.reduce_sum(S)
        n_2 = tf.reduce_sum(1 - S)
        #return - tf.reduce_mean( S * tf.log(S_hat + eps) + (1 - S) * tf.log(1 - S_hat + eps))
        return tf.reduce_sum(1 / n_1 * S * (1 - S_hat) + 1 / n_2 * (1 - S) * S_hat)
        #return tf.reduce_mean(S * (1 - S_hat) +  (1 - S) * S_hat)

    def _get_acc(self, S, S_pred):
        return 1. - tf.reduce_mean(S * (1 - S_pred) + (1 - S) * S_pred)


    def _get_loss(self):
        return self.recon_coeff * self.decoder_loss - self.auditor_coeff * self.auditor_loss