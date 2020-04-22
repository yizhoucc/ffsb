import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy

from stable_baselines.ddpg.policies import DDPGPolicy,FeedForwardPolicy



class SoftPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(SoftPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp",act_fun=tf.nn.selu, **_kwargs)


# class SoftPolicy(DDPGPolicy):


#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
#                  layer_norm=False, act_fun=tf.nn.selu, **kwargs):
#         super(SoftPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)
#         self.layer_norm = layer_norm

#         self.reuse = reuse
#         self._qvalue = None
#         if layers is None:
#             layers = [32,64]
#         self.layers = layers

#         assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

#         self.activ = act_fun

#     def make_actor(self, obs=None, reuse=False, scope="pi"):
#         if obs is None:
#             obs = self.processed_obs

#         with tf.variable_scope(scope, reuse=reuse):

#             pi_h = tf.layers.flatten(obs)
#             for i, layer_size in enumerate(self.layers):
#                 pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
#                 if self.layer_norm:
#                     pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
#                 pi_h = self.activ(pi_h)
#             self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], name=scope,
#                                                      kernel_initializer=tf.zeros_initializer))#random_uniform_initializer(minval=-3e-3,
#                                                                                               #        maxval=3e-3)))
#         return self.policy

#     def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
#         if obs is None:
#             obs = self.processed_obs
#         if action is None:
#             action = self.action_ph

#         with tf.variable_scope(scope, reuse=reuse):
           
#             qf_h = tf.layers.flatten(obs)
#             for i, layer_size in enumerate(self.layers):
#                 qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
#                 if self.layer_norm:
#                     qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
#                 qf_h = self.activ(qf_h)
#                 if i == 0:
#                     qf_h = tf.concat([qf_h, action], axis=-1)

#             # the name attribute is used in pop-art normalization
#             qvalue_fn = tf.layers.dense(qf_h, 1, name='qf_output',
#                                         kernel_initializer=tf.zeros_initializer)#random_uniform_initializer(minval=-3e-3,

#             self.qvalue_fn = qvalue_fn
#             self._qvalue = qvalue_fn[:, 0]
#         return self.qvalue_fn

#     def step(self, obs, state=None, mask=None):
#         return self.sess.run(self.policy, {self.obs_ph: obs})

#     def proba_step(self, obs, state=None, mask=None):
#         return self.sess.run(self.policy, {self.obs_ph: obs})

#     def value(self, obs, action, state=None, mask=None):
#         return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})

register_policy("SoftPolicy", SoftPolicy)