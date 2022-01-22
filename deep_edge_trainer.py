# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains Graph Embeddings (node embeddings + Edge function).

This trainer runs on the output of `create_dataset_arrays.py`.
"""

import edge_nn
import collections
import pickle
import os
import random

import numpy
from sklearn import metrics

import tensorflow as tf
# from tensorflow import flags
from tensorflow.python.platform import flags
import argparse
# from tensorflow import app
import tensorflow.compat.v1.gfile as gfile2
from tensorflow.io.gfile import GFile as gfile
# from tensorflow import gfile
# from tensorflow import logging

def check_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_dir", type=str, help='Directory where all dataset files live. All data files ' +
                    'must be located here. Including {train,test}.txt.npy, ' +
                    'train.pairs.*.txt.npy, {train,test}.neg.txt.npy.',
                      nargs='?', default="output/", const="output/")
  parser.add_argument("--restore", type=int, help='If set to 1 and dump was previously found, it will be restored.',
                      nargs='?', default=1, const=1)
  parser.add_argument("--num_nodes", type=int, help='Number of nodes in graph. Node IDs in all dataset files must range from 0 to (num_nodes - 1).',
                      nargs='?', default=0, const=0)
  parser.add_argument("--embed_dim",type=int, help='Size of node embedding.',
                      nargs='?', default=100, const=100)
  parser.add_argument("--dnn_dims", type=str, help='Comma-separated layer dimensions for deep embedding transformation.',
                      nargs='?', default='100,100', const='100,100')
  parser.add_argument("--projection_dim", type=int, help='Inner dimensions of L and R projection matrices. If set ' +
                     'to <= 0, then no Asymetric Projection is applied, and ' +
                     'edge function becomes a hadamard product of node ' +
                     'embeddings, followed by dot-product with a trainable ' +
                     'vector',
                      nargs='?', default=32, const=32)
  parser.add_argument("--num_projections", type=int, help='Number of asymmetric matrices (L and R) to use. If >1, ' +
                     'then output of DNN(left) x L_i x R_i x DNN(right) for i ' +
                     'in [0, this flag) will be concatenated, and a fully-'     +
                     'connected layer is added to give a scalar (edge score). ' +
                     'If set to 1, then no fully connected layer is added. We ' +
                     'found that setting to >1 gives only slight improvement.',
                      nargs='?', default=1, const=1)
  parser.add_argument("--description_prefix", type=str, help="You may write the prefix for the description",
                      nargs='?', default='test', const='test')
  parser.add_argument("--num_iterations", type=int, help="Number of iterations or epochs",
                      nargs='?', default=10, const=10)
  parser.add_argument("--save_every", type=int, help="Saves every this many iterations.",
                      nargs='?', default=500, const=500)
  parser.add_argument("--verbose", type=bool, help="If set, prints info during training.",
                      nargs='?', default=False, const=False)

  args = parser.parse_args()
  global dataset_dir, restore, embed_dim, dnn_dims, projection_dim, num_projections, description_prefix, num_iterations, save_every, verbose, num_nodes
  dataset_dir = args.dataset_dir
  num_nodes = args.num_nodes
  restore = args.restore
  embed_dim = args.embed_dim
  dnn_dims = args.dnn_dims
  projection_dim = args.projection_dim
  num_projections = args.num_projections
  description_prefix = args.description_prefix
  num_iterations = args.num_iterations
  save_every = args.save_every
  verbose = args.verbose
    
    
  

# flags.DEFINE_string('dataset_dir', '',
#                     'Directory where all dataset files live. All data files '
#                     'must be located here. Including {train,test}.txt.npy, '
#                     'train.pairs.*.txt.npy, {train,test}.neg.txt.npy.')
# flags.DEFINE_integer('restore', 1,
#                      'If set to 1 and dump was previously found, it will be '
#                      'restored.')
# flags.DEFINE_integer('embed_dim', 100, 'Size of node embedding.')
# flags.DEFINE_string('dnn_dims', '100,100',
#                     'Comma-separated layer dimensions for deep embedding '
#                     'transformation.')
# flags.DEFINE_integer('projection_dim', 32,
#                      'Inner dimensions of L and R projection matrices. If set '
#                      'to <= 0, then no Asymetric Projection is applied, and '
#                      'edge function becomes a hadamard product of node '
#                      'embeddings, followed by dot-product with a trainable '
#                      'vector')
# flags.DEFINE_integer('num_projections', 1,
#                      'Number of asymmetric matrices (L and R) to use. If >1, '
#                      'then output of DNN(left) x L_i x R_i x DNN(right) for i '
#                      'in [0, this flag) will be concatenated, and a fully-'
#                      'connected layer is added to give a scalar (edge score). '
#                      'If set to 1, then no fully connected layer is added. We '
#                      'found that setting to >1 gives only slight improvement.')
# flags.DEFINE_integer('num_nodes', 0,
#                      'Number of nodes in graph. Node IDs in all dataset files '
#                      'must range from 0 to (num_nodes - 1).')
# flags.DEFINE_string('description_prefix', 'test', '')
# flags.DEFINE_integer('num_iterations', 10, '')
# flags.DEFINE_integer('save_every', 500, 'Saves every this many iterations.')
# flags.DEFINE_boolean('verbose', False, 'If set, prints info during training.')

# FLAGS = flags.FLAGS

BATCH_SIZE = 1000  # Batch size during training.
NCE_K = 5  # Number of noise samples per anchor.


def Description():
  """Returns a string that encodes model hyper-parameters.
  
  String is used to create a sub-directory for storing the model files.
  """
  description = '%s.d%i_f%s_g%i' % (
      description_prefix, embed_dim,
      dnn_dims.replace(',', 'x'), projection_dim)
  return description


def DatasetFileName(filename):
  """Returns location of `filename` within --dataset_dir`."""
  return os.path.join(dataset_dir, filename)


def ModelFileName(suffix):
  """Location of filename `Description()`_`suffix` in `dataset_dir/dumps`."""
  return os.path.join(dataset_dir, 'dumps',
                      '%s_%s' % (Description(), suffix))


def InFile(suffix):
  """Opens file `ModelFileName(suffix)` for reading."""
  # return gfile.Open(ModelFileName(suffix))
  # changed
  return gfile(ModelFileName(suffix))


def OutFile(suffix):
  """Opens file `ModelFileName(suffix)` for writing."""
  # return gfile.Open(ModelFileName(suffix), 'w')
  # changed
  return gfile(ModelFileName(suffix), 'w')


def estimate_AUC(sess, nn, embeddings, positives, negatives):
  """Measures the AUC-ROC given positive and negative edges.
  
  Args:
    sess: TensorFlow session that contains parameters of edge neural net `nn`.
    nn: Edge Neural Network. It is expected that it contains members:
      embeddings_a: (B, D) float tensor for inputting embedding vectors, where
        `B` is batch size and `D` is the input node embedding dimensionality.
      embeddings_b: (B, D) float tensor for inputting embedding vectors. Every
        row corresponds to `embeddings_a`
      batch_size: int tensor, which will be fed `B`.
      output: Output float tensor with shape (B). Every entry `i` contains the
        edge score between `embeddings_a[i]` and `embeddings_b[i]`.
    embeddings: (|V|, D) float32 numpy.array containing all node embeddings.
    positives: (P, 2) int32 numpy.array containing positive edges, where
      `positives[j]` contains (node ID 1, node ID 2) with both IDs in range
      [0, |V|-1].
    negatives: (N, 2) int32 numpy.array containing negative edges.

  Returns:
    roc-auc score for ranking `positives` above `negatives`.
  """
  all_pairs = numpy.concatenate([positives, negatives], 0)
  all_scores = sess.run(
      nn.output,
      feed_dict={
          nn.embeddings_a: embeddings[all_pairs[:, 0]],
          nn.embeddings_b: embeddings[all_pairs[:, 1]],
          nn.batch_size: len(all_pairs),
      })
  pos_scores = all_scores[:len(positives)]
  neg_scores = all_scores[-len(negatives):]
  return metrics.roc_auc_score(
      [1] * len(pos_scores) + [0] * len(neg_scores),
      numpy.concatenate([pos_scores, neg_scores], 0))


class Evaluator(object):
  """Evaluates the accuracy of the model on the data.
  
  If you have a custom way to calculate evals, you can override this class.
  """

  def __init__(self, positive_pairs_file, negative_pairs_file):
    # self.pos_data = numpy.load(gfile2.Open(positive_pairs_file))
    # self.neg_data = numpy.load(gfile2.Open(negative_pairs_file))
    # changed
    self.pos_data = numpy.load(positive_pairs_file)
    self.neg_data = numpy.load(negative_pairs_file)

  def calculate_accuracy(self, session, nn, embeddings):
    return estimate_AUC(session, nn, embeddings, self.pos_data, self.neg_data)


class TrainNegatives(object):
  """Reads negatives file and stores in-memory all negatives per node."""

  def __init__(self, train_negatives_file):
    #train_negatives_arr = numpy.load(gfile.Open(train_negatives_file))
    # changed
    train_negatives_arr = numpy.load(train_negatives_file)
    self.negatives_dict = collections.defaultdict(list)
    for n1, n2 in train_negatives_arr:
      self.negatives_dict[n1].append(n2)
      # self.negatives_dict[n2].append(n1)

  def sample(self, nodes, num_negatives):
    """Returns `num_negatives` arrays each of size len(nodes)"""
    negatives = numpy.zeros(shape=(len(nodes), num_negatives), dtype='int32')
    for i, n in enumerate(nodes):
      node_negatives = self.negatives_dict[n]
      for j in range(num_negatives):
        selected_random_id = random.randint(0, len(node_negatives) - 1)
        negatives[i, j] = node_negatives[selected_random_id]

    return negatives


class TrainPairsReader(object):
  """Reads all train.pairs.*.npy files.

  Each time calling `next_pairs_array()` will return the next train.pairs.*.npy
  file. It round-robins across the files."""

  def __init__(self):
    files = gfile2.Glob(DatasetFileName('train.pairs.*.npy'))
    random.shuffle(files)
    self.train_npy_files = files
    self.next_idx = 0

  def next_pairs_array(self):
    #arr = numpy.load(gfile.Open(self.train_npy_files[self.next_idx]))
    # changged
    arr = numpy.load(self.train_npy_files[self.next_idx])
    # indices = range(len(arr))
    # changed
    indices = list(range(len(arr)))
    random.shuffle(indices)
    arr = arr[indices]
    self.next_idx = (self.next_idx + 1) % len(self.train_npy_files)
    return arr


def main(num_nodes):
  """Trains embeddings and edge function, recording eval metrics."""
  assert len(description_prefix) > 0
  assert len(dataset_dir) > 0
  if num_nodes == 0:
    index_pkl_file = DatasetFileName('index.pkl')
    # added
    with open(index_pkl_file, 'rb') as pickle_file:
      content = pickle.load(pickle_file)
    # num_nodes = len(pickle.load(gfile.Open(index_pkl_file))['index'])
    # changed
    num_nodes = len(content['index'])
  assert num_nodes > 0

  dumps_dir = DatasetFileName('dumps')
  if not gfile2.Exists(dumps_dir):
    gfile2.MakeDirs(dumps_dir)

  # Used for model selection. Note: Positive training data are positive node
  # pairs, generated via Random Walks on the train graph (train.txt.npy).
  TRAIN_EVALUATOR = Evaluator(DatasetFileName('train.txt.npy'),
                              DatasetFileName('train.neg.txt.npy'))

  directed_negs_file = DatasetFileName('test.directed.neg.txt.npy')
  if gfile2.Exists(directed_negs_file):
    print( 'evaluating on directed edges')
    test_negative_file = directed_negs_file
  else:
    test_negative_file = DatasetFileName('test.neg.txt.npy')

  TEST_EVALUATOR = Evaluator(DatasetFileName('test.txt.npy'),
                             test_negative_file)

  # Create the edge Neural Network.
  NN = edge_nn.EdgeNN()
  NN.build_net(
      embedding_dim=embed_dim, dnn_dims=dnn_dims,
      projection_dim=projection_dim,
      num_projections=num_projections)

  # Embeddings matrix
  EMBEDDINGS = numpy.array(
      numpy.random.uniform(
          low=-0.1, high=0.1, size=(num_nodes, embed_dim)),
      dtype='float32')

  # Initialize to be unit norm.
  EMBEDDINGS = (EMBEDDINGS.T / numpy.sqrt((EMBEDDINGS ** 2).sum(axis=1))).T

  # Positive training data.
  TRAIN_PAIRS = TrainPairsReader()

  # Initialize TensorFlow session.
  SESS = tf.compat.v1.Session()
  SESS.run(tf.compat.v1.global_variables_initializer())

  # Negative training data.
  TRAIN_NEGATIVES = TrainNegatives(
      DatasetFileName('train.neg_per_node.txt.npy'))

  ALL_VARS = tf.compat.v1.global_variables()

  TRAILS = {
      'train_auc': [],
      'test_auc': [],
      'embedding_lrs': [],
      'best_train_auc': 0,  # Best train AUC
      'best_test_auc': 0,  # Test AUC at Best train AUC
      'j': 0,
      'j_list': [],
      'i_list': [],
  }

  def restore(suffix=''):
    if not gfile.Exists(ModelFileName('embeddings.npy' + suffix)):
      print("Starting with fresh model...")
      return
    print("Restoring ...")

    EMBEDDINGS[:, :] = numpy.load(InFile('embeddings.npy' + suffix))

    trails = pickle.load(InFile('trails'))
    for k, v in trails.iteritems():
      TRAILS[k] = v

    net_values = dict(pickle.load(InFile('net.pkl' + suffix)))
    for v in ALL_VARS:
      if v.name in net_values:
        SESS.run(v.assign(net_values[v.name]))
    #if len(net_values) != len(ALL_VARS):
    #  import IPython; IPython.embed()
    #assert len(net_values) == len(ALL_VARS)
    #for (name, val), var in zip(net_values, ALL_VARS):
    #  assert name == var.name
    #  SESS.run(var.assign(val))

  def evaluate(j, ii):
    test_auc = TEST_EVALUATOR.calculate_accuracy(SESS, NN, EMBEDDINGS)
    train_auc = TRAIN_EVALUATOR.calculate_accuracy(SESS, NN, EMBEDDINGS)
    TRAILS['test_auc'].append(test_auc)
    TRAILS['train_auc'].append(train_auc)
    TRAILS['i_list'].append(ii)
    TRAILS['j_list'].append(j)

    if train_auc > TRAILS['best_train_auc']:
      TRAILS['best_train_auc'] = train_auc
      TRAILS['best_test_auc'] = test_auc
      save('.best')

    msg = '@%i (%i) test/train Best=%f/%f cur=%f/%f. - %s' % (
        j, ii, TRAILS['best_test_auc'], TRAILS['best_train_auc'], test_auc,
        train_auc, Description())

    print( msg)

  def save(suffix=''):
    if verbose:
      print ('Saving %s ...' % suffix)
    numpy.save(OutFile('embeddings.npy' + suffix), EMBEDDINGS)
    all_vals = SESS.run(ALL_VARS)
    pickle.dump(zip([v.name for v in ALL_VARS], all_vals),
                 OutFile('net.pkl' + suffix))
    pickle.dump(TRAILS, OutFile('trails' + suffix))
    if verbose:
      print ('Saved!')

  def train_on_pairs(j, pairs, lr_embedding=-1):
    batch_size = BATCH_SIZE
    sum_percent_change = 0
    num_percent_change = 0
    for i in range(0, len(pairs), batch_size):
      epoch = 2 * (j + float(i) / len(pairs))  # Approximation of the epoch.
      ii = i / batch_size
      end_i = min(i + batch_size, len(pairs))
      if end_i - i < 10:
        continue
      left_e = EMBEDDINGS[pairs[i:end_i, 0]]
      right_e = EMBEDDINGS[pairs[i:end_i, 1]]

      labels = numpy.ones(shape=(len(left_e)), dtype='float32')

      negatives = TRAIN_NEGATIVES.sample(pairs[i:end_i, 0], NCE_K)
      for k in range(NCE_K):
        left_e = numpy.concatenate(
            [left_e, EMBEDDINGS[pairs[i:end_i, 0]]], 0)
        right_e = numpy.concatenate(
            [right_e, EMBEDDINGS[negatives[:, k]]], 0)
        labels = numpy.concatenate(
            [labels, numpy.zeros(shape=len(negatives), dtype='float32')], 0)

      # Calculate gradients. grads[0] and grads[1] correspond to gradients for
      # the anchors and (positives,negatives), respectively. The remaining
      # grads[2:] correspond to gradients of Neural Network (NN)
      grads, loss = NN.get_gradients(SESS, left_e, right_e, labels)

      # Step on NN.
      nn_deltas = NN.apply_gradients(SESS, grads[2:], epoch)

      # Step on Embeddings. First, sum-up anchor (i.e. "left" gradients).
      embedding_l_grads = grads[0]
      sum_embedding_l_grad = embedding_l_grads[:len(negatives)]
      for k in range(NCE_K):
        s_j = (k + 1) * len(negatives)
        e_j = s_j + len(negatives)
        sum_embedding_l_grad += embedding_l_grads[s_j:e_j]

      # Determine Learning rate using PercentDelta (Abu-El-Haija, 2017).
      mean_percent_grad = numpy.mean(numpy.abs(
          sum_embedding_l_grad / (edge_nn.PlusEpsilon(
              EMBEDDINGS[pairs[i:end_i, 0], :]))))
      lr_embedding = edge_nn.PickLearnRate(
          mean_percent_grad, 300 * (j + float(i) / len(pairs)))

      delta_embeddings = lr_embedding * sum_embedding_l_grad
      if verbose:
        mean_percent_change = numpy.mean(
            abs(delta_embeddings /
                edge_nn.PlusEpsilon(EMBEDDINGS[pairs[i:end_i, 0], :])))
        print (f'Mean percent grad={mean_percent_grad}, percent change = {mean_percent_change}'  )
      
      # Update embeddings.
      EMBEDDINGS[pairs[i:end_i, 0], :] -= delta_embeddings

      if ii % save_every == 0:
        evaluate(j, ii)  # Populates TRAILS.
        TRAILS['embedding_lrs'].append(lr_embedding)
        save()

  if restore == 1:
    restore()

  while TRAILS['j'] < num_iterations:
    train_on_pairs(TRAILS['j'], TRAIN_PAIRS.next_pairs_array())
    TRAILS['j'] += 1
    save()

if __name__ == '__main__':
  check_args()
  main(num_nodes)
  # app.run()
