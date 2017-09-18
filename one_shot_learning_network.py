import tensorflow as tf
from tensorflow.python.ops.nn_ops import max_pool
from tensorflow.python.ops import rnn_cell

class BidirectionalLSTM:
    """
    embedding training examples, function g
    """
    def __init__(self, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
             f                                                                                           neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = True
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes[0]

    def __call__(self, inputs, name, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        from future import static_bidirectional_rnn
        with tf.name_scope('bid-lstm' + name), tf.variable_scope('bid-lstm'):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()

            fw_lstm_cell = rnn_cell.LSTMCell(num_units=self.layer_sizes, activation=tf.nn.tanh)
            bw_lstm_cell = rnn_cell.LSTMCell(num_units=self.layer_sizes, activation=tf.nn.tanh)

            outputs, output_state_fw, output_state_bw = static_bidirectional_rnn(
                fw_lstm_cell,
                bw_lstm_cell,
                inputs,
                dtype=tf.float32
            )
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bid-lstm')
        return outputs, output_state_fw, output_state_bw



class DistanceNetwork:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        """
        This module calculates the cosine/euclidean distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """

        with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module'):
            if self.reuse:
                tf.variable_scope().reuse_variables()
            similarities = []
            for support_image in tf.unpack(support_set, axis=1):
                if name == "cosine":
                    eps = 1e-10
                    sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
                    support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
                    dot_product = tf.batch_matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
                    dot_product = tf.squeeze(dot_product, [1, ])
                    cosine_similarity = dot_product * support_magnitude
                    similarities.append(cosine_similarity)
                elif name == "euclidean":
                    euc_similarity = tf.reduce_sum(tf.square(support_image-input_image), 1, keep_dims = True)
                    similarities.append(-euc_similarity)
                else:
                    print("Unsupported distance type.")
                    assert False

        similarities = tf.concat(concat_dim=1, values=similarities)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')

        return similarities


class AttentionalClassify:
    def __init__(self):
        self.reuse = False

    def __call__(self, similarities, support_set_y, name, training=False):
        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size, 1]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :param name: The name of the op to appear on tf graph
        :param training: Flag indicating training or evaluation stage (True/False)
        :return: Softmax pdf
        """
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification'):
            if self.reuse:
                tf.variable_scope().reuse_variables()
            softmax_similarities = tf.nn.softmax(similarities)
            preds = tf.squeeze(tf.batch_matmul(tf.expand_dims(softmax_similarities, 1), support_set_y))

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds

class ClassEmbedClassify:
    """Class Embeddings"""
    def __init__(self):
        self.reuse = False
    def __call__(self, similarities):
        with tf.name_scope('classembed-classification'), tf.variable_scope('classembed-classification'):
            if self.reuse:
                tf.variable_scope().reuse_variables()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        preds = None
        return preds


class OneshotNetwork:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob,
                 batch_size=100, num_channels=1, train_time=True, is_training=False, learning_rate=0.001, rotate_flag=False, fce=False, classes_train = 2, classes_test = 5,
                 num_samples_per_class=1, num_queries_per_class = 1,  network_name = "MN"):

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, n_samples, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, n_samples, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_queries, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, n_queries]
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.train_time = train_time
        self.batch_size = batch_size
        self.fce = fce

        self.classes_train = classes_train
        self.classes_test = classes_test
        self.num_samples_per_class = num_samples_per_class
        self.num_queries_per_class = num_queries_per_class


        self.g = Classifier(self.batch_size, num_channels=num_channels, layer_sizes=[64, 64, 64 ,64])
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size)
        self.dn = DistanceNetwork()

        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.target_image = target_image
        self.target_label = target_label

        self.keep_prob = keep_prob
        self.is_training = is_training
        self.k = None
        self.rotate_flag = rotate_flag

        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate
        self.network_name = network_name

        if self.network_name == "MN":
            self.classify = AttentionalClassify()
        elif self.network_name =="PN":
            self.classify = ClassEmbedClassify()
        else:
            print("Network Unsupported.")
            assert False


    def loss(self):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """

        with tf.name_scope("losses"):
            self.flag = 0
            def fn2():
                self.flag += 1
                self.n_samples = self.classes_test*self.num_samples_per_class
                self.n_queries = self.classes_test*self.num_queries_per_class
                return tf.constant(1)

            def fn1():
                self.flag += 1
                self.n_samples = self.classes_train*self.num_samples_per_class
                self.n_queries = self.classes_train*self.num_queries_per_class
                return tf.constant(0)
            _ = tf.cond(self.train_time, fn1, fn2)

            if self.flag>1:
                self.n_samples = self.classes_train*self.num_samples_per_class
                self.n_queries = self.classes_train*self.num_queries_per_class

            nclasses = self.n_samples/self.num_samples_per_class
            support_set_labels = tf.one_hot(self.support_set_labels, nclasses)  # one hot encode
            support_set_labels = tf.slice(support_set_labels, [0,0,0], [-1, self.n_samples, -1])

            encoded_images = []
            for image in tf.unpack(self.support_set_images, axis=1)[:self.n_samples]:
                #produce embeddings for support set images
                gen_encode = self.g(image_input=image, training=self.is_training, keep_prob=self.keep_prob)
                encoded_images.append(gen_encode)
            encoded_images = tf.pack(encoded_images, axis = 1)
            preds = []

            if self.network_name == "MN":
                for image in tf.unpack(self.target_image, axis=1):  #produce embeddings for support set images
                    single_target = self.g(image_input=image, training=self.is_training, keep_prob=self.keep_prob)
                    similarities = self.dn(support_set=encoded_images, input_image=single_target, name="cosine",
                                           training=self.is_training)
                    single_pred = self.classify(similarities, support_set_y=self.support_set_labels, name='classify', training=self.is_training)
                    preds.append(single_pred)

            elif self.network_name == "PN":
                support_image = tf.expand_dims(encoded_images, 3) #[bs, sl, 64,1]
                support_set_labels = tf.expand_dims(support_set_labels, 2) #[bs, sl, 1, nc]

                class_embedding = tf.reduce_sum(tf.batch_matmul(support_image, support_set_labels), 1) #[bs, sl, 64, nc]
                class_embedding = class_embedding/self.num_samples_per_class
                class_embedding = tf.transpose(tf.squeeze(class_embedding), perm = [0, 2, 1]) #[bs, nc, 64]
                for image in tf.unpack(self.target_image, axis=1)[:nclasses]:  #produce embeddings for support set images
                    single_target = self.g(image_input=image, training=self.is_training, keep_prob=self.keep_prob)
                    single_pred = self.dn(support_set=class_embedding, input_image=single_target, name="euclidean", training=self.is_training)
                    preds.append(single_pred)

            else:
                print("Network Unsupported.")
                assert False

            target_label = tf.slice(self.target_label, [0,0], [-1, nclasses])
            preds = tf.pack(preds, axis = 1)
            correct_prediction = tf.equal(tf.argmax(preds, 2), tf.cast(target_label, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            crossentropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_label,
                                                                                              logits=preds))
            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

        return {
            self.classify: tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss'),
            self.dn: tf.add_n(tf.get_collection('accuracy'), name='accuracy')
        }

    def train(self, losses):

        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """
        c_opt = tf.train.AdamOptimizer(beta1=0.9, learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):  # Needed for correct batch norm usage
            if self.fce:
                train_variables = self.lstm.variables + self.g.variables
            else:
                train_variables = self.g.variables
            c_error_opt_op = c_opt.minimize(losses[self.classify],
                                            var_list=train_variables)

        return c_error_opt_op

    def init_train(self):
        """
        Get all ops, as well as all losses.
        :return:
        """
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        summary = tf.merge_all_summaries()
        return  summary, losses, c_error_opt_op


class Classifier:
    def __init__(self, batch_size, layer_sizes, num_channels=1):
        """
        Builds a CNN to produce embeddings(for training, before BidirectionalLSTM)
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes)==4, "layer_sizes should be a list of length 4"

    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """

        with tf.variable_scope('g'):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('conv_layers'):
                input_channel= image_input.get_shape().as_list()[-1]
                w0 = tf.get_variable('w0', [3,3,input_channel, self.layer_sizes[0]])
                w1 = tf.get_variable('w1', [3,3,self.layer_sizes[0], self.layer_sizes[1]])
                w2 = tf.get_variable('w2', [3,3,self.layer_sizes[1], self.layer_sizes[2]])
                w3 = tf.get_variable('w3', [2,2,self.layer_sizes[2], self.layer_sizes[3]])
                with tf.variable_scope('g_conv1'):
                    g_conv1_encoder = tf.nn.conv2d(image_input, w0, strides=[1,1,1,1], padding='VALID')
                    g_conv1_encoder = tf.nn.relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, is_training=training)
                    g_conv1_encoder = max_pool(g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv2'):
                    g_conv2_encoder = tf.nn.conv2d(g_conv1_encoder, w1, strides=[1,1,1,1], padding='VALID')
                    g_conv2_encoder = tf.nn.relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, is_training=training, reuse=None)
                    g_conv2_encoder = max_pool(g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv2_encoder = tf.nn.dropout(g_conv2_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv3'):
                    g_conv3_encoder = tf.nn.conv2d(g_conv2_encoder, w2, strides=[1,1,1,1],padding='VALID')
                    g_conv3_encoder = tf.nn.relu(g_conv3_encoder, name='outputs')
                    g_conv3_encoder = tf.contrib.layers.batch_norm(g_conv3_encoder, is_training=training, reuse=None)

                    g_conv3_encoder = max_pool(g_conv3_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv3_encoder = tf.nn.dropout(g_conv3_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv4'):
                    g_conv4_encoder = tf.nn.conv2d(g_conv3_encoder, w3, strides=[1,1,1,1] ,padding='VALID')
                    g_conv4_encoder = tf.nn.relu(g_conv4_encoder, name='outputs')
                    g_conv4_encoder = tf.contrib.layers.batch_norm(g_conv4_encoder, is_training=training, reuse = None)
                    g_conv4_encoder = max_pool(g_conv4_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv4_encoder = tf.nn.dropout(g_conv4_encoder, keep_prob=keep_prob)

            g_conv_encoder = g_conv4_encoder
            g_conv_encoder = tf.contrib.layers.flatten(g_conv_encoder)

        #
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return g_conv_encoder

