import tensorflow as tf
from tensorflow.python.ops.nn_ops import max_pool
from tensorflow.python.ops import rnn_cell, rnn

class ClassEmbedding(object):
    """Class Embeddings"""
    def __init__(self):
        self.reuse = False        
    def __call__(self, support_image, support_set_labels, num_samples_per_class, name, training = False):
        with tf.name_scope('class-embedding' + name), tf.variable_scope('class-embedding'):
            support_image = tf.expand_dims(support_image, 3) #[bs, sl, 64]
            support_set_labels = tf.expand_dims(support_set_labels, 2) #[bs, sl, nc]
            class_embedding = tf.reduce_mean(tf.batch_matmul(support_image, support_set_labels), 1) #[bs, sl, 64, nc]
            class_embedding = tf.transpose(tf.squeeze(class_embedding), perm = [2, 0, 1]) #[nc, bs, 64]
        return class_embedding

class DistanceNetwork_Euclidean:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        """
        This module calculates the cosine distance between each of the support set embeddings(C_k) and the target
        image embeddings.
        """
        """
        :param support_set: The embeddings of the support set images(class embeddings), tensor of shape [sequence_length, batch_size, 64]
        -->[nc,bs,64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with Euclidean similarities of shape [batch_size, nc]
        """
        with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module'):
            similarities = []
            for support_image in tf.unpack(support_set, num=None):
                euc_similarity = tf.reduce_sum(tf.square(support_image-input_image), 1, keep_dims = True)
                similarities.append(euc_similarity)
        similarities = tf.concat(concat_dim=1, values=similarities)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
        return similarities

class PrototypicalNet_cifar:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob,
                 batch_size=100, num_channels=3, is_training=False, learning_rate=0.001, rotate_flag=False, fce=False, 
                 num_classes_per_set=5,
                 num_samples_per_class=1):
        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, sequence_size, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        # mentioned but seems not better than bilstm
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.batch_size = batch_size
        self.fce = fce 
        # self.g = Classifier_cifar_PrototypicalNet(self.batch_size, num_channels=num_channels, layer_sizes=[64, 64, 64 ,64])
        self.g = Classifier_cifar_PrototypicalNet(self.batch_size, num_channels=num_channels, layer_sizes=[16, 16, 16 ,16])

        self.dn = DistanceNetwork_Euclidean()
        # self.classify = AttentionalClassify_prototypical()
        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.target_image = target_image
        self.target_label = target_label
        self.keep_prob = keep_prob
        self.is_training = is_training
        # self.k = None
        self.rotate_flag = rotate_flag
        # self.num_classes_per_set = num_classes_per_set
        self.num_classes_per_set = self.support_set_images.get_shape().as_list()[1]

        self.num_samples_per_class = num_samples_per_class

        self.learning_rate = learning_rate
        self.class_encode = ClassEmbedding()

    def loss(self):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        with tf.name_scope("losses"):
            self.support_set_labels = tf.one_hot(self.support_set_labels, self.num_classes_per_set, axis=-1)  # one hot encode
            encoded_images = []
            for image in tf.unpack(self.support_set_images, axis=1):  #produce embeddings for support set images
                gen_encode = self.g(image_input=image, training=self.is_training, keep_prob=self.keep_prob)
                encoded_images.append(gen_encode)

            encode_image_pack = tf.pack(encoded_images, axis = 1)
            class_encode = self.class_encode(support_image = encode_image_pack, support_set_labels = self.support_set_labels,\
            name = "class_encode", num_samples_per_class = self.num_samples_per_class)


            for image in tf.unpack(self.target_image, axis = 1):
                target_single_encode = self.g(image_input=image, training=False, keep_prob=self.keep_prob)
                similarity = 
                target_encode.append(temp)

            for 
            
            # target_encode = tf.pack(target_encode, axis = 1)
            # # HERE

            # target_encode = self.g(image_input=self.target_image, training=self.is_training, keep_prob=self.keep_prob)
            # similarities = self.dn(support_set=class_encode, input_image=target_encode, name="distance_calculation",
            #                        training=self.is_training)  #get similarity between support set embeddings and target

            # preds = tf.nn.softmax(similarities) # produce predictions for target probabilities
            # # (bs, query, nc)
            # correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label, tf.int64))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            crossentropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_label,
                                                                                              logits=similarities))
            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

        return {
            "crossentropy_loss": tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss'),
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
            train_variables = self.g.variables
            c_error_opt_op = c_opt.minimize(losses["crossentropy_loss"],
                                            var_list=train_variables)
        return c_error_opt_op

    def init_train(self):
        """
        Get all ops, as well as all losses.
        :return:
        """
        summary = tf.merge_all_summaries()
        losses = self.loss()
        c_error_opt_op = self.train(losses)
        return summary, losses, c_error_opt_op


class Classifier_cifar_PrototypicalNet:
    def __init__(self, batch_size, layer_sizes, num_channels=3):
        """
        Builds a CNN to produce embeddings
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
        :param image_input: Image input to produce embeddings for. [batch_size, 32, 32, 1/3]
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
                # w1 = tf.get_variable('w1', [3,3,self.layer_sizes[0], self.layer_sizes[1]])
                # w2 = tf.get_variable('w2', [3,3,self.layer_sizes[1], self.layer_sizes[2]])
                # w3 = tf.get_variable('w3', [3,3,self.layer_sizes[2], self.layer_sizes[3]])
                w1 = tf.get_variable('w1', [3,3,self.layer_sizes[0]])
                w2 = tf.get_variable('w2', [3,3,self.layer_sizes[0]])
                w3 = tf.get_variable('w3', [3,3,self.layer_sizes[0]])

                with tf.variable_scope('g_conv1'):
                    g_conv1_encoder = tf.nn.conv2d(image_input, w0, strides=[1,1,1,1], padding='SAME')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, is_training=training)
                    g_conv1_encoder = tf.nn.relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = max_pool(g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv2'):
                    g_conv2_encoder = tf.nn.dilation2d(g_conv1_encoder, w1, strides=[1,1,1,1], padding='SAME', rates=[1,2,2,1])
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, is_training=training, reuse=None)
                    g_conv2_encoder = tf.nn.relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = max_pool(g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv2_encoder = tf.nn.dropout(g_conv2_encoder, keep_prob=keep_prob)


                with tf.variable_scope('g_conv3'):
                    g_conv3_encoder = tf.nn.dilation2d(g_conv2_encoder, w2, strides=[1,1,1,1],padding='SAME', rates=[1,2,2,1])
                    g_conv3_encoder = tf.contrib.layers.batch_norm(g_conv3_encoder, is_training=training, reuse=None)
                    g_conv3_encoder = tf.nn.relu(g_conv3_encoder, name='outputs')
                    g_conv3_encoder = max_pool(g_conv3_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv3_encoder = tf.nn.dropout(g_conv3_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv4'):
                    g_conv4_encoder = tf.nn.dilation2d(g_conv3_encoder, w3, strides=[1,1,1,1] ,padding='SAME', rates=[1,2,2,1])
                    g_conv4_encoder = tf.contrib.layers.batch_norm(g_conv4_encoder, is_training=training, reuse = None)
                    g_conv4_encoder = tf.nn.relu(g_conv4_encoder, name='outputs')
                    g_conv4_encoder = max_pool(g_conv4_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv4_encoder = tf.nn.dropout(g_conv4_encoder, keep_prob=keep_prob)

            g_conv_encoder = g_conv4_encoder
            g_conv_encoder = tf.contrib.layers.flatten(g_conv_encoder)
            tf.get_variable_scope().reuse_variables()


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return g_conv_encoder

