import inspect
import os

import numpy as np
import tensorflow as tf
import time
import h5py

VGG_MEAN = [103.939, 116.779, 123.68]

"""
Based on https://github.com/antlerros/tensorflow-fast-neuralstyle
Download weigths from:

https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM

or from

https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
"""

class Vgg16:
    def __init__(self, vgg16_npy_path, h, w, channels=3, train=False):
        path = inspect.getfile( Vgg16 )
        path = os.path.abspath( os.path.join( path, os.pardir ) )
        path = os.path.join( path, vgg16_npy_path )
        vgg16_path = path
        
        print("Loading weights from %s ..."%vgg16_path)

        self.data_dict = None
        if vgg16_npy_path.split(".")[-1] == "h5":
            self.h5 = True
            self.data_dict = h5py.File(vgg16_path, 'r')
            print( "H5 file loaded" )
        else:
            self.h5 = False
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()
            print( "Npy file loaded" )

        self.h = h
        self.w = w
        self.ch = channels
        
        self.is_trainable = train
        
    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("Build model started")
        
        """
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """

        self.conv1_1 = self.conv_layer(rgb, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        """
        # no need for fully connected layers
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")
        """

        #self.data_dict = None
        print("Build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def from_dataset( self, name, column ):
        if self.h5:
            name = 'block%d_conv%d' % (int( name.split( "_" )[ 0 ][ -1 ] ), int( name[ -1 ] ))
            data = list(self.data_dict[ name ].items())[ column ][ 1 ][()]
            
            return data
        
        data = self.data_dict[ name ][ column ]
        return data

    def get_conv_filter(self, name):
        return tf.constant(self.from_dataset(name, 0), name="filter")

    def get_bias(self, name):
        return tf.constant(self.from_dataset(name, 1), name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.from_dataset(name, 0), name="weights")

    def get_layer_tensors( self, layers, enum ):
        '''
        Only to get all the names of all tensors to be sure!!
        vgg_layers_tensors = [ tensor.name for tensor in tf.get_default_graph().as_graph_def().node ]
        '''
        
        nl = list()
        for i in range( len( layers ) ):
            nl.append( layers[ i ].split( "/" ) )
            nl[ i ].insert( 1, "p" + str( enum ) )
            nl[ i ] = "/".join( nl[ i ] )
            
        return [tf.get_default_graph().get_tensor_by_name( name ) for name in nl]