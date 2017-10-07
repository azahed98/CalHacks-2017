import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

#Parameters
learning_rate = 0.01
num_epochs = 20
batch_size = 256
display_step = 20
n_input = 51
num_features = 35

input_series = tf.placeholder("int", [None, n_input])
clean_series = tf.placeholder("int", [None, n_input])

def encoder_uni_dir(num_units = 200):
	"""
	returns the lstm and states
	"""
	lstm = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple = True)

	hidden_state = tf.zeros([batch_size, lstm.state_size])
	current_state = tf.zeros([batch_size, lstm.state_size])
	state = hidden_state, current_state

	return lstm, state


def decoder(num_units = input_size):
	# Decoder Hidden layer with sigmoid activation #1
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple = True)

    hidden_state = tf.zeros([batch_size, lstm.state_size])
	current_state = tf.zeros([batch_size, lstm.state_size])
	state = hidden_state, current_state

	return lstm, state


#Construct model
encoder_op, hidden_encoder, cur_encoder = encoder_uni_dir(num_features)
decoder_op, hidden_decoder, cur_decoder = decoder(num_frequencies)

encoder_outputs_series, encoder_current_state = encoder_op(input_series)
decoder_outputs_series, decoder_current_state = decoder_op(encoder_output_series)
#Prediction
y = decoder_outputs_series

#Targets (Labels) are the input data

#Define loss and optimizer, minimal square error
cost = tf.reduce_mean(tf.pow(clean_series- y, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver(tf.train.SaveDef.V1)
# Initializing the variables
init = tf.global_variables_initializer()

errors = np.array([])
#Launch the graph
with tf.Session() as sess:
    sess.run(init)	
    total_batch = #int(mnist.train.num_examples/batch_size)
    plt.ion()
    #Train
    for epoch in range(num_epochs):
        #Loop over batches
        for i in range(total_batch):
        	batch_input, batch_ys, batch_cleans = 

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={input_series: batch_xs, clean_series})
        # Display logs per epoch step
            if epoch % display_step == 0:
            	errors = errors.append(c)
            	plt.figure()
                plt.plot(errors)
                plt.pause(0.5)
                print("Epoch:", '%04d' % (epoch+1),
                "cost=", "{:.9f}".format(c))


    print("Optimization Finished!")

    save_path = saver.save(sess, 'uni_directional.ckpt')
    print("Model saved in file: %s" % save_path)

   