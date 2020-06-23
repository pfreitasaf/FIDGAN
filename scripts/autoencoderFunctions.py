import data_utils
import utils
import json
import numpy as np
import tensorflow as tf
from mod_core_rnn_cell_impl import LSTMCell
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from time import time

# Helper Functions -----------------------------------------------------------------------------------------------------
def loadParameters(identifier):

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # Load parameters from a numpy file
    model_parameters = np.load(identifier).item()
    # restore np.load for future normal usage
    np.load = np_load_old
    return model_parameters

def dumpParameters(identifier, sess):
    # Save model parmaters to a numpy file
    dump_path = identifier + '.npy'
    model_parameters = dict()
    for v in tf.compat.v1.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    np.save(dump_path, model_parameters)
    print('Recorded', len(model_parameters), 'parameters to', dump_path)
    return True
#-----------------------------------------------------------------------------------------------------------------------

# Model Functions ------------------------------------------------------------------------------------------------------
def encoderModel(z, hidden_units, seq_length, batch_size, latent_dim, reuse=False, parameters=None):
    with tf.compat.v1.variable_scope('encoder') as scope:
        if(parameters != None):
            W_initializer = tf.compat.v1.constant_initializer(value=parameters['encoder/W:0'])
            b_initializer = tf.compat.v1.constant_initializer(value=parameters['encoder/b:0'])
            lstm_initializer = tf.compat.v1.constant_initializer(value=parameters['encoder/rnn/lstm_cell/weights:0'])
            bias_start = parameters['encoder/rnn/lstm_cell/biases:0']
            W = tf.compat.v1.get_variable(name='W', shape=[hidden_units, latent_dim], initializer=W_initializer, trainable=False)
            b = tf.compat.v1.get_variable(name='b', shape=latent_dim, initializer=b_initializer, trainable=False)
        
        else:
            W_initializer = tf.compat.v1.truncated_normal_initializer()
            b_initializer = tf.compat.v1.truncated_normal_initializer()
            lstm_initializer = None
            bias_start = 1.0
            W = tf.compat.v1.get_variable(name='W', shape=[hidden_units, latent_dim], initializer=W_initializer, trainable=True)
            b = tf.compat.v1.get_variable(name='b', shape=latent_dim, initializer=b_initializer, trainable=True)

        inputs = z
        cell = LSTMCell(num_units=hidden_units, state_is_tuple=True, initializer=lstm_initializer, bias_start=bias_start, reuse=reuse)
        rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(cell=cell,dtype=tf.float32, sequence_length=[seq_length] * batch_size, inputs=inputs)
        rnn_outputs_2d = tf.compat.v1.reshape(rnn_outputs, [-1, hidden_units])
        logits_2d = tf.compat.v1.matmul(rnn_outputs_2d, W) + b #out put weighted sum
        # output_2d = tf.nn.relu(logits_2d) # logits operation [-1, 1]
        # output_2d = tf.compat.v1.nn.tanh(logits_2d)
        output_3d = tf.compat.v1.reshape(logits_2d, [-1, seq_length, latent_dim])
    return output_3d

def encoderModel1(z, hidden_units, seq_length, batch_size, latent_dim, reuse=False, parameters=None):
    with tf.compat.v1.variable_scope('encoder') as scope:
        if(parameters != None):
            W_initializer = tf.compat.v1.constant_initializer(value=parameters['encoder/W:0'])
            b_initializer = tf.compat.v1.constant_initializer(value=parameters['encoder/b:0'])
            # lstm_initializer = tf.compat.v1.constant_initializer(value=parameters['encoder/rnn/lstm_cell/weights:0'])
            # bias_start = parameters['encoder/rnn/lstm_cell/biases:0']
            W = tf.compat.v1.get_variable(name='W', shape=[80, 150], initializer=W_initializer, trainable=False)
            b = tf.compat.v1.get_variable(name='b', shape=150, initializer=b_initializer, trainable=False)
        
        else:
            W_initializer = tf.compat.v1.truncated_normal_initializer()
            b_initializer = tf.compat.v1.truncated_normal_initializer()
            # lstm_initializer = None
            # bias_start = 1.0
            W = tf.compat.v1.get_variable(name='W', shape=[80,150], initializer=W_initializer, trainable=True)
            b = tf.compat.v1.get_variable(name='b', shape=150, initializer=b_initializer, trainable=True)

        inputs = z
        # cell = LSTMCell(num_units=hidden_units, state_is_tuple=True, initializer=lstm_initializer, bias_start=bias_start, reuse=reuse)
        # rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(cell=cell,dtype=tf.float32, sequence_length=[seq_length] * batch_size, inputs=inputs)
        rnn_outputs_2d = tf.compat.v1.reshape(inputs, [-1, 80])
        logits_2d = tf.compat.v1.matmul(rnn_outputs_2d, W) + b #out put weighted sum
        
        output_2d = tf.compat.v1.nn.tanh(logits_2d)
        output_3d = tf.compat.v1.reshape(output_2d, [-1, seq_length, latent_dim])
    return output_3d

def generatorModel(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None):
    with tf.compat.v1.variable_scope('generator') as scope:
        W_out_G_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/W_out_G:0'])
        b_out_G_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/b_out_G:0'])
        lstm_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
        bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.compat.v1.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.compat.v1.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)

        inputs = z
        cell = LSTMCell(num_units=hidden_units_g, state_is_tuple=True, initializer=lstm_initializer, bias_start=bias_start, reuse=reuse)
        rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(cell=cell,dtype=tf.float32, sequence_length=[seq_length] * batch_size, inputs=inputs)
        rnn_outputs_2d = tf.compat.v1.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.compat.v1.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_3d = tf.compat.v1.reshape(logits_2d, [-1, seq_length, num_generated_features])
        output_2d = tf.compat.v1.nn.tanh(logits_2d)
        output_3d_l = tf.compat.v1.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d, output_3d_l

# ---------------------
def encoderGeneratorInvertModel(settings, samples, para_path, g_tolerance=None, e_tolerance=0.1, n_iter=None, max_iter=10000, heuristic_sigma=None):
    samples = np.float32(samples)

    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    parameters = loadParameters(para_path)

    # create VARIABLE Z
    Z = tf.get_variable(name='Z', shape=[1, settings['seq_length'], settings['latent_dim']], initializer=tf.random_normal_initializer())
    # create outputs
    G_samples = generatorModel(Z, settings['hidden_units_g'], settings['seq_length'], 1, settings['num_generated_features'], reuse=False, parameters=parameters)

    fd = None

    # define loss mmd-based loss
    if heuristic_sigma is None:
        heuristic_sigma = mmd.median_pairwise_distance_o(samples)  # this is noisy
        print('heuristic_sigma:', heuristic_sigma)
    samples = tf.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])
    Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
    similarity_per_sample = tf.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    similarity = tf.reduce_mean(similarity_per_sample)
    reconstruction_error = 1 - similarity
    solver = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(reconstruction_error_per_sample, var_list=[Z])

    grad_Z = tf.gradients(reconstruction_error_per_sample, Z)[0]
    grad_per_Z = tf.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.reduce_mean(grad_per_Z)
    print('Finding latent state corresponding to samples...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        error = sess.run(reconstruction_error, feed_dict=fd)
        g_n = sess.run(grad_norm, feed_dict=fd)
        i = 0
        if not n_iter is None:
            while i < n_iter:
                _ = sess.run(solver, feed_dict=fd)
                error = sess.run(reconstruction_error, feed_dict=fd)
                i += 1
        else:
            if not g_tolerance is None:
                while g_n > g_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error, g_n = sess.run([reconstruction_error, grad_norm], feed_dict=fd)
                    i += 1
                    print(error, g_n)
                    if i > max_iter:
                        break
            else:
                while np.abs(error) > e_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error = sess.run(reconstruction_error, feed_dict=fd)
                    i += 1
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        Gs = sess.run(G_samples, feed_dict={Z: Zs})
        error_per_sample = sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.reset_default_graph()

    return Gs, Zs, error_per_sample, heuristic_sigma
# ---------------------


def discriminatorModel(x, hidden_units_d, reuse=False, parameters=None):
    with tf.compat.v1.variable_scope("discriminator") as scope:
        W_out_D_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/W_out_D:0'])
        b_out_D_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/b_out_D:0'])
        W_out_D = tf.compat.v1.get_variable(name='W_out_D', shape=[hidden_units_d, 1],  initializer=W_out_D_initializer)
        b_out_D = tf.compat.v1.get_variable(name='b_out_D', shape=1, initializer=b_out_D_initializer)
        lstm_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/rnn/lstm_cell/weights:0'])
        bias_start = parameters['discriminator/rnn/lstm_cell/biases:0']

        inputs = x
        cell = LSTMCell(num_units=hidden_units_d, state_is_tuple=True, initializer=lstm_initializer, bias_start=bias_start, reuse=reuse)
        rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)
        logits = tf.compat.v1.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
        output = tf.compat.v1.nn.sigmoid(logits)
    return output, logits

def discriminatorModelPred(x, hidden_units_d, reuse=False, parameters=None):
    with tf.compat.v1.variable_scope("discriminator_pred") as scope:
        W_out_D_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/W_out_D:0'])
        b_out_D_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/b_out_D:0'])
        W_out_D = tf.compat.v1.get_variable(name='W_out_D', shape=[hidden_units_d, 1],  initializer=W_out_D_initializer)
        b_out_D = tf.compat.v1.get_variable(name='b_out_D', shape=1, initializer=b_out_D_initializer)
        lstm_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/rnn/lstm_cell/weights:0'])
        bias_start = parameters['discriminator/rnn/lstm_cell/biases:0']

        inputs = x
        cell = LSTMCell(num_units=hidden_units_d, state_is_tuple=True, initializer=lstm_initializer, bias_start=bias_start, reuse=reuse)
        rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)
        logits = tf.compat.v1.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
        output = tf.compat.v1.nn.sigmoid(logits)
    return output, logits
#-----------------------------------------------------------------------------------------------------------------------

# Anomaly Detection Functions ------------------------------------------------------------------------------------------
def discriminatorTrainedModel(settings, samples, para_path):
    # Returns the discrimination results of num_samples testing samples from a trained model described by settings dict, for ONE sample discrimination
    
    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    num_samples = samples.shape[0]
    samples = np.float32(samples)
    num_variables = samples.shape[2]

    parameters = loadParameters(para_path)

    # Create the placeholder
    T = tf.compat.v1.placeholder(tf.float32, [num_samples, settings['seq_length'], num_variables])
    # create the discriminator (GAN)
    D_output, D_logits = discriminatorModel(T, settings['hidden_units_d'], reuse=False, parameters=parameters)

    # Run session
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        D_output, D_logits = sess.run([D_output, D_logits], feed_dict={T: samples})

    tf.compat.v1.reset_default_graph()
    return D_output, D_logits

def encoderGeneratorTrainedModels(settings, samples, para_path, para_path_autoencoder):
    # Return the latent space points corresponding to a set of a samples (from gradient descent), for ONE sample generation
    # time1 = time()
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    num_samples = samples.shape[0]
    samples = np.float32(samples)
    num_variables = samples.shape[2]

    # Load autoencoder parameters
    parameters_autoencoder = loadParameters(para_path_autoencoder)

    # Create the placeholder X
    X = tf.compat.v1.placeholder(tf.float32, [num_samples, settings['seq_length'], num_variables])
    # Create the Encoder
    E_outputs = encoderModel(X, settings['hidden_units_g'], settings['seq_length'], num_samples, settings['latent_dim'], reuse=False, parameters=parameters_autoencoder)
    # Create the Generator - Use para_path for the parameters trained in the GAN and parameters_autoencoder for the parameters trained in the autoencoder (if the generator is also trained in the autoencoder)
    G_output, G_logits = generatorModel(E_outputs, settings['hidden_units_g'], settings['seq_length'], num_samples, settings['num_generated_features'], reuse=False, parameters=parameters_autoencoder)
    # G_output = generatorModel(E_outputs, settings['hidden_units_g'], settings['seq_length'], num_samples, settings['num_generated_features'], reuse=False, parameters=parameters_autoencoder)
    # time2 = time1 - time()
    # print(time2)
    # Run session
    with tf.compat.v1.device('/gpu:0'):
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            G_output, G_logits = sess.run([G_output, G_logits], feed_dict={X: samples})
            # G_output = sess.run([G_output], feed_dict={X: samples})

    tf.compat.v1.reset_default_graph()
    return G_output, G_logits
    # return G_output
#-----------------------------------------------------------------------------------------------------------------------

# Anomaly Detection Score Functions ------------------------------------------------------------------------------------
def detection_Comb(Label_test, L_mb, I_mb, seq_step, tao):
    aa = Label_test.shape[0]
    bb = Label_test.shape[1]

    LL = (aa-1)*seq_step+bb

    Label_test = abs(Label_test.reshape([aa, bb]))
    L_mb = L_mb .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            D_L[i*seq_step+j] += Label_test[i, j]
            L_L[i * seq_step + j] += L_mb[i, j]
            Count[i * seq_step + j] += 1

    D_L /= Count
    L_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if D_L[i] > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

    cc = (D_L == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    Accu = float((N / LL) * 100)
    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')
    return Accu, precision, recall, f1


def sample_detection(D_test, L_mb, tao):
    # sample-wise detection for one dimension
    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    L = np.sum(L_mb, 1)
    L[L > 0] = 1

    D_L = np.empty([aa, ])

    for i in range(aa):
        if np.mean(D_test[i, :]) > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

    cc = (D_L == L)
    N = list(cc).count(True)
    Accu = float((N / (aa)) * 100)
    precision, recall, f1, _ = precision_recall_fscore_support(L, D_L, average='binary')
    return Accu, precision, recall, f1

def detection_D_I(DD, L_mb, I_mb, seq_step, tao):
    # point-wise detection for one dimension
    aa = DD.shape[0]
    bb = DD.shape[1]

    LL = (aa-1)*seq_step+bb

    DD = abs(DD.reshape([aa, bb]))
    L_mb = L_mb .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            D_L[i*10+j] += DD[i, j]
            L_L[i * 10 + j] += L_mb[i, j]
            Count[i * 10 + j] += 1

    D_L /= Count
    L_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(LL):
        if D_L[i] > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

        A = D_L[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1

    cc = (D_L == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    print('N:', N)
    Accu = (TP+TN)*100/(TP+TN+FP+FN)
    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN + 1)

    return Accu, Pre, Rec, F1, FPR, D_L

def detection_R_D_I(DD, Gs, T_mb, L_mb, seq_step, tao, lam):
    # point-wise detection for one dimension
    # (1-lambda)*R(x)+lambda*D(x)
    # lambda=0.5?
    # D_test, Gs, T_mb, L_mb  are of same size

    R = np.absolute(Gs - T_mb)
    R = np.mean(R, axis=2)
    aa = DD.shape[0]
    bb = DD.shape[1]

    LL = (aa - 1) * seq_step + bb

    DD = abs(DD.reshape([aa, bb]))
    DD = 1-DD
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    D_L = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    L_pre = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i * 10 + j] += DD[i, j]
            L_L[i * 10 + j] += L_mb[i, j]
            R_L[i * 10 + j] += R[i, j]
            Count[i * 10 + j] += 1
    D_L /= Count
    L_L /= Count
    R_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if (1-lam)*R_L[i] + lam*D_L[i] > tao:
            # false
            L_pre[i] = 1
        else:
            # true
            L_pre[i] = 0

        A = L_pre[i]
        # print('A:', A)
        B = L_L[i]
        # print('B:', B)
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1

    cc = (L_pre == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    #Accu = float((N / (aa*bb)) * 100)
    Accu = (TP+TN)*100/(TP+TN+FP+FN)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN + 1)

    return Accu, Pre, Rec, F1, FPR, L_pre

def detection_R_I(Gs, T_mb, L_mb, seq_step, tao):
    # point-wise detection for one dimension
    # (1-lambda)*R(x)+lambda*D(x)
    # lambda=0.5?
    # D_test, Gs, T_mb, L_mb  are of same size

    R = np.absolute(Gs - T_mb)
    R = np.mean(R, axis=2)
    aa = R.shape[0]
    bb = R.shape[1]

    LL = (aa - 1) * seq_step + bb

    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    L_L = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_pre = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            L_L[i * 10 + j] += L_mb[i, j]
            R_L[i * 10 + j] += R[i, j]
            Count[i * 10 + j] += 1
    L_L /= Count
    R_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if R_L[i] > tao:
            # false
            L_pre[i] = 1
        else:
            # true
            L_pre[i] = 0

        A = L_pre[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1

    cc = (L_pre == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    #Accu = float((N / (aa*bb)) * 100)

    Accu = (TP+TN)*100/(TP+TN+FP+FN)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN + 1)

    return Accu, Pre, Rec, F1, FPR, L_pre
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
def detectionRD(DD, Gs, T_mb, L_mb, seq_step, drScoreFlag):
    R = np.absolute(Gs - T_mb)  # Medida de distância entre o dado e o dado reconstruído
    R = np.mean(R, axis=2)
    aa = DD.shape[0]
    bb = DD.shape[1]

    LL = (aa - 1) * seq_step + bb

    DD = abs(DD.reshape([aa, bb]))
    if (drScoreFlag == 2):
        DD = 1-DD
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    D_L = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    L_pre = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i * 10 + j] += DD[i, j]     # Valor do discriminante para o ponto i, j
            L_L[i * 10 + j] += L_mb[i, j]   # Valor da label para o dado i, j
            R_L[i * 10 + j] += R[i, j]      # Valor da medida de distância entre o dado real e o dado reconstruído pelo encoder e gerador
            Count[i * 10 + j] += 1
    D_L /= Count
    L_L /= Count
    R_L /= Count

    return LL, D_L, R_L, L_L, L_pre

def detectionRD_Prediction(LL, D_L, R_L, L_L, L_pre, tao, lam, drScoreFlag):
    dr_score = np.zeros([LL, 1])

    if(drScoreFlag == 0):
        dr_score = D_L  # if only uses D
    elif(drScoreFlag == 1):
        dr_score = R_L  # if only uses G
    else:
        dr_score = (lam * D_L) + ((1 - lam) * R_L)

    Prighthalf = []
    Plefthalf = []
    L_Lright = []
    L_Lleft = []
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(LL):
        # if d[i] > 0:
        # if P[i] > 0.5:
        if dr_score[i] > tao:
            if(drScoreFlag == 0):
                L_pre[i] = 0    # true
                Plefthalf.append(dr_score[i])
                L_Lleft.append(L_L[i])
            else:
                L_pre[i] = 1    # false
                Prighthalf.append(dr_score[i])
                L_Lright.append(L_L[i])
        else:
            if(drScoreFlag == 0):
                L_pre[i] = 1    # false
                Prighthalf.append(dr_score[i])
                L_Lright.append(L_L[i])
            else:
                L_pre[i] = 0    # true
                Plefthalf.append(dr_score[i])
                L_Lleft.append(L_L[i])

        if L_pre[i] == 1 and L_L[i] == 1:
            TP += 1
        elif L_pre[i] == 1 and L_L[i] == 0:
            FP += 1
        elif L_pre[i] == 0 and L_L[i] == 0:
            TN += 1
        elif L_pre[i] == 0 and L_L[i] == 1:
            FN += 1

    # Prighthalf e Plefthalf nunca estarão vazios juntos
    if( len(Prighthalf) == 0 ):
        scaler = MinMaxScaler([0, 0.5])
        Plefthalf = scaler.fit_transform(Plefthalf)
        P = Plefthalf
        L_LOut = L_Lleft
    elif ( len(Plefthalf) == 0 ):
        scaler = MinMaxScaler([0.5, 1])
        Prighthalf = scaler.fit_transform(Prighthalf)
        P = Prighthalf
        L_LOut = L_Lright
    else:
        scaler = MinMaxScaler([0, 0.5])
        Plefthalf = scaler.fit_transform(Plefthalf)
        scaler = MinMaxScaler([0.5, 1])
        Prighthalf = scaler.fit_transform(Prighthalf)
        P = np.concatenate((Plefthalf, Prighthalf))
        L_LOut = np.concatenate((L_Lleft, L_Lright))

    Acc = (TP+TN)*100/(TP+TN+FP+FN)     # Accuracy
    Pre = (100 * TP) / (TP + FP + 1)    # Precision, true positive among all the detected positive
    Rec = (100 * TP) / (TP + FN + 1)    # Recall, true positive among all the real positive, true positive rate (tpr)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))  # The F1 score is the harmonic average of the precision and recall, where an F1
                                                    # score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    FPR = (100 * FP) / (FP + TN + 1)    # False positive rate

    return TP, TN, FP, FN, Acc, Pre, Rec, F1, FPR, L_LOut, P


def computeAndSaveDrScoreAndLabels(DD, Gs, T_mb, L_mb, seq_step, drScoreFlag, lam, savePathDrScore, savePathLabels):
    R = np.absolute(Gs - T_mb)  # Medida de distância entre o dado e o dado reconstruído
    R = np.mean(R, axis=2)
    aa = DD.shape[0]
    bb = DD.shape[1]

    LL = (aa - 1) * seq_step + bb

    DD = abs(DD.reshape([aa, bb]))
    if (drScoreFlag == 2):
        DD = 1-DD
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    D_L = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i * 10 + j] += DD[i, j]     # Valor do discriminante para o ponto i, j
            L_L[i * 10 + j] += L_mb[i, j]   # Valor da label para o dado i, j
            R_L[i * 10 + j] += R[i, j]      # Valor da medida de distância entre o dado real e o dado reconstruído pelo encoder e gerador
            Count[i * 10 + j] += 1
    D_L /= Count
    L_L /= Count
    R_L /= Count

    dr_score = np.zeros([LL, 1])
    if(drScoreFlag == 0):
        dr_score = D_L  # if only uses D
    elif(drScoreFlag == 1):
        dr_score = R_L  # if only uses G
    else:
        dr_score = (lam * D_L) + ((1 - lam) * R_L)

    # print("Saving...")
    np.save(savePathDrScore, dr_score)
    np.save(savePathLabels, L_L)
    print("Saving.......... drScores and Labels Saved")
    return dr_score, L_L

def computeAndSaveDandRLossesSingleG(DD1, DD2, Gs, T_mb, L_mb, seq_step, savePath_DL1, savePath_DL2, savePath_LL, savePath_RL):
    # R = np.absolute(Gs - T_mb)  # Medida de distância entre o dado e o dado reconstruído
    # R = np.mean(R, axis=2)
    R = (np.square(Gs - T_mb)).mean(axis=2)
    aa = DD1.shape[0]
    bb = DD1.shape[1]
    LL = (aa - 1) * seq_step + bb

    DD1 = abs(DD1.reshape([aa, bb]))
    DD2 = abs(DD2.reshape([aa, bb]))
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    D_L_1 = np.zeros([LL, 1])
    D_L_2 = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L_1[i * 10 + j] += DD1[i, j]     # Valor do discriminante para o ponto i, j
            D_L_2[i * 10 + j] += DD2[i, j]     # Valor do discriminante para o ponto i, j
            L_L[i * 10 + j] += L_mb[i, j]   # Valor da label para o dado i, j
            R_L[i * 10 + j] += R[i, j]      # Valor da medida de distância entre o dado real e o dado reconstruído pelo encoder e gerador
            Count[i * 10 + j] += 1
    D_L_1 /= Count
    D_L_2 /= Count
    L_L /= Count
    R_L /= Count

    np.save(savePath_DL1, D_L_1)
    np.save(savePath_DL2, D_L_2)
    np.save(savePath_LL, L_L)
    np.save(savePath_RL, R_L)
    print("Saving.......... DL1, DL2, LL and RL")
    return D_L_1, D_L_2, L_L, R_L

def computeAndSaveDandRLosses(DD1, DD2, Gs, Gs_l, T_mb, L_mb, seq_step, savePath_DL1, savePath_DL2, savePath_LL, savePath_RL, savePath_RL_log):
    # R = np.absolute(Gs - T_mb)  # Medida de distância entre o dado e o dado reconstruído
    # R = np.mean(R, axis=2)
    R = (np.square(Gs - T_mb)).mean(axis=2)
    # R_log = np.absolute(Gs_l - T_mb)  # Medida de distância entre o dado e o dado reconstruído
    # R_log = np.mean(R_log, axis=2)
    R_log = (np.square(Gs_l - T_mb)).mean(axis=2)
    aa = DD1.shape[0]
    bb = DD1.shape[1]
    LL = (aa - 1) * seq_step + bb

    DD1 = abs(DD1.reshape([aa, bb]))
    DD2 = abs(DD2.reshape([aa, bb]))
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    D_L_1 = np.zeros([LL, 1])
    D_L_2 = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    R_log_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L_1[i * 10 + j] += DD1[i, j]     # Valor do discriminante para o ponto i, j
            D_L_2[i * 10 + j] += DD2[i, j]     # Valor do discriminante para o ponto i, j
            L_L[i * 10 + j] += L_mb[i, j]   # Valor da label para o dado i, j
            R_L[i * 10 + j] += R[i, j]      # Valor da medida de distância entre o dado real e o dado reconstruído pelo encoder e gerador
            R_log_L[i * 10 + j] += R_log[i, j]      # Valor da medida de distância entre o dado real e o dado reconstruído pelo encoder e gerador
            Count[i * 10 + j] += 1
    D_L_1 /= Count
    D_L_2 /= Count
    L_L /= Count
    R_L /= Count
    R_log_L /= Count

    np.save(savePath_DL1, D_L_1)
    np.save(savePath_DL2, D_L_2)
    np.save(savePath_LL, L_L)
    np.save(savePath_RL, R_L)
    np.save(savePath_RL_log, R_log_L)
    print("Saving.......... DL1, DL2, LL and RL")
    return D_L_1, D_L_2, L_L, R_L, R_log_L


def computeAndSaveDLoss(DD1, DD2, T_mb, L_mb, seq_step, savePath_DL1, savePath_DL2, savePath_LL):
    aa = DD1.shape[0]
    bb = DD1.shape[1]
    LL = (aa - 1) * seq_step + bb

    DD1 = abs(DD1.reshape([aa, bb]))
    DD2 = abs(DD2.reshape([aa, bb]))
    L_mb = L_mb.reshape([aa, bb])

    D_L_1 = np.zeros([LL, 1])
    D_L_2 = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L_1[i * 10 + j] += DD1[i, j]     # Valor do discriminante para o ponto i, j
            D_L_2[i * 10 + j] += DD2[i, j]     # Valor do discriminante para o ponto i, j
            L_L[i * 10 + j] += L_mb[i, j]   # Valor da label para o dado i, j
            Count[i * 10 + j] += 1
    D_L_1 /= Count
    D_L_2 /= Count
    L_L /= Count

    np.save(savePath_DL1, D_L_1)
    np.save(savePath_DL2, D_L_2)
    np.save(savePath_LL, L_L)
    print("Saving.......... DL1, DL2, and LL")
    return D_L_1, D_L_2, L_L

def computeAndSaveRLoss(Gs, T_mb, L_mb, seq_step, savePath_LL, savePath_RL):
    R = np.absolute(Gs - T_mb)  # Medida de distância entre o dado e o dado reconstruído
    R = np.mean(R, axis=2)
    aa = Gs.shape[0]
    bb = Gs.shape[1]
    LL = (aa - 1) * seq_step + bb

    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    R_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            L_L[i * 10 + j] += L_mb[i, j]   # Valor da label para o dado i, j
            R_L[i * 10 + j] += R[i, j]      # Valor da medida de distância entre o dado real e o dado reconstruído pelo encoder e gerador
            Count[i * 10 + j] += 1
    L_L /= Count
    R_L /= Count

    np.save(savePath_LL, L_L)
    np.save(savePath_RL, R_L)
    print("Saving.......... LL and RL")
    return L_L, R_L
#-----------------------------------------------------------------------------------------------------------------------
