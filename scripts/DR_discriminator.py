import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import model
import mmd
from mod_core_rnn_cell_impl import LSTMCell
from sklearn.metrics import precision_recall_fscore_support
import math

def anomaly_detection_plot(D_test, T_mb, L_mb, D_L, epoch, identifier):

    aa = D_test.shape[0]
    bb = D_test.shape[1]
    D_L = D_L.reshape([aa, bb, -1])

    x_points = np.arange(bb)

    fig, ax = plt.subplots(4, 4, sharex=True)
    for m in range(4):
        for n in range(4):
            D = D_test[n * 4 + m, :, :]
            T = T_mb[n * 4 + m, :, :]
            L = L_mb[n * 4 + m, :, :]
            DL = D_L[n * 4 + m, :, :]
            ax[m, n].plot(x_points, D, '--g', label='Pro')
            ax[m, n].plot(x_points, T, 'b', label='Data')
            ax[m, n].plot(x_points, L, 'k', label='Label')
            ax[m, n].plot(x_points, DL, 'r', label='Label')
            ax[m, n].set_ylim(-1, 1)
    for n in range(4):
        ax[-1, n].xaxis.set_ticks(range(0, bb, int(bb/6)))
    fig.suptitle(epoch)
    fig.subplots_adjust(hspace=0.15)
    fig.savefig("./experiments/plots/DR_dis/" + identifier + "_epoch" + str(epoch).zfill(4) + ".png")
    plt.clf()
    plt.close()

    return True

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
            # print('index:', i*10+j)
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
    # print('D_L:', D_L)
    # print('L_L:', L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    Accu = float((N / LL) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    return Accu, precision, recall, f1,


def detection_logits_I(DL_test, L_mb, I_mb, seq_step, tao):
    aa = DL_test.shape[0]
    bb = DL_test.shape[1]

    LL = (aa-1)*seq_step+bb

    DL_test = abs(DL_test.reshape([aa, bb]))
    L_mb = L_mb .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i*seq_step+j] += DL_test[i, j]
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
    # print('D_L:', D_L)
    # print('L_L:', L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    Accu = float((N / LL) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    # true positive among all the detected positive
    # Pre = (100 * TP) / (TP + FP + 1)
    # # true positive among all the real positive
    # Rec = (100 * TP) / (TP + FN + 1)
    # # The F1 score is the harmonic average of the precision and recall,
    # # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    # F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN+1)

    return Accu, precision, recall, f1, FPR, D_L

def detection_statistic_I(D_test, L_mb, I_mb, seq_step, tao):
    # point-wise detection for one dimension

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    LL = (aa-1) * seq_step + bb
    # print('aa:', aa)
    # print('bb:', bb)
    # print('LL:', LL)

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    I_mb = I_mb.reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i * 10 + j)
            D_L[i * seq_step + j] += D_test[i, j]
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
    Accu = float((N / LL) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    # true positive among all the detected positive
    # Pre = (100 * TP) / (TP + FP + 1)
    # # true positive among all the real positive
    # Rec = (100 * TP) / (TP + FN + 1)
    # # The F1 score is the harmonic average of the precision and recall,
    # # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    # F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, precision, recall, f1, FPR, D_L

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
            # print('index:', i*10+j)
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
    # print('D_L:', D_L)
    # print('L_L:', L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    #Accu = float((N / LL) * 100)
    Accu = (TP+TN)*100/(TP+TN+FP+FN)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN+1)

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
    FPR = (100 * FP) / (FP + TN+1)

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
    FPR = (100 * FP) / (FP + TN+1)

    return Accu, Pre, Rec, F1, FPR, L_pre


def sample_detection(D_test, L_mb, tao):
    # sample-wise detection for one dimension

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    L = np.sum(L_mb, 1)
    # NN = 0-10
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
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (aa)) * 100)
    # Accu = (TP+TN)*100/(TP+TN+FP+FN)

    precision, recall, f1, _ = precision_recall_fscore_support(L, D_L, average='binary')

    return Accu, precision, recall, f1


def CUSUM_det(spe_n, spe_a, labels):

    mu = np.mean(spe_n)
    sigma = np.std(spe_n)

    kk = 3*sigma
    H = 15*sigma
    print('H:', H)

    tar = np.mean(spe_a)

    mm = spe_a.shape[0]

    SH = np.empty([mm, ])
    SL = np.empty([mm, ])

    for i in range(mm):
        SH[-1] = 0
        SL[-1] = 0
        SH[i] = max(0, SH[i-1]+spe_a[i]-(tar+kk))
        SL[i] = min(0, SL[i-1]+spe_a[i]-(tar-kk))


    count = np.empty([mm, ])
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(mm):
        A = SH[i]
        B = SL[i]
        AA = H
        BB = -H
        if A <= AA and B >= BB:
            count[i] = 0
        else:
            count[i] = 1

        C = count[i]
        D = labels[i]
        if C == 1 and D == 1:
            TP += 1
        elif C == 1 and D == 0:
            FP += 1
        elif C == 0 and D == 0:
            TN += 1
        elif C == 0 and D == 1:
            FN += 1

    cc = (count == labels)
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (mm)) * 100)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, Pre, Rec, F1, FPR


def SPE(X, pc):
    a = X.shape[0]
    b = X.shape[1]

    spe = np.empty([a])
    # Square Prediction Error (square of residual distance)
    #  spe = X'(I-PP')X
    I = np.identity(b, float) - np.matmul(pc.transpose(1, 0), pc)
    # I = np.matmul(I, I)
    for i in range(a):
        x = X[i, :].reshape([b, 1])
        y = np.matmul(x.transpose(1, 0), I)
        spe[i] = np.matmul(y, x)

    return spe



def generator_o(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None, learn_scale=True):
    """
    If parameters are supplied, initialise as such
    """
    # It is important to specify different variable scopes for the LSTM cells.
    with tf.compat.v1.variable_scope("generator_o") as scope:

        W_out_G_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/W_out_G:0'])
        b_out_G_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/b_out_G:0'])
        try:
            scale_out_G_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/scale_out_G:0'])
        except KeyError:
            scale_out_G_initializer = tf.compat.v1.constant_initializer(value=1)
            assert learn_scale
        lstm_initializer = tf.compat.v1.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
        bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.compat.v1.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.compat.v1.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.compat.v1.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer, trainable=False)

        inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                        state_is_tuple=True,
                        initializer=lstm_initializer,
                        bias_start=bias_start,
                        reuse=reuse)
        rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.compat.v1.float32,
            sequence_length=[seq_length] * batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.compat.v1.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.compat.v1.matmul(rnn_outputs_2d, W_out_G) + b_out_G #out put weighted sum
        output_2d = tf.compat.v1.nn.tanh(logits_2d) # logits operation [-1, 1]
        output_3d = tf.compat.v1.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d


def discriminator_o(x, hidden_units_d, reuse=False, parameters=None):

    with tf.compat.v1.variable_scope("discriminator_0") as scope:

        W_out_D_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/W_out_D:0'])
        b_out_D_initializer = tf.compat.v1.constant_initializer(value=parameters['discriminator/b_out_D:0'])

        W_out_D = tf.compat.v1.get_variable(name='W_out_D', shape=[hidden_units_d, 1],  initializer=W_out_D_initializer)
        b_out_D = tf.compat.v1.get_variable(name='b_out_D', shape=1, initializer=b_out_D_initializer)

        inputs = x

        # cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=reuse)
        cell = LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=reuse)
        rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)
        logits = tf.compat.v1.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D # output weighted sum
        output = tf.compat.v1.nn.sigmoid(logits) # y = 1 / (1 + exp(-x)). output activation [0, 1]. Probability??
        # sigmoid output ([0,1]), Probability?

    return output, logits
    
def invert(settings, samples, para_path, g_tolerance=None, e_tolerance=0.1, n_iter=None, max_iter=10000, heuristic_sigma=None):
    """
    Return the latent space points corresponding to a set of a samples
    ( from gradient descent )
    Note: this function is designed for ONE sample generation
    """
    # num_samples = samples.shape[0]
    # cast samples to float32

    samples = np.float32(samples)
    # get the model
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))
    # get parameters
    parameters = model.load_parameters(para_path)
    # # assertions
    # assert samples.shape[2] == settings['num_generated_features']
    # create VARIABLE Z
    Z = tf.compat.v1.get_variable(name='Z', shape=[1, settings['seq_length'], settings['latent_dim']], initializer=tf.compat.v1.random_normal_initializer())
    # create outputs
    G_samples = generator_o(Z, settings['hidden_units_g'], settings['seq_length'], 1, settings['num_generated_features'], reuse=False, parameters=parameters)
    # generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
    # generator_settings = dict((k, settings[k]) for k in generator_vars)
    # G_samples = model.generator(Z, **generator_settings, reuse=True)

    fd = None

    # define loss mmd-based loss
    if heuristic_sigma is None:
        # heuristic_sigma = mmd.median_pairwise_distance_o(samples)  # this is noisy
        heuristic_sigma = mmd.median_pairwise_distance_o(samples) + np.array(0.0000001, dtype=np.float32)
        print('heuristic_sigma:', heuristic_sigma)
    samples = tf.compat.v1.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])
    Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.compat.v1.constant(value=heuristic_sigma, shape=(1, 1)))
    similarity_per_sample = tf.compat.v1.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    # reconstruction_error_per_sample = tf.reduce_sum((tf.nn.l2_normalize(G_samples, dim=1) - tf.nn.l2_normalize(samples, dim=1))**2, axis=[1,2])
    similarity = tf.compat.v1.reduce_mean(similarity_per_sample)
    reconstruction_error = 1 - similarity
    # updater
    #    solver = tf.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Z])
    solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.05).minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Z])

    grad_Z = tf.compat.v1.gradients(reconstruction_error_per_sample, Z)[0]
    grad_per_Z = tf.compat.v1.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.compat.v1.reduce_mean(grad_per_Z)
    # solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Z])
    print('Finding latent state corresponding to samples...')

    # sess = tf.compat.v1.Session()
    # sess.run(tf.compat.v1.global_variables_initializer())
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        # with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
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
                    # error = sess.run(reconstruction_error, feed_dict=fd)
                    Zs, error_per_sample, error = sess.run([Z, reconstruction_error_per_sample, reconstruction_error], feed_dict=fd)
                    i += 1
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        Gs = sess.run(G_samples, feed_dict={Z: Zs})
        error_per_sample= sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.compat.v1.reset_default_graph()

    return Gs, Zs, error_per_sample, heuristic_sigma


def invert2(settings, samples, para_path, g_tolerance=None, e_tolerance=0.1, n_iter=None, max_iter=10000, heuristic_sigma=None):
    """
    Return the latent space points corresponding to a set of a samples (from gradient descent)
    Note: this function is designed for ONE sample generation
    """
    # num_samples = samples.shape[0]
    # cast samples to float32

    samples = np.float32(samples)

    # get the model
    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    # print('Inverting', 1, 'samples using model', settings['identifier'], 'at epoch', epoch,)
    # if not g_tolerance is None:
    #     print('until gradient norm is below', g_tolerance)
    # else:
    #     print('until error is below', e_tolerance)

    # get parameters
    parameters = model.load_parameters(para_path)
    # # assertions
    # assert samples.shape[2] == settings['num_generated_features']
    # create VARIABLE Z
    Z = tf.compat.v1.get_variable(name='Z', shape=[1, settings['seq_length'], settings['latent_dim']], initializer=tf.compat.v1.random_normal_initializer())
    # create outputs

    G_samples = generator_o(Z, settings['hidden_units_g'], settings['seq_length'], 1, settings['num_generated_features'], reuse=False, parameters=parameters)
    # generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
    # generator_settings = dict((k, settings[k]) for k in generator_vars)
    # G_samples = model.generator(Z, **generator_settings, reuse=True)

    fd = None

    # define loss mmd-based loss
    if heuristic_sigma is None:
        # heuristic_sigma = mmd.median_pairwise_distance_o(samples)  # this is noisy
        # heuristic_sigma = mmd.median_pairwise_distance_o(samples)  + np.(0.0000001).astype(dtype=float32)
        heuristic_sigma = mmd.median_pairwise_distance_o(samples) + np.array(0.0000001, dtype=np.float32)
        print('heuristic_sigma:', heuristic_sigma)
    samples = tf.compat.v1.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])
    # Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))

    # base = 1.0
    # sigma_list = [1, 2, 4, 8, 16]
    # sigma_list = [sigma / base for sigma in sigma_list]
    # Kxx, Kxy, Kyy, len_sigma_list = mmd._mix_rbf_kernel2(G_samples, samples, sigma_list)

    # ---------------------
    X = G_samples
    Y = samples
    sigmas = tf.constant(value=heuristic_sigma, shape=(1, 1))
    wts = [1.0] * sigmas.get_shape()[0]
    if len(X.shape) == 2:
        # matrix
        XX = tf.compat.v1.matmul(X, X, transpose_b=True)
        XY = tf.compat.v1.matmul(X, Y, transpose_b=True)
        YY = tf.compat.v1.matmul(Y, Y, transpose_b=True)
    elif len(X.shape) == 3:
        # tensor -- this is computing the Frobenius norm
        XX = tf.compat.v1.tensordot(X, X, axes=[[1, 2], [1, 2]])
        XY = tf.compat.v1.tensordot(X, Y, axes=[[1, 2], [1, 2]])
        YY = tf.compat.v1.tensordot(Y, Y, axes=[[1, 2], [1, 2]])
    else:
        raise ValueError(X)
    X_sqnorms = tf.compat.v1.diag_part(XX)
    Y_sqnorms = tf.compat.v1.diag_part(YY)
    r = lambda x: tf.compat.v1.expand_dims(x, 0)
    c = lambda x: tf.compat.v1.expand_dims(x, 1)
    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(tf.compat.v1.unstack(sigmas, axis=0), wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.compat.v1.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.compat.v1.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.compat.v1.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
    Kxx = K_XX
    Kxy = K_XY
    Kyy = K_YY
    wts = tf.compat.v1.reduce_sum(wts)
    # ---------------------



    similarity_per_sample = tf.compat.v1.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    # reconstruction_error_per_sample = tf.reduce_sum((tf.nn.l2_normalize(G_samples, dim=1) - tf.nn.l2_normalize(samples, dim=1))**2, axis=[1,2])
    reconstruction_error = 1 - tf.compat.v1.reduce_mean(similarity_per_sample)


    # updater
    # solver = tf.compat.v1.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Z])
    solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0).minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Z])

    # grad_Z = tf.compat.v1.gradients(reconstruction_error_per_sample, Z)[0]
    grad_Z = tf.compat.v1.gradients(reconstruction_error, Z)[0]
    grad_per_Z = tf.compat.v1.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.compat.v1.reduce_mean(grad_per_Z)
    # solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Z])
    print('Finding latent state corresponding to samples...')

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    with tf.compat.v1.Session() as sess:
        # graph = tf.compat.v1.Graph()
        # graphDef = graph.as_graph_def()
        sess.run(tf.compat.v1.global_variables_initializer())
        error = sess.run(reconstruction_error, feed_dict=fd)
        g_n = sess.run(grad_norm, feed_dict=fd)
        # print(g_n)
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
                # while ((np.abs(error) > e_tolerance) or (math.isnan(error))):
                    solver.run(feed_dict=fd)
                    # error = sess.run(reconstruction_error, feed_dict=fd)

                    reconstruction_error_out, reconstruction_error_per_sample_out, similarity_per_sample_out, Kxy_out, G_samples_out, samples_out, Z_out, X_sqnorms_out, Y_sqnorms_out, XY_out, sigmas_out = sess.run([reconstruction_error, reconstruction_error_per_sample, similarity_per_sample, Kxy, G_samples, samples, Z, X_sqnorms, Y_sqnorms, XY, sigmas], feed_dict=fd)
                    if(math.isnan(reconstruction_error_out)):
                        print("nan")

                    # Zs, Gs, Kxy_s, error_per_sample, error = sess.run([Z, G_samples, Kxy, reconstruction_error_per_sample, reconstruction_error], feed_dict=fd)
                    # Zs, Gs, Kxy_s, error_per_sample, error, _ = sess.run([Z, G_samples, Kxy, reconstruction_error_per_sample, reconstruction_error, solver], feed_dict=fd)
                    i += 1
                    # print(error)
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        Gs = sess.run(G_samples, feed_dict={Z: Zs})
        error_per_sample = sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.compat.v1.reset_default_graph()

    return Gs, Zs, error_per_sample, heuristic_sigma

def invert2(settings, samples, para_path, g_tolerance=None, e_tolerance=0.1, n_iter=None, max_iter=10000, heuristic_sigma=None):
    """
    Return the latent space points corresponding to a set of a samples (from gradient descent)
    Note: this function is designed for ONE sample generation
    """
    # num_samples = samples.shape[0]
    # cast samples to float32

    samples = np.float32(samples)

    # get the model
    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    # get parameters
    parameters = model.load_parameters(para_path)
    Z = tf.compat.v1.get_variable(name='Z', shape=[1, settings['seq_length'], settings['latent_dim']], initializer=tf.compat.v1.random_normal_initializer())
    # create outputs
    G_samples = generator_o(Z, settings['hidden_units_g'], settings['seq_length'], 1, settings['num_generated_features'], reuse=False, parameters=parameters)
    # generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
    # generator_settings = dict((k, settings[k]) for k in generator_vars)
    # G_samples = model.generator(Z, **generator_settings, reuse=True)
    # G_samples = model.generator(Z, settings['hidden_units_g'], settings['seq_length'], 1, settings['num_generated_features'], reuse=False, parameters=parameters)

    fd = None

    # define loss mmd-based loss
    if heuristic_sigma is None:
        heuristic_sigma = mmd.median_pairwise_distance_o(samples)  # this is noisy
        print('heuristic_sigma:', heuristic_sigma)
    samples = tf.compat.v1.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])
    Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
    similarity_per_sample = tf.compat.v1.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    # reconstruction_error_per_sample = tf.reduce_sum((tf.nn.l2_normalize(G_samples, dim=1) - tf.nn.l2_normalize(samples, dim=1))**2, axis=[1,2])
    similarity = tf.compat.v1.reduce_mean(similarity_per_sample)
    reconstruction_error = 1 - similarity
    # updater
    # solver = tf.compat.v1.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Z])
    solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.1).minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Z])

    grad_Z = tf.compat.v1.gradients(reconstruction_error_per_sample, Z)[0]
    grad_per_Z = tf.compat.v1.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.compat.v1.reduce_mean(grad_per_Z)
    # solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Z])
    print('Finding latent state corresponding to samples...')

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    with tf.compat.v1.Session() as sess:
        # graph = tf.compat.v1.Graph()
        # graphDef = graph.as_graph_def()
        sess.run(tf.compat.v1.global_variables_initializer())
        error = sess.run(reconstruction_error, feed_dict=fd)
        g_n = sess.run(grad_norm, feed_dict=fd)
        # print(g_n)
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
                    # print(error)
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        Gs = sess.run(G_samples, feed_dict={Z: Zs})
        error_per_sample = sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.compat.v1.reset_default_graph()

    return Gs, Zs, error_per_sample, heuristic_sigma

def dis_trained_model(settings, samples, para_path):
    """
    Return the discrimination results of num_samples testing samples from a trained model described by settings dict
    Note: this function is designed for ONE sample discrimination
    """

    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    num_samples = samples.shape[0]
    samples = np.float32(samples)
    num_variables = samples.shape[2]
    # samples = np.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])

    # get the parameters, get other variables
    # parameters = model.load_parameters(settings['sub_id'] + '_' + str(settings['seq_length']) + '_' + str(epoch))
    parameters = model.load_parameters(para_path)
    # settings['sub_id'] + '_' + str(settings['seq_length']) + '_' + str(epoch)

    # create placeholder, T samples
    # T = tf.placeholder(tf.float32, [settings['batch_size'], settings['seq_length'], settings['num_generated_features']])

    T = tf.placeholder(tf.float32, [num_samples, settings['seq_length'], num_variables])

    # create the discriminator (GAN)
    # normal GAN
    D_t, L_t = discriminator_o(T, settings['hidden_units_d'], reuse=False, parameters=parameters)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.device('/gpu:1'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        D_t, L_t = sess.run([D_t, L_t], feed_dict={T: samples})

    tf.reset_default_graph()
    return D_t, L_t

def dis_D_model(settings, samples, para_path):
    """
    Return the discrimination results of  num_samples testing samples from a trained model described by settings dict
    Note: this function is designed for ONE sample discrimination
    """

    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    # num_samples = samples.shape[0]
    samples = np.float32(samples)
    samples = np.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])

    # get the parameters, get other variables
    parameters = model.load_parameters(para_path)
    # create placeholder, T samples

    T = tf.compat.v1.placeholder(tf.float32, [1, settings['seq_length'], settings['num_generated_features']])

    # create the discriminator (GAN or CGAN)
    # normal GAN
    D_t, L_t = discriminator_o(T, settings['hidden_units_d'], reuse=False, parameters=parameters)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        D_t, L_t = sess.run([D_t, L_t], feed_dict={T: samples})

    tf.compat.v1.reset_default_graph()
    return D_t, L_t