import tensorflow as tf
import numpy as np
import pdb
import json
from mod_core_rnn_cell_impl import LSTMCell  # modified to allow initializing bias in lstm

import data_utils
import plotting
import model
import mmd
import utils
import eval
import DR_discriminator
import autoencoderFunctions
import os

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.privacy_accountant.tf import accountant

from time import time

begin = time()

tf.compat.v1.disable_eager_execution()
# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
data_path = './datasets/' + settings['data_load_from']
print('Loading data from', data_path)
samples, labels, index = data_utils.get_data(settings["data"], settings["seq_length"], settings["seq_step"],
                                             settings["num_signals"], settings["sub_id"], settings["eval_single"],
                                             settings["eval_an"], data_path)
# --- save settings, data --- #
# no need
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

class myADclass():
    def __init__(self, epoch, settings=settings, samples=samples, labels=labels, index=index):
        self.epoch = epoch
        self.settings = settings
        # self.samples = samples
        self.samples = samples[1:1001, :, :]
        self.labels = labels
        self.index = index

    def ADfunc(self):
        timeG = []
        
        num_samples_t = self.samples.shape[0]
        # num_samples_t = 100

        batch_size = self.settings['batch_size']
        # batch_size = 1
        num_batches = num_samples_t // batch_size

        print('samples shape:', self.samples.shape)
        print('num_samples_t', num_samples_t)
        print('batch_size', batch_size)
        print('num_batches', num_batches)

        # -- only discriminate one batch for one time -- #
        D_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        DL_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        GG = np.empty([num_samples_t, self.settings['seq_length'], self.settings['num_signals']])
        T_samples = np.empty([num_samples_t, self.settings['seq_length'], self.settings['num_signals']])
        L_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        I_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
 
        # for batch_idx in range(num_batches):
        for batch_idx in range(0, num_batches):
            # for batch_idx in range(0, 5):
            model.display_batch_progression(batch_idx, num_batches)
            start_pos = batch_idx * batch_size
            end_pos = start_pos + batch_size
            T_mb = self.samples[start_pos:end_pos, :, :]
            L_mmb = self.labels[start_pos:end_pos, :, :]
            I_mmb = self.index[start_pos:end_pos, :, :]

            para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
            D_t, L_t = DR_discriminator.dis_D_model(self.settings, T_mb, para_path)
            # D_t, L_t = autoencoderFunctions.discriminatorTrainedModel(self.settings, T_mb, para_path)

            time1 = time()
            Gs, Zs, error_per_sample, heuristic_sigma = DR_discriminator.invert(self.settings, T_mb, para_path,
                                                                                g_tolerance=None,
                                                                                e_tolerance=0.1, n_iter=None,
                                                                                max_iter=10,
                                                                                heuristic_sigma=None)
            timeG = np.append(timeG, time() - time1)

            D_test[start_pos:end_pos, :, :] = D_t
            DL_test[start_pos:end_pos, :, :] = L_t
            GG[start_pos:end_pos, :, :] = Gs
            T_samples[start_pos:end_pos, :, :] = T_mb
            L_mb[start_pos:end_pos, :, :] = L_mmb
            I_mb[start_pos:end_pos, :, :] = I_mmb

        # Completes the sample data that wasn't in the last batch because the batch wasn't complete
        start_pos = num_batches * batch_size
        end_pos = start_pos + batch_size
        size = samples[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([batch_size - size, samples.shape[1], samples.shape[2]])
        batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)
        
        para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
        D_t, L_t = DR_discriminator.dis_D_model(self.settings, T_mb, para_path)
        # D_t, L_t = autoencoderFunctions.discriminatorTrainedModel(self.settings, T_mb, para_path)
        # time1 = time()
        Gs, Zs, error_per_sample, heuristic_sigma = DR_discriminator.invert(self.settings, T_mb, para_path,
                                                                            g_tolerance=None,
                                                                            e_tolerance=0.1, n_iter=None,
                                                                            max_iter=10,
                                                                            heuristic_sigma=None)
        # timeG = np.append(time() - time1)

        np.save(path_AD_autoencoder_results + "/timeG.npy", timeG)

        D_test[start_pos:end_pos, :, :] = D_t[:size, :, :]
        DL_test[start_pos:end_pos, :, :] = L_t[:size, :, :]
        GG[start_pos:end_pos, :, :] = Gs[:size, :, :]
        T_samples[start_pos:end_pos, :, :] = T_mb[:size, :, :]
        L_mmb = self.labels[start_pos:end_pos, :, :]
        I_mmb = self.index[start_pos:end_pos, :, :]
        L_mb[start_pos:end_pos, :, :] = L_mmb
        I_mb[start_pos:end_pos, :, :] = I_mmb

        #------------------------------------------------------------
        savePath_DL1 = path_AD_autoencoder_results + "/DL1" + ".npy"
        savePath_DL2 = path_AD_autoencoder_results + "/DL2" + ".npy"
        savePath_LL = path_AD_autoencoder_results + "/LL" +  ".npy"
        savePath_RL = path_AD_autoencoder_results + "/RL" +  ".npy"
        D_L_1, D_L_2, L_L, R_L = autoencoderFunctions.computeAndSaveDandRLossesSingleG(D_test, DL_test, GG, T_samples, L_mb, self.settings['seq_step'], savePath_DL1, savePath_DL2, savePath_LL, savePath_RL)
        #------------------------------------------------------------
        return


if __name__ == "__main__":
    print('Main Starting...')

    for epochGAN in [4]:
        path_AD_autoencoder_results = settings["path_AD_autoencoder_results_invert"] + str(epochGAN)
        try:
            os.mkdir(path_AD_autoencoder_results)
        except:
            pass
        print("\n\nepochGAN: ", epochGAN)
        ad = myADclass(epochGAN)
        ad.ADfunc()

    print('Main Terminating...')
    end = time() - begin
    print('Testing terminated | Execution time=%d s' % (end))

