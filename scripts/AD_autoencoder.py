import data_utils
import utils
import model
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from mod_core_rnn_cell_impl import LSTMCell  # modified to allow initializing bias in lstm
import autoencoderFunctions
from sklearn import metrics
import matplotlib.pyplot as plt
import os

begin = time()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load Data ------------------------------------------------------------------------------------------------------------
# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
# samples, pdf, labels = data_utils.get_data(settings)
data_path = './datasets/' + settings['data_load_from']
print('Loading data from', data_path)
settings["eval_an"] = False
settings["eval_single"] = False
samples, labels, index = data_utils.get_data(settings["data"], settings["seq_length"], settings["seq_step"],
                                             settings["num_signals"], settings['sub_id'], settings["eval_single"],
                                             settings["eval_an"], data_path)
# -- number of variables -- #
num_variables = samples.shape[2]
print('num_variables:', num_variables)
# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)
#-----------------------------------------------------------------------------------------------------------------------

# Class Definition -----------------------------------------------------------------------------------------------------
class myADclass():
    def __init__(self, epoch_GAN, epoch_autoencoder, settings=settings, samples=samples, labels=labels, index=index):
        self.epoch_autoencoder = epoch_autoencoder
        self.epoch_GAN = epoch_GAN
        self.settings = settings
        self.samples = samples
        self.labels = labels
        self.index = index
        
    def ADfunc(self):
        num_samples_t = self.samples.shape[0]
        num_batches = num_samples_t // self.settings['batch_size']

        print('samples shape:', self.samples.shape)
        print('num_samples_t', num_samples_t)
        print('batch_size', self.settings['batch_size'])
        print('num_batches', num_batches)

        # -- only discriminate one batch for one time -- #
        GG = np.empty([num_samples_t, self.settings['seq_length'], self.settings['num_signals']])
        GG_l = np.empty([num_samples_t, self.settings['seq_length'], self.settings['num_signals']])
        T_samples = np.empty([num_samples_t, self.settings['seq_length'], self.settings['num_signals']])
        D_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        DL_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        L_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        I_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])

        for batch_idx in range(num_batches):
            model.display_batch_progression(batch_idx, num_batches)
            start_pos = batch_idx * self.settings['batch_size']
            end_pos = start_pos + self.settings['batch_size']
            T_mb = self.samples[start_pos:end_pos, :, :]
            L_mmb = self.labels[start_pos:end_pos, :, :]
            I_mmb = self.index[start_pos:end_pos, :, :]

            # GAN parameters path to load pre trained discriminator and generator
            para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(self.settings['seq_length']) + '_' + str(self.epoch_GAN) + '.npy'
            # Discriminator output values using pre trained GAN discriminator model
            if(dgbConfig != 2):
                D_output, D_logits = autoencoderFunctions.discriminatorTrainedModel(self.settings, T_mb, para_path)
                D_test[start_pos:end_pos, :, :] = D_output
                DL_test[start_pos:end_pos, :, :] = D_logits

            # Encoder parameters path to load pre trained encoder
            # Generator output values using pre trained encoder and (GAN) generator model
            if(dgbConfig != 1):
                G_output, G_logits = autoencoderFunctions.encoderGeneratorTrainedModels(self.settings, T_mb, para_path, path_autoencoder_training_parameters)
                GG[start_pos:end_pos, :, :] = G_output
                GG_l[start_pos:end_pos, :, :] = G_logits

            T_samples[start_pos:end_pos, :, :] = T_mb
            L_mb[start_pos:end_pos, :, :] = L_mmb
            I_mb[start_pos:end_pos, :, :] = I_mmb

        # Completes the sample data that wasn't in the last batch because the batch wasn't complete
        start_pos = num_batches * self.settings['batch_size']
        end_pos = start_pos + self.settings['batch_size']
        size = samples[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([self.settings['batch_size'] - size, samples.shape[1], samples.shape[2]])
        batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)
        
        para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(self.settings['seq_length']) + '_' + str(self.epoch_GAN) + '.npy'
        if(dgbConfig != 2):
            D_output, D_logits = autoencoderFunctions.discriminatorTrainedModel(self.settings, batch, para_path)
            D_test[start_pos:end_pos, :, :] = D_output[:size, :, :]
            DL_test[start_pos:end_pos, :, :] = D_logits[:size, :, :]
        
        if(dgbConfig != 1):
            G_output, G_logits = autoencoderFunctions.encoderGeneratorTrainedModels(self.settings, batch, para_path, path_autoencoder_training_parameters)
            GG[start_pos:end_pos, :, :] = G_output[:size, :, :]
            GG_l[start_pos:end_pos, :, :] = G_logits[:size, :, :]

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
        savePath_RL_log = path_AD_autoencoder_results + "/RL_log" +  ".npy"

        if(dgbConfig == 1):
            D_L_1, D_L_2, L_L = autoencoderFunctions.computeAndSaveDLoss(D_test, DL_test, T_samples, L_mb, self.settings['seq_step'], savePath_DL1, savePath_DL2, savePath_LL)
        elif(dgbConfig == 2):
            L_L, R_L = autoencoderFunctions.computeAndSaveRLoss(GG, T_samples, L_mb, self.settings['seq_step'], savePath_LL, savePath_RL)
        else:
            D_L_1, D_L_2, L_L, R_L, R_log_L = autoencoderFunctions.computeAndSaveDandRLosses(D_test, DL_test, GG, GG_l, T_samples, L_mb, self.settings['seq_step'], savePath_DL1, savePath_DL2, savePath_LL, savePath_RL, savePath_RL_log)
        #------------------------------------------------------------

        return
#-----------------------------------------------------------------------------------------------------------------------

# Main Executtion ------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print('Main Starting...')

    # Locally defined parameters and paths ---------------------------------------------------------------------------------
    dgbConfig = settings["dgbConfig"]
    epochGAN_list = settings["epochGAN_AD_autoencoder_list"]                      # 8
    epoch_autoencoder_list = settings["epoch_autoencoder_AD_autoencoder_list"]    # 284
    # path_autoencoder_training_parameters = settings["path_autoencoder_training_parameters"] + str(epoch_autoencoder) + '.npy'
    # path_AD_autoencoder_results = settings["path_AD_autoencoder_results"] + str(epochGAN) + "_epochAutoencoder" + str(epoch_autoencoder)
    # try:
    #     os.mkdir(path_AD_autoencoder_results)
    # except:
    #     pass
    #-----------------------------------------------------------------------------------------------------------------------

    # print("epochGAN: ", epochGAN)
    # print("epoch_autoencoder: ", epoch_autoencoder)
    # ad = myADclass(epochGAN, epoch_autoencoder)
    # ad.ADfunc()

    # Only D for choosing the best GAN Epoch
    # for epochGAN in range(100):
    #     path_autoencoder_training_parameters = settings["path_autoencoder_training_parameters"] + '.npy'
    #     path_AD_autoencoder_results = settings["path_AD_autoencoder_results"] + str(epochGAN)
    #     try:
    #         os.mkdir(path_AD_autoencoder_results)
    #     except:
    #         pass
    #     print("\n\nepochGAN: ", epochGAN)
    #     ad = myADclass(epochGAN, 0)
    #     ad.ADfunc()

    # for epochGAN in epochGAN_list:
    #     for epoch_autoencoder in epoch_autoencoder_list:
    #         path_autoencoder_training_parameters = settings["path_autoencoder_training_parameters"] + str(epoch_autoencoder) + '.npy'
    #         path_AD_autoencoder_results = settings["path_AD_autoencoder_results"] + str(epochGAN) + "_epochAutoencoder" + str(epoch_autoencoder)
    #         try:
    #             os.mkdir(path_AD_autoencoder_results)
    #         except:
    #             pass
    #         print("\n\nepochGAN: ", epochGAN)
    #         print("epoch_autoencoder: ", epoch_autoencoder)
    #         ad = myADclass(epochGAN, epoch_autoencoder)
    #         ad.ADfunc()

    for epochGAN in [10]:
        for epoch_autoencoder in range(300):
            path_autoencoder_training_parameters = settings["path_autoencoder_training_parameters"] + str(epoch_autoencoder) + '.npy'
            path_AD_autoencoder_results = settings["path_AD_autoencoder_results"] + str(epochGAN) + "_epochAutoencoder" + str(epoch_autoencoder)
            try:
                os.mkdir(path_AD_autoencoder_results)
            except:
                pass
            print("\n\nepochGAN: ", epochGAN)
            print("epoch_autoencoder: ", epoch_autoencoder)
            ad = myADclass(epochGAN, epoch_autoencoder)
            ad.ADfunc()

    # # Only D
    # for epochGAN in range(100):
    #     path_AD_autoencoder_results = settings["path_AD_autoencoder_results"] + str(epochGAN)
    #     try:
    #         os.mkdir(path_AD_autoencoder_results)
    #     except:
    #         pass
    #     print("\n\nepochGAN: ", epochGAN)
    #     ad = myADclass(epochGAN, 0)
    #     ad.ADfunc()

    print('Main Terminating...')
    end = time() - begin
    print('Testing terminated | Execution time=%d s' % (end))
#-----------------------------------------------------------------------------------------------------------------------

