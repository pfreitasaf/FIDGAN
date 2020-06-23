import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pickle
import utils
from time import time

#plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['font.family'] = 'serif'

begin = time()

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# Functions ------------------------------------------------------------------------------------------------------------
def loadDRLosses(savePath_DL1, savePath_DL2, savePath_LL, savePath_RL):
    DL1 = np.load(savePath_DL1)
    DL2 = np.load(savePath_DL2)
    LL = np.load(savePath_LL)
    RL = np.load(savePath_RL)
    return DL1, DL2, LL, RL

def computeDRScores(DL1, DL2, RL):    
    DL1 = (-1) * DL1
    DL2 = (-1) * DL2
    # RL =  (-1) * RL
    drScores = []
    dr_score = DL1
    drScores.append(dr_score)
    dr_score = DL2
    drScores.append(dr_score)
    dr_score = RL
    drScores.append(dr_score)
    for l in range(1,100):
        lam = l*0.01
        dr_score = (lam * DL1) + ((1 - lam) * RL)
        drScores.append(dr_score)
        dr_score = (lam * DL2) + ((1 - lam) * RL)
        drScores.append(dr_score)
    return drScores

def computeCombDRScores(DL1, DL2, RL):
    DL1 = (-1) * DL1
    DL2 = (-1) * DL2
    drScores = []
    for l in range(1,100):
        lam = l*0.01
        dr_score = (lam * DL1) + ((1 - lam) * RL)
        drScores.append(dr_score)
        dr_score = (lam * DL2) + ((1 - lam) * RL)
        drScores.append(dr_score)
    return drScores

def computeDScores(DL1, DL2):
    DL1 = (-1) * DL1
    DL2 = (-1) * DL2
    drScores = []
    dr_score = DL1
    drScores.append(dr_score)
    dr_score = DL2
    drScores.append(dr_score)
    return drScores

def computeRScores(RL):
    drScores = []
    dr_score = RL
    drScores.append(dr_score)
    return drScores

def computeAucRocs(LL, drScores):
    aucRocMax = 0
    aucRocMaxIndex = 0
    index = 0
    for dr_score in drScores:
        aucRoc = roc_auc_score(LL, dr_score)
        print("aucRoc: ", aucRoc)
        if(aucRocMax < aucRoc):
            aucRocMax = aucRoc
            aucRocMaxIndex = index
        index = index + 1
    print("aucRocMaxIndex: ", aucRocMaxIndex)
    print("aucRocMax: ", aucRocMax)
    return aucRocMaxIndex, aucRocMax

def plotAucRoc(LL, drScores, aucRocMaxIndex, savePathPlotAucRoc, label, linestyle, marker, color, markersize):
    fpr, tpr, thresholds = metrics.roc_curve(LL, drScores[aucRocMaxIndex], pos_label=1)

    fpr2 = []
    tpr2 = []
    fpr2.append(fpr[0])
    tpr2.append(tpr[0])
    for i in range(1, fpr.size):
        d = pow(( pow(fpr2[-1] - fpr[i], 2) + pow(tpr2[-1] - tpr[i], 2) ), 0.5)
        if( d > 0.07 ):
            fpr2.append(fpr[i])
            tpr2.append(tpr[i])

    # font_path = "Times_New_Roman.ttf"
    # prop = font_manager.FontProperties(fname=font_path)
    # tnrFont = {'fontname':'Times_New_Roman'}

    prop = fm.FontProperties(fname='times_new_roman.ttf')
    # ax.set_title('This is some random font', fontproperties=prop, size=32)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr2, tpr2, label=label, linestyle=linestyle, marker=marker, color=color, linewidth=1, markersize=markersize)
    # plt.xlabel('False positive rate', **tnrFont, fontsize=10, fontproperties=prop)
    plt.xlabel('False positive rate', fontproperties=prop, size=14)
    plt.ylabel('True positive rate', fontproperties=prop, size=14)
    plt.title('ROC curves', fontproperties=prop, size=14)
    plt.legend(loc='best', prop=prop, fontsize=25)
    plt.show()
    plt.savefig(savePathPlotAucRoc)

#-----------------------------------------------------------------------------------------------------------------------

# Main Executtion ------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print('Main Starting...')
    
    for epochGAN in [10]:
        for epoch_autoencoder in range(15):
        # for epoch_autoencoder in range(0,35):
            # for epoch_autoencoder in [5]:
            path_AD_autoencoder_results = settings["path_AD_autoencoder_results"] + str(epochGAN) + "_epochAutoencoder" + str(epoch_autoencoder)
            savePath_DL1 = path_AD_autoencoder_results + "/DL1.npy"
            savePath_DL2 = path_AD_autoencoder_results + "/DL2.npy"
            savePath_LL = path_AD_autoencoder_results + "/LL.npy"
            savePath_RL = path_AD_autoencoder_results + "/RL.npy"
            savePathPlotAucRoc = settings["path_savePlotAucRoc"] + "/ROC_Curve_" + str(epochGAN) + "_" + str(epoch_autoencoder) + ".pdf"

            print("\epochGAN: ", epochGAN)
            print("epoch_autoencoder: ", epoch_autoencoder)
            DL1, DL2, LL, RL = loadDRLosses(savePath_DL1, savePath_DL2, savePath_LL, savePath_RL)
            RL = -RL

            print("Scores with only D")
            drScores_D = computeDScores(DL1, DL2)
            aucRocMaxIndex_D, aucRocMax_D = computeAucRocs(LL, drScores_D)
            labelOnlyD = 'ROC Curve Only D; AUC = ' + str('%.4f'%(aucRocMax_D))

            print("Scores with only G")
            drScores_G = computeRScores(RL)
            aucRocMaxIndex_G, aucRocMax_G = computeAucRocs(LL, drScores_G)
            labelOnlyR = 'ROC Curve Only G; AUC = ' + str('%.4f'%(aucRocMax_G))

            print("Scores combining D and G")
            drScores_CombDG = computeCombDRScores(DL1, DL2, RL)
            aucRocMaxIndex_CombDG, aucRocMax_CombDG = computeAucRocs(LL, drScores_CombDG)
            labelCombDR = 'ROC Curve D and G Combined; AUC = ' + str('%.4f'%(aucRocMax_CombDG))

            f = open(settings["path_savePlotAucRoc"] + "/aucRocMaxValues.txt", "a")
            f.write("\nepochGAN: " + str(epochGAN) + "epoch_autoencoder: " + str(epoch_autoencoder) + "   -   aucRocMax_D: "  + str(aucRocMax_D) + ", aucRocMax_G: "  + str(aucRocMax_G) + ", aucRocMax_CombDG: "  + str(aucRocMax_CombDG))
            f.close()

            # plotAucRoc(LL, drScores_D, aucRocMaxIndex_D, savePathPlotAucRoc, labelOnlyD, linestyle=':', marker='o', color='r', markersize = 6)
            # plotAucRoc(LL, drScores_G, aucRocMaxIndex_G, savePathPlotAucRoc, labelOnlyR, linestyle='-.', marker='s', color='g', markersize = 7)
            # plotAucRoc(LL, drScores_CombDG, aucRocMaxIndex_CombDG, savePathPlotAucRoc, labelCombDR, linestyle='--', marker='^', color='b', markersize = 6)

    print('Main Terminating...')
    end = time() - begin
    print('Testing terminated | Execution time=%d s' % (end))
#-----------------------------------------------------------------------------------------------------------------------

