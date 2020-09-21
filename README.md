# FID-GAN

This repository contains code for the paper, Intrusion Detection for Cyber-Physical Systems using Generative Adversarial Networks in Fog Environment, by Paulo Freitas de Araujo-Filho, Georges Kaddoum, Divanilson R. Campelo, Aline Gondim Santos, David Macêdo, andCleber Zanchettin. This paper was published in the IEEE Internet of Things Journal: https://ieeexplore.ieee.org/document/9199878.

Please contact me at **paulo.freitas-de-araujo-filho.1@ens.etsmtl.ca** for more information regarding our code and paper. You can also cite our work:
**P. F. de Araujo-Filho, G. Kaddoum, D. R. Campelo, A. G. Santos, D. Macêdo and C. Zanchettin, "Intrusion Detection for Cyber-Physical Systems using Generative Adversarial Networks in Fog Environment," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2020.3024800**.



Quickstart
Python3 and Tensorflow 2.1

Please refer to the paper in order to find where to acquire the three datasets used.

To train the GAN model:
""" python scripts/RGAN.py --settings_file nslkdd/wadi/swat """

To train the Autoencoder model:
""" python scripts/autoencoder.py --settings_file nslkdd/wadi/swat """

To do anomaly detection:
""" python scripts/AD_computeResults.py --settings_file nslkdd_test/wadi_test/swat_test """



Part of this repository contains code for the paper: D. Li, D. Chen, B. Jin, L. Shi, J. Goh, and S.-K. Ng, “MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks,” in Springer Int. Conf. on Artif. Neural Netw., 2019, pp. 703–716.
