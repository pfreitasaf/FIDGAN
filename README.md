# FID-GAN

Intrusion Detection for Cyber-Physical Systems using Generative Adversarial Networks in Fog Environment
This repository contains code for the paper, Intrusion Detection for Cyber-Physical Systems using Generative Adversarial Networks in Fog Environment, by Paulo Freitas de Araujo-Filho, Georges Kaddoum, Divanilson R. Campelo, Aline Gondim Santos, David Macêdo, andCleber Zanchettin.

Reference Papers and Code
Part of this repository contains code for the paper, MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks, by Dan Li, Dacheng Chen, Jonathan Goh, and See-Kiong Ng.

Quickstart
Python3 and Tensorflow 2.1

Please refer to the paper in order to find where to acquire the three datasets used.

To train the GAN model:
""" python scripts/RGAN.py --settings_file nslkdd/wadi/swat """

To train the Autoencoder model:
""" python scripts/autoencoder.py --settings_file nslkdd/wadi/swat """

To do anomaly detection:
""" python scripts/AD_computeResults.py --settings_file nslkdd_test/wadi_test/swat_test """

Please send me an email at paulo.freitas-de-araujo-filho.1@ens.etsmtl.ca for more information regarding the code or the paper.