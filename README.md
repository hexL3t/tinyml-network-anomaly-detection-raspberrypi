# TinyML Network Anomaly Detection (UNSW-NB15 & Raspberry Pi 2)

## Overview
This repository contains the complete design, iteration, data preprocessing pipeline, and deployment artifacts for a dual-layer network anomaly detection system implemented for edge hardware (Raspberry Pi 2, ARMv7). Developed across Assessment 2 (Design) and Assessment 3 (Prototype & Deployment), this project secures resource-constrained Internet of Things (IoT) environments against cyber threats without relying on cloud infrastructure.

## Key Performance Metrics
* **Target Hardware:** Raspberry Pi 2 (ARMv7, 1 GB RAM, 900 MHz quad-core ARM Cortex-A7)
* **Accuracy:** 84.30% (Binary classification: Normal vs. DoS)
* **Memory Footprint:** ~1.6 KB RAM / ~24 KB Flash (Compiled int8 model)
* **Inference Latency:** <1ms per sample
* **Deployment Mode:** Air-gapped edge environment

## Architecture & Methodology
* **Dataset:** UNSW-NB15 network intrusion dataset, filtered and augmented via custom preprocessing scripts.
* **Dual-Layer Defense:** 
  1. *Supervised Neural Network:* Rapid binary classification for known attack vectors.
  2. *Unsupervised K-Means Anomaly Block:* Outlier clustering for zero-day threat identification.
* **Optimization:** Built using Edge Impulse, utilizing EON Tuner for memory and latency constraints, compiled to an optimized C++ library.

## Public Project Link
* **Edge Impulse Studio Version 2:** [View Public Project Repository](https://studio.edgeimpulse.com/)

## Hardware Demonstration
Watch the live prototype demonstration of the model performing real-time inference on the Raspberry Pi 2:

https://github.com/hexL3t/tinyml-network-anomaly-detection-raspberrypi/demonstrations/LiveHardware_Demo.mov

## Repository Structure
* `/assessment`: Contains formal written documentation, reports (e.g., Assessment 2 PDF), and experiment logs.
* `/assets`: System architecture diagrams, UMAP cluster visualisations, hardware photos, and deployment video demonstrations (`demo_video.mp4`).
* `/dataset`: Raw and processed CSV files (`unsw_nb15_augmented_v2.csv`, `unsw_nb15_filtered.csv`) along with mapping and testing subsets.
* `/scripts`: Python automation scripts for data filtering (`filter_dataset_v2.py`), DOS sample augmentation (`augument_dos_samples.py`), live network capture (`live_capture.py`), and local inference execution (`run_inference.py`).

## License
This project is licensed under the 3-Clause BSD License.