# Source code and datasets for the paper "FreeTab-WF: Scaling Website Fingerprinting from Single-Tab to Multi-Tab".
The usage of this model's code is as follows, and the dataset can be obtained via [the shared link](https://drive.google.com/file/d/1hhuG6Cjzwz9hKF2w0DFjxWbdNSGCKURO/view?usp=sharing) in Google Drive Cloud.

## Basic Environment:
GPU: NVIDIA GeForce RTX 4060 (8GB)  
Python Version: 3.12  
PyTorch Version: 2.5.0 
CUDA Version: 12.4

## Description & Useage

### Install
You can use conda commands in the virtual environment provided by Anaconda to install the basic packages mentioned in the code that need to be loaded (such as numpy, os, sys, etc.).

### Dataset Detals
We use the tshark tool for automatic packet capture. We select 100 popular websites as monitored sites, among which 50 are used for Chrome-based collection and the remaining 50 for Tor-based collection. For each monitored website, we randomly select 10 subpages as monitored pages.

**Single-Tab Datasets:**

We collect 100 traffic samples for the homepage of each monitored website, and 10 traffic samples for each of its 10 subpages. These single-tab traces serve as the pool for training composition.

**Multi-Tab Datasets:**

We randomly sample website combinations from the monitored set. As a result, our dataset covers diverse browsing patterns, including traces composed entirely of homepages, entirely of subpages, or mixtures of both. A browsing sequence may also visit both the homepage and a subpage of the same website within a single session. For instance, a browsing sequence such as **"Homepage A → Subpage C5 → Homepage B → Subpage B1 → Subpage A6"** contains 5 webpage visits, yet corresponds to only 3 distinct websites (A, B, and C). Such a trace is treated as a 3-tab sample under our labeling scheme. This design significantly enriches the compositional diversity of our dataset and better reflects real-world user browsing behavior.

**Open-World Datasets:**

We selected 12,000 websites that are not included in our monitored website list and collected randomly composed 2-, 3-, 4-, 5-, and 6-tab traffic samples, totaling 24,000 samples. Additionally, when visiting these websites, we randomly open subpages for some of them to better reflect real-world browsing behavior.

### Training Dataset Generation
We design an **Online Random Composition** method to generate the training dataset. At each training iteration, this method dynamically selects single-tab traffic samples from the pool of collected traces—covering both homepages and subpages—and concatenates them into synthetic multi-tab samples in real time. This approach exposes the model to a virtually unlimited variety of website combinations without requiring real concurrent traffic. The detailed procedure is described in the paper and implemented in **Online_Random_Composition.py**.

### Model Training & Evaluation
You can download the dataset from the shared link above and use the provided scripts for model training and evaluation. We provide code for both closed-world and open-world evaluation settings, as well as scripts for experiments with and without fine-tuning.

During evaluation, results are reported separately for 2-, 3-, 4-, and 5-tab test sets for analytical purposes only. The model itself does not take the number of tabs as input, nor does it rely on any post-processing that assumes prior knowledge of the tab count. This grouping is purely for statistical breakdown; the same trained model is applied uniformly across all test samples regardless of the actual number of tabs.
