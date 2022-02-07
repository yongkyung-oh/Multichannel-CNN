# [Multichannel Convolution Neural Network for Gas Mixture Classification](https://yongkyung-oh.github.io/Multichannel-CNN/)
(Working) YongKyung Oh, Chiehyeon Lim, Junghye Lee, Sewon Kim and SungIl Kim, **Multichannel Convolution Neural Network for Gas Mixture Classification**

Corresponding author: SungIl Kim (sungil.kim@unist.ac.kr)

## Abstract
Concomitant with people beginning to understand their legal rights or entitlement to complain, complaints of offensive odors and smell pollution have increased significantly. Consequently, monitoring gases and identifying their types and causes in real time has become a critical issue in the modern world. In particular, toxic gases that may be generated at industrial sites or odors in daily life consist of hybrid gases made up of various chemicals. Understanding the types and characteristics of these mixed gases is an important issue in many areas. However, mixed gas classification is challenging because the gas sensor arrays for mixed gases must process complex nonlinear high-dimensional data. In addition, obtaining sufficient training data is expensive. To overcome these challenges, this paper proposes a novel method for mixed gas classification based on analogous image representations with multiple sensor-specific channels and a convolutional neural network (CNN) classifier. The proposed method maps a gas sensor array into a multichannel image with data augmentation, and then utilizes a CNN for feature extraction from such images. The proposed method was validated using public mixture gas data from the UCI machine learning repository and real laboratory experiments. The experimental results indicate that it outperforms the existing classification approaches in terms of the balanced accuracy and weighted F1 scores. Additionally, we evaluated the performance of the proposed method in various experimental settings in terms of data representation, data augmentation, and parameter initialization, so that practitioners can easily apply it to artificial olfactory systems.


## Methodology
### Overview
![method](figs/method_fig_1.jpg "Method")

### Input adaptation
<p align="center"><img width="60%" src="figs/method_fig_2.jpg" /></p>

### Experiment reuslts 
<p align="center"><img width="60%" src="figs/result.jpg" /></p>
