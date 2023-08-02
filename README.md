# Weights Reset (WR)

Weights Reset is a simple yet effective regularization technique that prevents overfitting and helps avoid vanishing and exploding gradients in deep neural networks. This GitHub repository contains the implementation of the Weights Reset method in Python, along with an example usage on the Caltech-101 and CIFAR-100 datasets. The code is built using the Keras deep learning framework and includes a simple sequential model architecture. The repository also includes a Jupyter notebooks that demonstrates the effectiveness of the Weights Reset method on the datasets compared to other popular regularization techniques. 

## Paper

https://www.mdpi.com/2079-3197/11/8/148

## Code

| Description | Notebook                   | Open in Colab                                                                  |
|-------------|----------------------------|--------------------------------------------------------------------------------|
| WR vs other regularizations           | [compare_regs.ipynb](./compare_regs.ipynb)       | [link](https://colab.research.google.com/github/amcircle/weights-reset/blob/master/compare_regs.ipynb) |
| WR configs comparison           | [wr_configs.ipynb](./compare_regs.ipynb)       | [link](https://colab.research.google.com/github/amcircle/weights-reset/blob/master/wr_configs.ipynb) |

## Cite paper

```
@Article{computation11080148,
    AUTHOR = {Plusch, Grigoriy and Arsenyev-Obraztsov, Sergey and Kochueva, Olga},
    TITLE = {The Weights Reset Technique for Deep Neural Networks Implicit Regularization},
    JOURNAL = {Computation},
    VOLUME = {11},
    YEAR = {2023},
    NUMBER = {8},
    ARTICLE-NUMBER = {148},
    URL = {https://www.mdpi.com/2079-3197/11/8/148},
    ISSN = {2079-3197},
    DOI = {10.3390/computation11080148}
}
```