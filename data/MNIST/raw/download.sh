#!/bin/bash
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz