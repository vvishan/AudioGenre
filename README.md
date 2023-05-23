# Music Genre Classification

# About the dataset:
It consists of 1000 audio files each having 30 seconds duration. There are 10 classes ( 10 music genres) each containing 100 audio tracks. Each track is in .wav format. It contains audio files of the following 10 genres:

Blues
Classical
Country
Disco
Hiphop
Jazz
Metal
Pop
Reggae
Rock

# Music Genre Classification approach:
There are various methods to perform classification on this dataset. Some of these approaches are:

Multiclass support vector machines
K-means clustering
K-nearest neighbors
Convolutional neural networks
We will use K-nearest neighbors algorithm because in various researches it has shown the best results for this problem.

K-Nearest Neighbors is a popular machine learning algorithm for regression and classification. It makes predictions on data points based on their similarity measures i.e distance between them.

# Feature Extraction:
The first step for music genre classification project would be to extract features and components from the audio files. It includes identifying the linguistic content and discarding noise.

1. Imports:
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random 
import operator

import math
import numpy as np
