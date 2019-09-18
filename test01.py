# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA as ICA
from pydub import AudioSegment
from pydub.playback import play

# song = AudioSegment.from_wav("piano_C4.wav")
# play(song*3)



plt.plot(np.arange(1000)**2)

signal1 = np.sin(np.arange(256)*0.15)
signal2 = np.sin(np.arange(256)*0.06)


combinations = np.dot(np.random.random((200,2)), np.vstack((signal1,signal2)))
combinations += np.random.random(combinations.shape)*.1

PCA_t = PCA(n_components = 10)
PCA_t.fit(combinations)
ideal_n = np.where(np.cumsum(PCA_t.explained_variance_ratio_) > .99)[0][0] +1

ICA_t = ICA(n_components = ideal_n, tol = 1e-8)
scores = ICA_t.fit_transform(combinations)
mixings =  ICA_t.mixing_
loadings = ICA_t.components_

# tops = mixings



plt.close()
fig, ax = plt.subplots(2,2)
ax[0,0].plot(signal1)
ax[0,1].plot(signal2)
for i in range(combinations.shape[0]):
    ax[1,0].plot(combinations[i,:])

for i in range(loadings.shape[0]):
    ax[1,1].plot(loadings[i,:])

