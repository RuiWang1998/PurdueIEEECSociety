import sklearn
from sklearn.manifold import TSNE

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_embedded = TSNE(2).fit_transform(outputs)
X_embedded.shape


X_PCA = PCA(3).fit_transform(outputs)
print(X_PCA.shape)

#X_embedded = TSNE(2).fit_transform(X_PCA)
#print(X_embedded.shape)

color = 0
for i in range(len((X_embedded))):
  el = X_embedded[i]
  if i % 51 == 0 and not i==0:
    color+=1
    color=color%10
  plt.scatter(el[0], el[1], color="C" + str(color))


def model_distance(model, pic1, pic2):
    file1 = (pic1)
    inp1 = create_input_rgbd(file1)
    file1 = (pic2)
    inp2 = create_input_rgbd(file1)

    return model.predict([inp1, inp2])