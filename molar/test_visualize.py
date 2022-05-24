from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

feature = "../tests/predict_base/base_max.txt"
feature = np.loadtxt(feature)

preds = "../tests/predict_base/preds_base.txt"
preds = np.loadtxt(preds)
preds_df = pd.DataFrame(preds)

n_components = 2
tsne = TSNE(n_components=n_components, perplexity=40)
print(tsne.fit_transform(feature).shape)

embed = pd.DataFrame(tsne.fit_transform(feature))
contrastive_df = pd.concat([embed, preds_df], axis=1)
contrastive_df.columns = ['x', 'y', 'intent']

plt.figure(figsize=(20, 15))
fig = sns.scatterplot(data = contrastive_df, x='x', y='y', hue='intent')
plt.show()