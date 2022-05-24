from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def visualize(last_hidden_state, preds):

    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=40)

    embed = pd.DataFrame(tsne.fit_transform(last_hidden_state))
    preds_df = pd.DataFrame(preds)
    contrastive_df = pd.concat([embed, preds_df], axis=1)
    contrastive_df.columns = ['x', 'y', 'intent']

    plt.figure(figsize=(20, 15))
    fig = sns.scatterplot(data = contrastive_df, x='x', y='y', hue="intent")
    plt.show()

def visualize_preprocess(features, test_data):
    features = np.reshape(features, (test_data, -1, 768))
    features_max = np.max(features, axis=1)
    return features_max