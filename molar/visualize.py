from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize(last_hidden_state, labels):

    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=40)
    print(tsne.fit_transform(last_hidden_state).shape)

    positive_embed = pd.DataFrame(tsne.fit_transform(last_hidden_state))

    plt.figure(figsize=(20, 15))
    fig = sns.scatterplot(data = positive_embed, x='x', y='y', hue="x")
    plt.show()