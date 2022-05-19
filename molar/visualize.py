from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize(hidden_state):
    # print("positive input shape:", len(positive_ids))
    # print("negative input shape:", negative_ids.shape)

    positive_last_hidden_state = hidden_state # (batchsize 128, sequence length 64, hidden size=768)
    # negative_last_hidden_state = negative.last_hidden_state
    print("hidden_ type: ", type(positive_last_hidden_state))
    print("positive hidden_state shape", len(positive_last_hidden_state))
    print("positive hidden_state shape", positive_last_hidden_state.shape)

    # positive_last_hidden_state_np = positive_last_hidden_state.squeeze()
    # negative_last_hidden_state_np = negative_last_hidden_state.squeeze().detach.numpy()
    # print(len(positive_last_hidden_state_np))
    X = positive_last_hidden_state.cpu()
    X = X.reshape(-1, X.shape[-1])
    print(X.shape)
    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=40)
    # try:
    #     positive_embed = pd.DataFrame(tsne.fit_transform(X), columns=['x', 'y'])
    # except ValueError:
    #     print("ValueError") \
    positive_embed = pd.DataFrame(tsne.fit_transform(X), columns=['x', 'y'])

    # negative_embed = pd.DataFrame(tsne.fit_transform(negative_last_hidden_state_np), columns=['x', 'y'])

    plt.figure(figsize=(20, 15))
    fig = sns.scatterplot(data = positive_embed, x='x', y='y')
    plt.show()