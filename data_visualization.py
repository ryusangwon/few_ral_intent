from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

input_sequence = "This article is about the online encyclopedia. For Wikipedia's home page, see Main Page. For the English edition, see English Wikipedia."
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
input_tokens = tokenizer(input_sequence, return_tensors="pt")
print(input_tokens)

config = RobertaConfig.from_pretrained("roberta-base")

model = RobertaModel.from_pretrained("roberta-base")
test_model = RobertaModel(config)

# print(model(tokenizer(input)))
print(input_tokens)
print(type(input_tokens))

output = model(**input_tokens)

last_hidden_state = output.last_hidden_state
print(last_hidden_state)
print(last_hidden_state.shape) # [batch_size, 문장길이, 단어벡터의 차원]
last_hidden_state_np = last_hidden_state.squeeze().detach().numpy()

print(last_hidden_state_np.shape)

import seaborn as sns

n_components=2
tsne = TSNE(n_components=n_components)


model = TSNE(n_components= 2,init='pca',perplexity=40)
embed = pd.DataFrame(model.fit_transform(last_hidden_state_np), columns=['x', 'y'])

plt.figure(figsize=(20, 15))
fig = sns.scatterplot(data = embed, x='x',y="y")
plt.show()