### **Previous Works**

<br>

#### 1. Sentencepiece

Construct Vocab

1. Make Corpus 
  - Convert csv file to text file.
  - Do Subword segmentation using SentencePiece.

2. Install SentencePiece

```
pip3 install sentencepiece
```

<br>

#### 2. Transformer (Attention is All You Need)

1. Input Embedding
2. Position Embedding

$$ PE_{pos, 2i} = sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{pos, 2i+1} = cos(pos/10000^{2i/d_{model}}) $$

4. Scaled Dot-Product Attention

$$ Attention(Q, K, V) = softmax(QK^{T}/\sqrt{d_{k}})V $$

3. Multi-head Attention

$$ MultiHead(Q, K, V) = Concat(head_{1},\cdots , head_{h})W^{O} $$

$$ head_{i} = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V}) $$

4. Position-wise Feed-Forward Networks

$$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2} $$

5. Encoder & Decoder

<br>

#### 3. BERT

