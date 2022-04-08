## **Previous Works**

<br>

### 1. Sentencepiece

<br>

1. Construct Vocab

  - Make Corpus
  - Convert csv file to text file.
  - Do Subword segmentation using SentencePiece.

<br>

2. Install SentencePiece

```
pip3 install sentencepiece
```

----

### 2. Transformer (Attention is All You Need)

<br>

<img src = "https://user-images.githubusercontent.com/78716763/154789522-febe9a85-da67-442b-b641-fe79bee9b4c7.png" width="340">

<br>

1. Input Embedding

<br>

2. Position Embedding (+Positional Encoding)

<br>

<img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\bg_black&space;PE_{pos,&space;2i}&space;=&space;sin(pos/10000^{2i/d_{model}})" title="\bg_black PE_{pos, 2i} = sin(pos/10000^{2i/d_{model}})" />

<img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\bg_black&space;PE_{pos,&space;2i&plus;1}&space;=&space;cos(pos/10000^{2i/d_{model}})" title="\bg_black PE_{pos, 2i+1} = cos(pos/10000^{2i/d_{model}})" />

<br>

4. Scaled Dot-Product Attention

<br>

<img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\bg_black&space;Attention(Q,&space;K,&space;V)&space;=&space;softmax(QK^{T}/\sqrt{d_{k}})V" title="\bg_black Attention(Q, K, V) = softmax(QK^{T}/\sqrt{d_{k}})V" />

<br>

3. Multi-head Attention

<br>

<img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\bg_black&space;MultiHead(Q,&space;K,&space;V)&space;=&space;Concat(head_{1},\cdots&space;,&space;head_{h})W^{O}" title="\bg_black MultiHead(Q, K, V) = Concat(head_{1},\cdots , head_{h})W^{O}" />


<img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\bg_black&space;head_{i}&space;=&space;Attention(QW_{i}^{Q},&space;KW_{i}^{K},&space;VW_{i}^{V})" title="\bg_black head_{i} = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})" />

<br>

4. Position-wise Feed-Forward Networks

<br>

<img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\bg_black&space;FFN(x)&space;=&space;max(0,&space;xW_{1}&space;&plus;&space;b_{1})W_{2}&space;&plus;&space;b_{2}" title="\bg_black FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}" />

<br>

5. Encoder & Decoder

----

### 3. BERT

<br>

1. Transfer Learning: Unsupervised Pre-training + Supervised Fine-Tuning

<br>

<img src = "https://user-images.githubusercontent.com/78716763/154789630-7e0624a7-a3aa-47f7-882f-8ee264f2bf4d.png" width="680">

<br>

2. BERT input Embedding: Token Embedding + segmentation Embedding + Position Embedding

<br>

<img src = "https://user-images.githubusercontent.com/78716763/154789774-4f5f5662-aec6-4cb3-8a1f-196310a27051.png" width="680">

<br>

3. Masked Language Model

<br>

4. Next Sentence Prediction

----

### 4. RoBERTa

<br>

1. Train model longer with bigger batches with more data

<br>

2. Full-Sentences without Next Sentence Prediction(NSP)

<br>

3. Dynamically changing masking pattern

<br>

4. Large minibatch

----

### 5. SBERT(Sentence-BERT)

<br>

<img src = "https://user-images.githubusercontent.com/78716763/162454759-5af17a7b-d3ad-4f7d-922f-b575b5b726c3.png" width="680">

<br>

1. Siamese network & Triplet Loss
