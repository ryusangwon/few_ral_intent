#!/usr/bin/env python
# coding: utf-8

# In[5]:


# sentencepiece 다루기

import os
import pandas as pd
import sentencepiece as spm
import sys
import csv


# In[6]:


csv.field_size_limit(sys.maxsize)


# In[15]:


# in_file의 csv파일을 txt파일로 변환한다.

in_file = '../web-crawler/kowiki/kowiki_20220216.csv' # csv file 주소 입력
out_file = './kor_wiki.txt'
SEPARATOR = u"\u241D"

df = pd.read_csv(in_file, sep=SEPARATOR, engine='python')
with open(out_file, 'w') as f:
    for index, row in df.iterrows():
        f.write(row['text']) # title은 저장하지 않고 text만 저장한다.
        f.write('\n\n\n\n') # 구분자


# In[16]:


# .model 파일과 .vocab 파일을 생성한다.

corpus = 'kor_wiki.txt'
prefix = 'kor_wiki'
vocab_size = 8000
spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=prefix,
    vocab_size=vocab_size+7)


# In[17]:


vocab_file = './kor_wiki.model'
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

lines = [
    '한국어를 공부해보겠습니다',
    '문장 단위로 나누어 볼께요',
    '문맥을 이해하는 것이 중요해요',
]

for line in lines:
    pieces = vocab.encode_as_pieces(line)
    ids = vocab.encode_as_ids(line)
    print(line)
    print(pieces)
    print(ids)
    print()


# In[ ]:




