+++
title = "Torchtext Snippets"
author = ["KK"]
date = 2019-07-01T21:28:00+08:00
lastmod = 2020-04-02T21:08:27+08:00
tags = ["Python", "torchtext", "PyTorch"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## Load separate files {#load-separate-files}

`data.Field` parameters is [here](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field).

When calling `build_vocab`, torchtext will add `<unk>` in vocabulary list. Set `unk_token=None` if you want to remove it. If `sequential=True` (default), it will add `<pad>` in vocab. `<unk>` and `<pad>` will add at the beginning of vocabulary list by default.

`LabelField` is similar to Field, but it will set `sequential=False`, `unk_token=None` and `is_target=Ture`

```python
INPUT = data.Field(lower=True, batch_first=True)
TAG = data.LabelField()

train, val, test = data.TabularDataset.splits(path=base_dir.as_posix(), train='train_data.csv',
                                                validation='val_data.csv', test='test_data.csv',
                                                format='tsv',
                                                fields=[(None, None), ('input', INPUT), ('tag', TAG)])
```


## Load single file {#load-single-file}

```python
all_data = data.TabularDataset(path=base_dir / 'gossip_train_data.csv',
                               format='tsv',
                               fields=[('text', TEXT), ('category', CATEGORY)])
train, val, test = all_data.split([0.7, 0.2, 0.1])
```


## Create iterator {#create-iterator}

```python
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(32, 256, 256), shuffle=True,
    sort_key=lambda x: x.input)
```


## Load pretrained vector {#load-pretrained-vector}

```python
vectors = Vectors(name='cc.zh.300.vec', cache='./')

INPUT.build_vocab(train, vectors=vectors)
TAG.build_vocab(train, val, test)
```


## Check vocab sizes {#check-vocab-sizes}

You can view vocab index by `vocab.itos`.

```python
tag_size = len(TAG.vocab)
```


## Use field vector in model {#use-field-vector-in-model}

```python
vec = INPUT.vocab.vectors

class Model:
    nn.Embedding.from_pretrained(vec, freeze=False)
```


## Convert text to vector {#convert-text-to-vector}

```python
s = ' '.join(segmentize(s))
s = INPUT.preprocess(s)
vec = INPUT.process([s])
```
