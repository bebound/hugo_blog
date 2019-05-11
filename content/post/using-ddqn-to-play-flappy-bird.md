+++
title = "Architectures"
author = ["KK"]
lastmod = 2019-05-11T22:36:01+08:00
tags = ["machine learning", "word2vec"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

#### Hierarchical Softmax {#hierarchical-softmax}

Encode words into a huffman tree, then each word has a Huffman code. The probability of it's probability \\(P(w\lvert Context(\omega))\\) can change to choose the right path from root the the leaf node, each node is a binary classification. Suppose code \\(0\\) is a positive label, \\(1\\) is negative label. If the probability of a positive classification is
\\[\sigma(X^T\_\omega \theta)=\frac{1}{1+e^{-X^T\_\omega}}\\]

Then the probability of negative classification is
\\[1-\sigma(X^T\_\omega \theta)\\]

<img src="/images/doc2vec_hierarchical_softmax.png" alt="doc2vec_hierarchical_softmax.png" width="400" />
足球's Huffman code is \\(1001\\), then it's probability in each node are

\\[\begin{aligned}
p(d\_2^\omega\lvert X\_\omega,\theta^\omega\_1&=1-\sigma(X^T\_\omega \theta^\omega\_1))\\\\\\
p(d^\omega\_3\lvert X\_\omega,\theta^\omega\_2&=\sigma(X^T\_\omega \theta^\omega\_2))\\\\\\
p(d^\omega\_4\lvert X\_\omega,\theta^\omega\_3&=\sigma(X^T\_\omega \theta^\omega\_3))\\\\\\
p(d^\omega\_5\lvert X\_\omega,\theta^\omega\_4&=1-\sigma(X^T\_\omega \theta^\omega\_4))\\\\\\
\end{aligned}\\]

where \\(\theta\\) is parameter in the node.

The probability of the `足球` is the production of these equation.

Generally,

\\[p(\omega\lvert Context(\omega))=\prod\_{j=2}^{l\omega}p(d^\omega\_j\lvert X\_\omega,\theta^\omega\_{j-1})\\]


#### Negative Sampling {#negative-sampling}

Choose some negative sample, add the probability of the negative word into loss function. Maximize the positive words' probability and minimize the negative words' probability.

Let \\(P(D=0 \lvert \omega,c)\\) be the probability that \\((\omega,c)\\) did not come from the corpus data. Then the objective function will be

\\[\theta = \text{argmax} \prod\_{(\omega,c)\in D} P(D=1\lvert \omega,c,\theta) \prod\_{(\omega,c)\in \tilde{D}} P(D=0\lvert \omega,c,\theta)\\]

where \\(\theta\\) is the parameters of the model(\\(\upsilon\\) and \\(u\\)).

Ref:

-   [word2vec原理推导与代码分析](<http://www.hankcs.com/nlp/word2vec.html>)
-   [CS 224D: Deep Learning for NLP Lecture Notes: Part I](<http://cs224d.stanford.edu/lecture%5Fnotes/notes1.pdf>)
-   [word2vec 中的数学原理详解（一）目录和前言](<http://blog.csdn.net/itplus/article/details/37969519>)


## <span class="org-todo done DONE">DONE</span> Parameters in dov2vec {#parameters-in-dov2vec}

Here are some parameter in `gensim`'s `doc2vec` class.


#### window {#window}

window is the maximum distance between the predicted word and context words used for prediction within a document. It will look behind and ahead.

In `skip-gram` model, if the window size is 2, the training samples will be this:(the blue word is the input word)

{{< figure src="/images/doc2vec_window.png" width="400" >}}


#### min\_count {#min-count}

If the word appears less than this value, it will be skipped


#### sample {#sample}

High frequency word like `the` is useless for training. `sample` is a threshold for deleting these higher-frequency words. The probability of keeping the word \\(w\_i\\) is:

\\[P(w\_i) = (\sqrt{\frac{z(\omega\_i)}{s}} + 1) \cdot \frac{s}{z(\omega\_i)}\\]

where \\(z(w\_i)\\) is the frequency of the word and \\(s\\) is the sample rate.

This is the plot when `sample` is 1e-3.

{{< figure src="/images/doc2vec_negative_sample.png" width="400" >}}


#### negative {#negative}

Usually, when training a neural network, for each training sample, all of the weights in the neural network need to be tweaked. For example, if the word pair is ('fox', 'quick'), then only the word quick's neurons should output 1, and all of the other word neurons should output 0.

But it would takes a lot of time to do this when we have billions of training samples. So, instead of update all of the weight, we random choose a small number of "negative" words (default value is 5) to update the weight.(Update their wight to output 0).

So when dealing with word pair ('fox','quick'), we update quick's weight to output 1, and other 5 random words' weight to output 1.

The probability of selecting word \\(\omega\_i\\) is \\(P(\omega\_i)\\):

\\[P(\omega\_i)=\frac{{f(\omega\_i)}^{{3}/{4}}}{\sum\_{j=0}^{n}\left({f(\omega\_j)}^{{3}/{4}}\right)}\\]

\\(f(\omega\_j)\\) is the frequency of word \\(\omega\_j\\).

Ref:

-   [Word2Vec Tutorial - The Skip-Gram Model](<http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/>)
-   [Word2Vec Tutorial Part 2 - Negative Sampling](<http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/>)


## <span class="org-todo done DONE">DONE</span> Semi-supervised text classification using doc2vec and label spreading {#semi-supervised-text-classification-using-doc2vec-and-label-spreading}

Here is a simple way to classify text without much human effort and get a impressive performance.

It can be divided into two steps:

1.  Get train data by using keyword classification
2.  Generate a more accurate classification model by using doc2vec and label spreading


#### Keyword-based Classification {#keyword-based-classification}

Keyword based classification is a simple but effective method. Extracting the target keyword is a monotonous work. I use this method to automatic extract keyword candidate.

1.  Find some most common words to classify the text.
2.  Use this equation to calculate the score of each word appears in the text.
    \\[ score(i) = \frac{count(i)}{all\\\_count(i)^{0.3}}\\]
    which \\(all\\\_count(i)\\) is the word i's word count in all corpus, and \\(count(i)\\) is the word i's word count in positive corpus.
3.  Check the top words, add it to the final keyword list. Repeat this process.

Finally, we can use the keywords to classify the text and get the train data.


#### Classification by `doc2vec` and Label Spreading {#classification-by-doc2vec-and-label-spreading}

Keyword-based classification sometimes produces the wrong result, as it can't using the semantic information in the text. Fortunately, Google has open sourced `word2vec`, which can be used to produce semantically meaningful word embeddings. Furthermore, sentences can also be converted to vectors by using `doc2vec`. Sentences which has closed meaning also have short vector distance.

So the problem is how to classify these vectors.

1.  Using corpus to train the `doc2vec` model.
2.  Using `doc2vec` model to convert sentence into vector.
3.  Using label spreading algorithm to train a classify model to classify the vectors.


## <span class="org-todo done DONE">DONE</span> TextCNN with PyTorch and Torchtext on Colab {#textcnn-with-pytorch-and-torchtext-on-colab}

[PyTorch](https://pytorch.org) is a really powerful framework to build the machine learning models. Although some features is missing when compared with TensorFlow (For example, the early stop function, History to draw plot), its code style is more intuitive.

[Torchtext](https://github.com/pytorch/text) is a NLP package which is also made by `=pytorch=` team. It provide a way to read text, processing and iterate the texts.

[Google Colab](https://colab.research.google.com) is a Jupyter notebook environment host by Google, you can use free GPU and TPU to run your modal.

Here is a simple tuturial to build a TextCNN modal and run it on Colab.

The [TextCNN paper](https://arxiv.org/abs/1408.5882) was published by Kim in 2014. The model's idea is pretty simple, but the performance is impressive. If you trying to solve the text classificaton problem, this model is a good choice to start with.

The main architecture is shown below:

{{< figure src="/images/textcnn.png" width="400" >}}

It uses different kernels to extract text features, then use the softmax regression to classify text base on the features.

Now we can build this model step by step.

First build the model. The model I use is CNN-multichannel, which contains two sets of word embedding. Both of them is the copy of word embedding generate from corpus, but only one set will update embedding during training.

The code is below:

```python
class textCNNMulti(nn.Module):
    def __init__(self,args):
        super().__init__()
        dim = args['dim']
        n_class = args['n_class']
        embedding_matrix=args['embedding_matrix']
        kernels=[3,4,5]
        kernel_number=[150,150,150]
        self.static_embed = nn.Embedding.from_pretrained(embedding_matrix)
        self.non_static_embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.convs = nn.ModuleList([nn.Conv2d(2, number, (size, dim),padding=(size-1,0)) for (size,number) in zip(kernels,kernel_number)])
        self.dropout=nn.Dropout()
        self.out = nn.Linear(sum(kernel_number), n_class)

    def forward(self, x):
        non_static_input = self.non_static_embed(x)
        static_input = self.static_embed(x)
        x = torch.stack([non_static_input, static_input], dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.out(x)
        return x
```

Second, convert text into word index, so each sentence become a vector for training.

```python

TEXT = data.Field(lower=True,batch_first=True)
LABEL = data.Field(sequential=False)

train, val, test = datasets.SST.splits(TEXT, LABEL, 'data/',fine_grained=True)

TEXT.build_vocab(train, vectors="glove.840B.300d")
LABEL.build_vocab(train,val,test)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(128, 256, 256),shuffle=True)

```

`=Field=` defines how to process text, here is the most common parameters:

> sequential – Whether the datatype represents sequential data. If False, no tokenization is applied. Default: True.
>
> use\_vocab – Whether to use a Vocab object. If False, the data in this field should already be numerical. Default: True.
>
> preprocessing – The Pipeline that will be applied to examples using this field after tokenizing but before numericalizing. Many Datasets replace this attribute with a custom preprocessor. Default: None.
>
> batch\_first – Whether to produce tensors with the batch dimension first. Default: False.

`=datasets.SST.splits=` will load the `=SST=` datasets, and split into train, validation, and test Dataset objects.

`=build_vocab=` will create the Vocab object for Field, which contains the information to convert word into word index and vice versa. Also, the word embedding will save as `=Field.Vocab.vectors=`. `=vectors=` contains all of the word embedding. Torchtext can download some pretrained vectors automatically, such as `=glove.840B.300d=`, `=fasttext.en.300d=`. You can also load your vectors in this way, `=xxx.vec=` should be the standard word2vec format.

```python
from torchtext.vocab import Vectors

vectors = Vectors(name='xxx.vec', cache='./')
TEXT.build_vocab(train, val, test, vectors=vectors)
```

`=data.BucketIterator.splits=` will returns iterators that loads batches of data from datasets, and the text in same batch has similar lengths.

Now, we can start to train the model. First we wrap some parameters into `=args=`, it contains settings like output class, learning rate, log interval and so on.

```python
args={}
args['vocb_size']=len(TEXT.vocab)
args['dim']=300
args['n_class']=len(LABEL.vocab)-1
args['embedding_matrix']=TEXT.vocab.vectors
args['lr']=0.001
args['momentum']=0.8
args['epochs']=180
args['log_interval']=100
args['test_interval']=500
args['save_dir']='./'
```

Finally, we can train the model.

```python
model=textCNNMulti(args)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'],momentum=args['momentum'])
criterion = nn.CrossEntropyLoss()
steps=0
for epoch in range(1, args['epochs']+1):
    for i,data in enumerate(train_iter):
        steps+=1

        x, target = data.text, data.label
        x=x.cuda()

        target.sub_(1)
        target=target.cuda()

        output = model(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

You can found `textcnn.ipynb` [here](https://github.com/bebound/textcnn).

Ref:

1.  [Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882)
2.  [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
3.  [Torchtext Docs](https://torchtext.readthedocs.io/en/latest/data.html)
4.  [Castor](https://github.com/castorini/Castor)
