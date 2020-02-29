+++
title = "The Annotated The Annotated Transformer"
author = ["KK"]
date = 2019-09-01T16:00:00+08:00
lastmod = 2020-02-29T14:47:38+08:00
tags = ["Machine Learning", "Transformer"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Thanks for the articles I list at the end of this post, I understand how transformers works. These posts are comprehensive, but there are some points that confused me.

First, this is the graph that was referenced by almost all of the post related to Transformer.

{{< figure src="/images/transformer_main.png" width="400" >}}

Transformer consists of these parts: Input, Encoder\*N, Output Input, Decoder\*N, Output. I'll explain them step by step.


## Input {#input}

The input word will map to 512 dimension vector. Then generate Positional Encoding(PE) and add it to the original embeddings.


### Positional Encoding {#positional-encoding}

The transformer model does not contains recurrence and convolution. In order to let the model capture the sequence of input word, it add PE into embeddings.

{{< figure src="/images/transformer_add_pe.png" width="500" >}}

PE will generate a 512 dimension vector for each position:

\\[\begin{align\*}
    PE\_{(pos,2i)} = sin(pos / 10000^{2i/d\_{model}}) \\\\\\
    PE\_{(pos,2i+1)} = cos(pos / 10000^{2i/d\_{model}})
\end{align\*}\\]
The even and odd dimension use `sin` and `cos` function respectively.

For example, the second word's PE should be: \\(sin(2 / 10000^{0 / 512}), cos(2 / 10000^{0 / 512}), sin(2 / 10000^{2 / 512}), cos(2 / 10000^{2 / 512})\text{...}\\)

The value range of PE is `(-1,1)`, and each position's PE is slight different, as `cos` and `sin` has different frequency. Also, for any fixed offset k, \\(PE\_{pos+k}\\) can be represented as a linear function of \\(PE\_{pos}\\).

For even dimension, let \\(10000^{2i/d\_{model}}\\) be \\(\alpha\\), for even dimension:

\\[\begin{aligned}
PE\_{pos+k}&=sin((pos+k)/\alpha) \\\\\\
&=sin(pos/\alpha)cos(k/\alpha)+cos(pos/\alpha)sin(k/\alpha)\\\\\\
&=PE\_{pos\\_even}K\_1+PE\_{pos\\_odd}K\_2
\end{aligned}\\]

{{< figure src="/images/transformer_pe1.png" width="500" >}}

The PE implementation in [tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/5bfe69a7d68b7d61d51fac36c6088f94b9d6fdc6/tensor2tensor/layers/common%5Fattention.py#L457) use `sin` in first half of dimension and `cos` in the rest part of dimension.

{{< figure src="/images/transformer_pe2.png" width="500" >}}


## Encoder {#encoder}

There are 6 Encoder layer in Transformer, each layer consists of two sub-layer: Multi-Head Attention and Feed Forward Neural Network.


### Multi-Head Attention {#multi-head-attention}

Let's begin with single head attention. In short, it maps word embeddings to `q` `k` `v` and use `q` `k` `v` vector to calculate the attention.

The input words map to `q` `k` `v` by multiply the Query, Keys Values matrix. Then for the given Query, the attention for each word in sentence will be calculated by this formula: \\(\mathrm{attention}=\mathrm{softmax}(\frac{qk^T}{\sqrt{d\_k}})v\\), where `q` `k` `v` is a 64 dimension vector.

{{< figure src="/images/transformer_self_attention.png" width="500" >}}

Matrix view:

\\(Attention(Q, K, V) = \mathrm{softmax}(\frac{(XW^Q)(XW^K)^T}{\sqrt{d\_k}})(XW^V)\\) where \\(X\\) is the input embedding.

The single head attention only output a 64 dimension vector, but the input dimension is 512. How to transform back to 512? That's why transformer has multi-head attention.

Each head has its own \\(W^Q\\) \\(W^K\\) \\(W^V\\) matrix, and produces \\(Z\_0,Z\_1...Z\_7\\),(\\(Z\_0\\)'s shape is `(512, 64)`) the concat the outputted vectors as \\(O\\). \\(O\\) will multiply a weight matrix \\(W^O\\) (\\(W^O\\)'s shape is `(512, 512)`) and the result is \\(Z\\), which will be sent to Feed Forward Network.

{{< figure src="/images/transformer_multihead.png" width="500" >}}

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

The whole procedure looks like this:

{{< figure src="/images/transformer_multihead_all.png" width="500" >}}


### Add & Norm {#add-and-norm}

This layer works like this line of code: `norm(x+dropout(sublayer(x)))` or `x+dropout(sublayer(norm(x)))`. The sublayer is Multi-Head Attention or FF Network.


#### Layer Normalization {#layer-normalization}

Layer Norm is similar to Batch Normalization, but it tries to normalize the whole layer's features rather than each feature.(**Scale** and **Shift** also apply for each feature) More details can be found in this [paper](https://arxiv.org/abs/1607.06450).

{{< figure src="/images/transformer_layer_norm.png" width="500" >}}


### Position-wise Feed Forward Network {#position-wise-feed-forward-network}

This layer is a Neural Network whose size is `(512, 2048, 512)`. The exact same feed-forward network is independently applied to each position.

{{< figure src="/images/transformer_encoder.png" width="500" >}}


## Output Input {#output-input}

Same as Input.


## Decoder {#decoder}

The decoder is pretty similar to Encoder. It also has 6 layers, but has 3 sublayers in each Decoder. It add a masked multi-head-attention at the beginning of Decoder.


### Masked Multi-Head Attention {#masked-multi-head-attention}

This layer is used to block future words during training. For example, if the output is `<bos> hello world <eos>`. First, we should use `<bos>` as input to predict `hello`, `hello world <eos>` will be masked to 0.


### Key and Value in Decoder Multi-Head Attention Layer {#key-and-value-in-decoder-multi-head-attention-layer}

In Encoder, the `q` `k` `v` vector is generated by \\(XW^Q\\), \\(XW^K\\) and \\(XW^V\\). In the second sub-layer of Decoder, `q` `k` `v` was generated by \\(XW^Q\\), \\(YW^K\\) and \\(YW^V\\), where \\(Y\\) is the Encoder's output, \\(X\\) is the `<init of sentence>` or previous output.

The animation below illustrates how to apply the Transformer to machine translation.

{{< figure src="/images/transformer_translate.gif" width="500" >}}


## Output {#output}

Using a linear layer to predict the output.


## Ref: {#ref}

1.  [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
2.  [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
3.  [The Transformer – Attention is all you need](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XMb3ZC97FPs)
4.  [Seq2seq pay Attention to Self Attention: Part 2](https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-2-cf81bf32c73d)
5.  [Transformer模型的PyTorch实现](https://juejin.im/post/5b9f1af0e51d450e425eb32d)
6.  [How to code The Transformer in Pytorch](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)
7.  [Deconstructing BERT, Part 2: Visualizing the Inner Workings of Attention](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)
8.  [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
9.  [Dive into Deep Learning - 10.3 Transformer](https://d2l.ai/chapter%5Fattention-mechanisms/transformer.html)
