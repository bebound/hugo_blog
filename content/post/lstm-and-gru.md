+++
title = "LSTM and GRU"
author = ["kk"]
date = 2018-04-22T14:39:00+08:00
tags = ["machine learning", "LSTM", "GRU"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## LSTM {#lstm}

The avoid the problem of vanishing gradient and exploding gradient in vanilla RNN, LSTM was published, which can remember information for longer periods of time.

Here is the structure of LSTM:
![](/images/LSTM_LSTM.png)

The calculate procedure are:

\\[\begin{aligned}
f\_t&=\sigma(W\_f\cdot[h\_{t-1},x\_t]+b\_f)\\\\\\
i\_t&=\sigma(W\_i\cdot[h\_{t-1},x\_t]+b\_i)\\\\\\
o\_t&=\sigma(W\_o\cdot[h\_{t-1},x\_t]+b\_o)\\\\\\
\tilde{C\_t}&=tanh(W\_C\cdot[h\_{t-1},x\_t]+b\_C)\\\\\\
C\_t&=f\_t\ast C\_{t-1}+i\_t\ast \tilde{C\_t}\\\\\\
h\_t&=o\_t \ast tanh(C\_t)
\end{aligned}\\]

\\(f\_t\\),\\(i\_t\\),\\(o\_t\\) are forget gate, input gate and output gate respectively. \\(\tilde{C\_t}\\) is the new memory content. \\(C\_t\\) is cell state. \\(h\_t\\) is the output.

Use \\(f\_t\\) and \\(i\_t\\) to update \\(C\_t\\), use \\(o\_t\\) to decide which part of hidden state should be outputted.


## GRU {#gru}

{{< figure src="/images/LSTM_GRU.png" >}}

\\[\begin{aligned}
z\_t&=\sigma(W\_z\cdot[h\_{t-1},x\_t])\\\\\\
r\_t&=\sigma(W\_r\cdot[h\_{t-1},x\_t])\\\\\\
\tilde{h\_t}&=tanh(W\cdot[r\_t \ast h\_{t-1},x\_t])\\\\\\
h\_t&=(1-z\_t)\ast h\_{t-1}+z\_t \ast \tilde{h\_t}
\end{aligned}\\]

\\(z\_t\\) is update gate, \\(r\_t\\) is reset gate, \\(\tilde{h\_t}\\) is candidate activation, \\(h\_t\\) is activation.

Compare with LSTM, GRU merge cell state and hidden state to one hidden state, and use \\(z\_t\\) to decide how to update the state rather than \\(f\_t\\) and \\(i\_t\\).

Ref:

1.  [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
