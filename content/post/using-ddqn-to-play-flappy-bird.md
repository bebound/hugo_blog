+++
title = "Using Dueling DQN to Play Flappy Bird"
author = ["KK"]
date = 2019-04-14T17:10:00+08:00
lastmod = 2019-07-26T23:38:41+08:00
tags = ["Machine Learning"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

PyTorch provide a simple DQN implementation to solve the cartpole game. However, the code is incorrect, it diverges after training (It has been discussed [here](https://discuss.pytorch.org/t/dqn-example-from-pytorch-diverged/4123)).

The official code's training data is below, it's high score is about 50 and finally diverges.

{{< figure src="/images/ddqn_official.png" width="400" >}}

There are many reason that lead to divergence.

First it use the difference of two frame as input in the tutorial, not only it loss the cart's absolute information(This information is useful, as game will terminate if cart moves too far from centre), but also confused the agent when difference is the same but the state is varied.

Second, small replay memory. If the memory is too small, the agent will forget the strategy it has token in some state. I'm not sure whether `10000` memory is big enough, but I suggest using a higher value.

Third, the parameters. `learning_rate`, `target_update_interval` may cause fluctuation. Here is a example on [stackoverflow](https://stackoverflow.com/questions/49837204/performance-fluctuates-as-it-is-trained-with-dqn). I also met this problem when training cartpole agent. The reward stops growing after 1000 episode.

{{< figure src="/images/ddqn_cartpole_fluctuate.png" width="400" >}}

After doing some research on the cartpole DNQ code, I managed to made a model to play the flappy bird. Here are the changes from the original cartpole code. Most of the technology can be found in this papers: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

1.  Dueling DQN

    The vanilla DQN has the overestimate problem. As the `max` function will accumulate the noise when training. This leads to converging at suboptimal point. Two following architectures are submitted to solve this problem.

    \\[ Q(s, a) = r + \gamma \max\_{a'}[Q(s', a')] \\]

    Double DQN was published two year later DQN. It has two value function, one is used to choose the action with max Q value, another one is used to calculate the Q value of this action.

    \\[ a^{max}(S'\_j, w) = \arg\max\_{a'}Q(\phi(S'\_j),a,w) \\]

    \\[ Q(s,a) = r + \gamma Q'(\phi(S'\_j),a^{max}(S'\_j, w),w') \\]

    Dueling DQN is another solution. It has two estimator, one estimates the score of current state, another estimates the action score.

    \\[ Q(S,A,w,\alpha, \beta) = V(S,w,\alpha) + A(S,A,w,\beta) \\]

    In order to distinguish the score of the actions, the return the Q-value will minus the mean action score:

    `x=val+adv-adv.mean(1,keepdim=True)`

    {{< figure src="/images/ddqn_duel_dqn.png" width="400" >}}

    In this project, I use dueling DQN.

2.  Image processing

    I grayscale the image, then remove the background color.

3.  Stack frames

    I use the last 4 frame as the input. This should help the agent to know the change of environment.

4.  Extra FC before last layer

    I add a FC between the image features and the FC for calculate Q-Value.

5.  Frame Skipping

    Frame-skipping means agent sees and selects actions on every k frame instead of every frame, the last action is repeated on skipped frames. This method will accelerate the training procedure. I have tried different frame skipping values. When `frame=1`, I got max reward, about 100. When `frame=2`, the max reward drop to 50. When k=4, the agent failed to play the game, and max reward stays at 0. Although frame skipping is not working with flappy bird, someone has prove that it does work in many cases. More details can be found in this [post](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/).

6.  Prioritized Experience Replay

    This idea was published [here](https://arxiv.org/abs/1511.05952). It's a very simple idea: replay high TD error experience more frequently. My code implementation is not efficient. But in cartpole game, this technology help the agent converge faster. Here is the result on cartpole. The formoer one is uniform replay, the later is prioritized replay.

    <img src="/images/ddqn_cartpole_normal.png" alt="ddqn_cartpole_normal.png" width="400" />
    <img src="/images/ddqn_cartpole_prioritized.png" alt="ddqn_cartpole_prioritized.png" width="400" />

7.  Colab and Kaggle Kernel

    My MacBook doesn't support CUDA, so I use these two website to train the model. Here are the comparison of them. During training, Kaggle seems more stable, Colab usually disconnected after 1h.

    |                      | Colab         | Kaggle Kernel   |
    |----------------------|---------------|-----------------|
    | GPU                  | Tesla T4(16G) | Tesla P100(16G) |
    | RAM                  | 13G           | 13G             |
    | Max training time    | 12h           | 9h              |
    | Export trained model | Google Drive  | -               |

---

The lesson I learnt from this project is patience. It takes a long time(maybe hundreds of thousand steps) to see whether this model works, and there are so many parameters can effect the final performance. It takes me about 3 weeks to build the final model. So if you want to build your own model, be patient and good luck. Here are two articles talking about the debugging and hyperparameter tuning in DQN:

-   [DQN debugging using Open AI gym Cartpole](https://adgefficiency.com/dqn-debugging/)
-   [DDQN hyperparameter tuning using Open AI gym Cartpole](https://adgefficiency.com/dqn-tuning/)

Here are something may help with this task.

-   [Visdom](https://github.com/facebookresearch/visdom)

    It's a visualization tool made by FaceBook. It's more convenient to use it rather than generate graph manually by matplotlib. Besides `reward` and `mean_q`, these variable are also useful when debugging: total frames, TD-error, loss and action\_distribution. [tensorboardX](https://github.com/lanpa/tensorboardX) is a alternative for those who has experience in TensorFlow.

-   Advanced image pre-processing

    In this project, I just grayscalize the image. A more advance technology such as binarize should help agent to filter unimportant detail of game output.
    ![](/images/ddqn_binary_preprocessing.png)

-   Other Improvements

    [Rainbow](https://arxiv.org/abs/1710.02298) introduce many other extensions to enhance DQN, some of them have been discussed in this post.

    {{< figure src="/images/ddqn_rainbow.png" width="400" >}}

I'm still trying to find better solution. I've uploaded code to this [repo](https://github.com/bebound/flappy%5Fbird%5Fdqn/).

Ref:

1.  [PyTorch REINFORCEMENT LEARNING (DQN) TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement%5Fq%5Flearning.html)
2.  [强化学习](https://www.cnblogs.com/pinard/category/1254674.html) (A series of Chinese post about reinforcement learning)
3.  [Deep Reinforcement Learning for Flappy Bird](http://cs229.stanford.edu/proj2015/362%5Freport.pdf)
4.  [Flappy-Bird-Double-DQN-Pytorch](https://github.com/ttaoREtw/Flappy-Bird-Double-DQN-Pytorch)
5.  [DeepRL-Tutorials](https://github.com/qfettes/DeepRL-Tutorials)
6.  [Speeding up DQN on PyTorch: how to solve Pong in 30 minutes](https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)

<!--listend-->

-   Update 19-04-26:

    Colab's GPU has upgrade to Tesla T4 from K80, now it becomes my best bet.

-   Update 19-05-07

    TensorBoard is now natively supported in PyTorch after version 1.1

-   Update 19-07-26

    If you run out of RAM in Colab, it will show up an option to double the RAM.
