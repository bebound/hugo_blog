+++
title = "Program Crash Caused by CPU Instruction"
author = ["KK"]
date = 2020-05-17T17:36:00+08:00
lastmod = 2020-05-17T17:37:40+08:00
tags = ["Python"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

It's inevitable to dealing with bugs in coding career. The main part of coding are implementing new features, fixing bugs and improving performance. For me, there are two kinds of bugs that is difficult to tackle: those are hard to reproduce, and those occur in code not wrote by you.

Recently, I met a bug which has both features mentioned before. I write a Spark program to analyse the log and cluster them. Last week I update the code, use Facebook's [faiss](https://github.com/facebookresearch/faiss) library to accelerate the process of find similar vector. After I push the new code to spark, the program crashed. I found this log on Spark driver:

```nil
java.io.EOFException
ERROR PythonRUnner: Python worker exited unexpectedly (crashed).
```

Because the Python Worker is created by Spark JVM, I can't get the internal state of Python Worker. By inserting log into Code, I get the rough position of crash code. But the code looks good.

I have tested the code on my develop environment. My develop machine is Using Spark 2.4. but the Spark platform is using Spark 3.0. I guess maybe there is some compatible problem on Spark 3.0. So I use the same docker images as Spark platform to run the code. The code works as expected without crash. That's wired, the docker has isolate the environment, how could same docker image produce different output?

I search the error from google, some said it's because spark is running out of memory. This doesn't seem correct, this update shouldn't increase the RAM usage. I still gave it a try and no luck.

Alright, this update add faiss to the code, maybe faiss lead to the crash, as Python doesn't raise any other. If the crash is caused by the C code in faiss, this makes sense. First, I write a code with spark and faiss, the program crashed. Then I wrote a code only contains faiss, it still crashed. So I can confirm that the crash is cause by faiss and Spark is innocent. Even stranger, when running on Spark platform, sometimes the script crashes, sometimes not.

But why faiss only crash on the Spark Platform? I ask the colleague to know the detail of the failed job and know that the docker's exit code is 132. `132` means illegal instruction. I search illegal instruction on faiss's GitHub issue. I found this issue: [Illegal instruction (core dumped)](https://github.com/facebookresearch/faiss/pulls).

By compare the host server's CPU instruction. The crashed ones lack of `avx2` instruction. `avx2` is added after the Intel Fourth generation core (Haswell). The develop server is using sixth generation CPU, and some platform server is too to support this instruction. By adding a parameter to enforce the script scheduling on new server, the crash disappears.

PS: Running faiss code `index.add(xx)` will not trigger the crash, but calling `faiss.seach(xx)` does. When I trying to locate the code which cause the crash, the `faiss` package was imported correctly and the index is built normally. This mislead me to believe that faiss code is working.
