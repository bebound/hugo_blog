+++
title = "Some Useful Shell Tools"
author = ["KK"]
date = 2017-05-07T15:34:00+08:00
lastmod = 2019-05-12T21:20:29+08:00
tags = ["Shell"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Here are some shell tools I use, which can boost your productivity.


## [Prezto](https://github.com/sorin-ionescu/prezto) {#prezto}

A zsh configuration framework. Provides auto completion, prompt theme and lots of modules to work with other useful tools. I extremely love the `agnoster` theme.

{{< figure src="/images/shell_agnoster.png" width="400" >}}


## [Fasd](https://github.com/clvv/fasd) {#fasd}

Help you to navigate between folders and launch application.

Here are the official usage example:
\`\`\`
  v def conf       =>     vim /some/awkward/path/to/type/default.conf
  j abc            =>     cd /hell/of/a/awkward/path/to/get/to/abcdef
  m movie          =>     mplayer /whatever/whatever/whatever/awesome\_movie.mp4
  o eng paper      =>     xdg-open /you/dont/remember/where/english\_paper.pdf
  vim \`f rc lo\`    =>     vim /etc/rc.local
  vim \`f rc conf\`  =>     vim /etc/rc.conf
\`\`\`


## [pt](https://github.com/monochromegane/the%5Fplatinum%5Fsearcher) {#pt}

A fast code search tool similar to `ack`.


## [fzf](https://github.com/junegunn/fzf) {#fzf}

A great fuzzy finder, it can also integrate with vim by [fzf.vim](https://github.com/junegunn/fzf.vim)

{{< figure src="/images/shell_fzf.gif" width="400" >}}


## [thefuck](https://github.com/nvbn/thefuck) {#thefuck}

Magnificent app which corrects your previous console command.

{{< figure src="/images/shell_thefuck.gif" width="400" >}}
