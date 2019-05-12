+++
title = "Deploy Nikola Org Mode on Travis"
author = ["KK"]
date = 2018-11-03T14:22:00+08:00
lastmod = 2019-05-12T21:20:02+08:00
tags = ["Python", "Nikola", "Org Mode"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Recently, I enjoy using `Spacemacs`, so I decided to switch to org file from Markdown for writing blog. After several attempts, I managed to let Travis convert org file to HTML. Here are the steps.


## Install Org Mode plugin {#install-org-mode-plugin}

First you need to install Org Mode plugin on your computer following the official guide: [Nikola orgmode plugin](https://plugins.getnikola.com/v8/orgmode/).


## Edit `conf.el` {#edit-conf-dot-el}

`Org Mode` will convert to HTML to display on Nikola. Org Mode plugin will call Emacs to do this job. When I run `nikola build`, it shows this message: `Please install htmlize from https://github.com/hniksic/emacs-htmlize`. I'm using `Spacemacs`, the `htmlize` package is already downloaded if the `org` layer is enabled. I just need to add htmlize folder to load-path. So here is the code:

```elisp
(setq dir "~/.emacs.d/elpa/27.0/develop/")
(if(file-directory-p dir)
    (let ((default-directory dir))
      (normal-top-level-add-subdirs-to-load-path)))
(require 'htmlize)
```

This package is also needed on Travis, the similar approach is required.


## Modify `.travis.yml` {#modify-dot-travis-dot-yml}

Travis is using ubuntu 14.04, and the default Emacs version is 24, and the Org Mode version is below 8.0, which not match the requirements. The easiest solution is to update Emacs to 25. So in the `before_install` section, add these code:

```yaml
- sudo add-apt-repository ppa:kelleyk/emacs -y
- sudo apt-get update
```

In the `install` section, add these code:

```yaml
- sudo apt-get remove emacs
- sudo apt autoremove
- sudo apt-get install emacs25
```

The default emacs doesn't contains `htmlize` package. So add `git clone https://github.com/hniksic/emacs-htmlize ~/emacs-htmlize` into `before_install` section.

Finally, modify `conf.el` for Travis Emacs, add GitHub repo to `load-path`: `(add-to-list 'load-path "~/emacs-htmlize/")`

Voila, the org file should show up.

The full `.travis.yml` is below:

```yaml
language: python
cache: apt
sudo: false
addons:
  apt:
    packages:
    - language-pack-en-base
branches:
  only:
  - src
python:
- 3.6
before_install:
- sudo add-apt-repository ppa:kelleyk/emacs -y
- sudo apt-get update
- openssl aes-256-cbc -K $encrypted_a5c638e4bedc_key -iv $encrypted_a5c638e4bedc_iv
  -in travis.enc -out travis -d
- git config --global user.name 'bebound'
- git config --global user.email 'bebound@gmail.com'
- git config --global push.default 'simple'
- pip install --upgrade pip wheel
- echo -e 'Host github.com\n    StrictHostKeyChecking no' >> ~/.ssh/config
- eval "$(ssh-agent -s)"
- chmod 600 travis
- ssh-add travis
- git remote rm origin
- git remote add origin git@github.com:bebound/bebound.github.io
- git fetch origin master
- git branch master FETCH_HEAD
- git clone https://github.com/hniksic/emacs-htmlize ~/emacs-htmlize
install:
- pip install 'Nikola[extras]'==7.8.15
- sudo apt-get remove emacs
- sudo apt autoremove
- sudo apt-get install emacs25
script:
- nikola build && nikola github_deploy -m 'Nikola auto deploy [ci skip]'
notifications:
  email:
    on_success: change
    on_failure: always
```

And here is the `conf.el`:

```elisp
(setq dir "~/.emacs.d/elpa/27.0/develop/")
(if(file-directory-p dir)
    (let ((default-directory dir))
      (normal-top-level-add-subdirs-to-load-path)))
(add-to-list 'load-path "~/emacs-htmlize/")
(require 'htmlize)
```
