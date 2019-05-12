+++
title = "Preview LaTeX in Org Mode in MacOS"
author = ["KK"]
date = 2019-05-12T20:26:00+08:00
lastmod = 2019-05-12T21:23:04+08:00
tags = ["Emacs", "Org Mode", "LaTeX"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## Using the right Emacs Version {#using-the-right-emacs-version}

I failed to preview LaTeX with `emacs-plus`. If you have installed `d12frosted/emacs-plus`, uninstall it and use `emacs-mac`.

```nil
brew tap railwaycat/emacsmacport
brew install emacs-mac
```

If you like the fancy spacemacs icon, install it with cask: `brew cask install emacs-mac-spacemacs-icon`


## Install Tex {#install-tex}

-   Download and install BasicTeX.pkg [here](http://www.tug.org/mactex/morepackages.html).
-   Add `/Library/TeX/texbin` to PATH.
-   Install `dvisvgm` by `sudo tlmgr update --self && sudo tlmgr install dvisvgm`


## Emacs settings {#emacs-settings}

-   Add TeX related bin to path: `(setenv "PATH" (concat (getenv "PATH") ":/Library/TeX/texbin"))`
-   Tell Org Mode to create svg images: `(setq org-latex-create-formula-image-program 'dvisvgm)`

Now you can see the rendered LaTeX equation by calling `org-preview-latex-fragment`.

If you want to load LaTeX previews automatically at startup, add this at the beginning of org file: `#+STARTUP: latexpreview`.
