+++
title = "Enable C Extension for gensim on Windows"
author = ["KK"]
date = 2017-06-10T14:43:00+08:00
lastmod = 2019-11-29T00:29:07+08:00
tags = ["Python"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

These days, I’m working on some text classification works, and I use `gensim=’s =doc2vec` function.

When using gensim, it shows this warning message:
\`\`\`
C extension not loaded for Word2Vec, training will be slow.
\`\`\`

I search this on Internet and found that gensim has rewrite some part of the code using \`cython\` rather than \`numpy\` to get better performance. A compiler is required to enable this feature.

I tried to install mingw and add it into the path, but it's not working.

Finally, I tried to install [Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017) and it works.

If this output a none `-1` digit, then it's fine.
\`\`\`python3
from gensim.models import word2vec
print(word2vec.FAST\_VERSION)
\`\`\`
