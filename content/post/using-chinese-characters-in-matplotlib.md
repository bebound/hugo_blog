+++
title = "Using Chinese Characters in Matplotlib"
author = ["kk"]
date = 2018-10-04T15:53:00+08:00
lastmod = 2019-02-14T23:37:45+08:00
tags = ["Python", "Matplotlib"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

After searching from Google, here is easiest solution. This should also works on other languages:

```python
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.font_manager as fm
f = "/System/Library/Fonts/PingFang.ttc"
prop = fm.FontProperties(fname=f)

plt.title("你好",fontproperties=prop)
plt.show()
```

Output:
![](/images/matplot_chinese.png)
