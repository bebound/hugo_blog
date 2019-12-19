+++
title = "Circular Import in Python"
author = ["KK"]
date = 2019-03-10T10:59:00+08:00
lastmod = 2019-12-18T00:11:20+08:00
tags = ["Python"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Recently, I found a really good example code for Python circular import, and I'd like to record it here.

Here is the code:

{{< highlight python3 "linenos=table, linenostart=1" >}}
# X.py
def X1():
    return "x1"

from Y import Y2

def X2():
    return "x2"
{{< /highlight >}}

{{< highlight python3 "linenos=table, linenostart=1" >}}
# Y.py
def Y1():
    return "y1"

from X import X1

def Y2():
    return "y2"
{{< /highlight >}}

Guess what will happen if you run `python X.py` and `python Y.py`?

Here is the answer, the first one outputs this:

```nil
Traceback (most recent call last):
  File "X.py", line 4, in <module>
    from Y import Y2
  File "/Users/kk/Y.py", line 4, in <module>
    from X import X1
  File "/Users/kk/X.py", line 4, in <module>
    from Y import Y2
ImportError: cannot import name Y2
```

The second one runs normally.

If this is the same as you thought, you already know how python import works. You don't need to read this post.


## Python import machinery {#python-import-machinery}

When Python imports a module for the first time, it create a new module object and set `sys.modules[module_name]=module object` , then executes execute in module object to define its content. If you import that module again, Python will just return the object save in `sys.modules`.

In `X.py` line 5, Python add `Y` into `sys.modules` and start execute code in `Y.py`. In `Y.xy` line5, it pause import Y, add `X` into `sys.modules`, and execute code `X.py`. Back to `X.py` line5, Python find `Y` in `sys.modules` and try to import Y2 in Y. But `Y2` is not yet defined, so the ImportError was raised.


## How to fix {#how-to-fix}

-   Change import order.
-   Wrap function call related to other module into `configure` function, call it manually.
-   Dynamic import(use import within a function).


## Ref: {#ref}

1.  [Python Circular Imports](https://stackabuse.com/python-circular-imports/)
2.  [Python Cirluar Importing](https://stackoverflow.com/questions/22187279/python-circular-importing)
3.  [Circular imports in Python](https://stackoverflow.com/questions/744373/circular-or-cyclic-imports-in-python)
4.  [Effective Python: 59 Specific Ways to Write Better Python](https://www.amazon.com/Effective-Python-Specific-Software-Development/dp/0134034287)
5.  [Python doc: The import system](https://docs.python.org/3/reference/import.html)
