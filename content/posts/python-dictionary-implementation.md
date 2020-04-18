+++
title = "Python Dictionary Implementation"
author = ["KK"]
date = 2019-02-17T21:48:00+08:00
lastmod = 2020-04-18T14:35:55+08:00
tags = ["Python"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

## Overview {#overview}

1.  CPython allocation memory to save dictionary, the initial table size is 8, entries are saved as `<hash,key,value>` in each slot(The slot content changed after Python 3.6).
2.  When a new key is added, python use `i = hash(key) & mask` where `mask=table_size-1` to calculate which slot it should be placed. If the slot is occupied, CPython using a probing algorithm to find the empty slot to store new item.
3.  When 2/3 of the table is full, the table will be resized.
4.  When getting item from dictionary, both `hash` and `key` must be equal.


## Resizing {#resizing}

When elements size is below 50000, the table size will increase by a factor of 4 based on used slots. Otherwise, it will increase by a factor of 2. The dictionary size is always \\(2^{n}\\).

| dict size | resize when elements in dict | new table size |
|-----------|------------------------------|----------------|
| 8         | 6                            | 32             |
| 32        | 22                           | 128            |
| 128       | 86                           | 512            |

Removing item from dictionary doesn't lead to shrink table. The value of the item will marks as null but not empty. When looking up element in dictionary, it will keep probing once find this special mark. So deleting element from Python will not decrease the memory using. If you really want to do so, you can create a new dictionary from old one and delete old one.


## Probing {#probing}

CPython used a modified **random probing** algorithm to choose the empty slot. This algorithm can traval all of the slots in a pseudo random order.

The travel order can be calculated by this formula: `j = ((5*j) + 1) mod 2**i`, where `j` is slot index.

For example, if table size is 8, and the calculate slot index is 2, then the traversal order should be:

`2` -> `(5*2+1) mod 8 = 3` -> `(5*3+1) mod 8 = 0` -> `(5*0+1) mod 8 = 1` -> `6` -> `7` -> `4` -> `5` -> `2`

CPython changed this formula by adding `perturb` and `PERTURB_SHIFT` variables, where `perturb` is hash value and `PERTURB_SHIFT` is 5. By adding `PERTURB_SHIFT`, the probe sequence depends on every bit in the hash code, and the collision probability is decreased. And `perturb` will eventually becomes to 0, this ensures that all of the slots will be checked.

```nil
j = (5*j) + 1 + perturb;
perturb >>= PERTURB_SHIFT;
j = j % 2**i
```


## Dictionary improvement after 3.6 {#dictionary-improvement-after-3-dot-6}

CPython 3.6 use a compact representation to save entries, and "The memory usage of the new dict() is between 20% and 25% smaller compared to Python 3.5".


### Compact Hash Table {#compact-hash-table}

As mentioned before, entries saved in the form of `<hash,key,value>`. This will takes 3B on 64 bit machine. And no matter how much item is added into the dictionary, the memory usage is the same(3B\*table\_size).

After 3.6, CPython use two structure to save data. One is **index**, another is the **real data**.

For example, if the table size is 8, and there is an item in slot 1, the **index** looks like this:

`[null, 0, null, null, null, null, null, null]`

And the **real data** is:

```nil
| hash | key  | value |
| xxx1 | yyy1 | zzz1  |
```

0 represents the items index on **real data**. If another item is added in slot 3, the new **index** become this:

`[null, 0, null, 1, null, null, null, null]`

The **real data** become this:

```nil
| hash | key  | value |
| xxx1 | yyy1 | zzz1  |
| xxx2 | yyy2 | zzz2  |
```

This saves memory, especially when table is not full.


## Ref: {#ref}

1.  [How are Python's Built In Dictionaries Implemented](https://stackoverflow.com/questions/327311/how-are-pythons-built-in-dictionaries-implemented)
2.  [cpython source code](https://hg.python.org/cpython/file/52f68c95e025/Objects/dictobject.c#l33)
3.  [Is it possible to give a python dict an initial capacity (and is it useful)](https://stackoverflow.com/questions/3020514/is-it-possible-to-give-a-python-dict-an-initial-capacity-and-is-it-useful/3020810)
4.  [Python dictionary implementation](http://www.laurentluce.com/posts/python-dictionary-implementation/)
