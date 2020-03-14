+++
title = "C3 Linearization and Python MRO(Method Resolution Order)"
author = ["KK"]
date = 2020-03-14T17:37:00+08:00
lastmod = 2020-03-14T18:08:21+08:00
tags = ["Python", "MRO"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Python supports multiple inheritance, its class can be derived from more than one base classes. If the specified attribute or methods was not found in current class, how to decide the search sequence from superclasses? In simple scenario, we know left-to right, bottom to up. But when the inheritance hierarchy become complicated, it's not easy to answer by intuition.

For instance, what's search sequence of class M?

```python
class X:pass
class Y: pass
class Z:pass
class A(X,Y):pass
class B(Y,Z):pass
class M(B,A,Z):pass
```

{{< figure src="/images/python_mro.png" width="400" >}}

The answer is: `M, B, A, X, Y, Z, object`


## C3 Algorithm {#c3-algorithm}

How did Python generate this sequence? After Python 2.3, it use `C3 Linearization` algorithm.

C3 follows these two equation:

```nil
L[object] = [object]
L[C(B1…BN)] = [C] + merge(L[B1]…L[BN], [B1, … ,BN])
```

`L[C]` is the MRO of class C, it will evaluate to a list.

The key process is **merge**, it get a list and generate a list by this way:

1.  First, check the first list's head element(`L[B1]`) as H.
2.  If H is not in the tail of other list, output it, and remove it from all of the list, then go to step 1. Otherwise, check the next list's head as H, go to step 2. (tail means the rest of the list except the first element)
3.  If **merge**'s list is empty, end algorithm. If list is not empty but not able to find element to output, raise error.

That seems complicated, I'll use the previous example again to explain the calculation of C3.

Let's begin with the easy ones. Firstly, calculate `A`'s MRO:

```nil
L[A(X,Y)]=[A]+merge(L[X],L[Y],[X,Y])
         =[A]+merge([X,obj],[Y,obj],[X,Y])
         # X is not tail of other list, use it as H
         =[A,X]+merge([obj],[Y,obj],[Y])
         # obj is in the tail of[Y.obj], use Y as H
         =[A,X,Y]+merge([obj],[obj]]
         =[A,X,Y,obj]
```

`B`'s MRO `[B,Y,Z,obj]` and `Z`'s MRO `[z,obj]` can also be calculated.

Now we can get `M`'s MRO:

```nil
L[M(B,A,Z)]=[M]+merge(L[B],L[A],L[Z],[B,A,Z])
         =[M]+merge([B,Y,Z,obj],[A,X,Y,obj],[Z,obj],[B,A,Z])
         =[M,B]+merge([Y,Z,obj],[A,X,Y,obj],[Z,obj],[A,Z])
         # Y is in the tail of [A,X,Y,obj], use A as H
         =[M,B,A]+merge([Y,Z,obj],[X,Y,obj],[Z,obj],[Z])
         # Y is in the tail of [X,Y,obj], use X as H
         =[M,B,A,X]+merge([Y,Z,obj],[Y,obj],[Z,obj],[Z])
         =[M,B,A,X,Y]+merge([Z,obj],[obj],[Z,obj],[Z])
         =[M,B,A,X,Y,X]+merge([obj],[obj],[obj])
         =[M,B,A,X,Y,X,obj]
```


## MRO and super() {#mro-and-super}

`super` also use C3 to find the inherited method to execute.

For instance, `C`'s MRO is `C,A,B,C,obj`, so after `enter B`, it will output `enter A` rather than `enter base`.

```python3
class Base:
    def __init__(self):
        print('enter base')
        print('leave base')


class A(Base):
    def __init__(self):
        print('enter A')
        super(A, self).__init__()
        print('leave A')


class B(Base):
    def __init__(self):
        print('enter B')
        super(B, self).__init__()
        print('leave B')


class C(A, B):
    def __init__(self):
        print('enter C')
        super(C, self).__init__()
        print('leave C')

c = C()
```

```nil
enter C
enter A
enter B
enter base
leave base
leave B
leave A
leave C
```

`super` works like this, it will get `inst`'s MRO, find `cls`'s index, return next class in MRO. (In python3, `super(A,self)` can be write as `super()`)

```python3
def super(cls, inst):
    mro = inst.__class__.mro()
    return mro[mro.index(cls) + 1]
```

When running this line `super(C, self).__init__()`, self is `C`'s instance, mro is:

```nil
[<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Base'>, <class 'object'>]
```

So it returns `A`, and A will execute `__init__()`, then calling `super(A, self).__init__()`, end enter `B`'s `__init__()`. (`C`'s instance will pass as `self` in the calling chain.)


## Ref: {#ref}

1.  [The Python 2.3 Method Resolution Order](https://www.python.org/download/releases/2.3/mro/)
2.  [Python Multiple Inheritance](https://www.programiz.com/python-programming/multiple-inheritance)
3.  [python之理解super及MRO列表](https://www.jianshu.com/p/de7d38c84443)
4.  [Python的MRO以及C3线性化算法](https://www.cnblogs.com/miyauchi-renge/p/10922092.html)
5.  [C3 linearization](https://en.wikipedia.org/wiki/C3%5Flinearization)
