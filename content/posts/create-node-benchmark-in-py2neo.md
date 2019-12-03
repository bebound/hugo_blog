+++
title = "Create Node Benchmark in Py2neo"
author = ["KK"]
date = 2018-11-05T15:55:00+08:00
lastmod = 2019-11-29T00:29:06+08:00
tags = ["Python"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Recently, I'm working on a neo4j project. I use `Py2neo` to interact with graph db. Alghough `Py2neo` is a very pythonic and easy to use, its performance is really poor. Sometimes I have to manually write cypher statement by myself if I can't bear with the slow excution. Here is a small script which I use to compare the performance of 4 diffrent ways to insert nodes.

```python
import time

from graph_db import graph

from py2neo.data import Node, Subgraph


def delete_label(label):
    graph.run('MATCH (n:{}) DETACH DELETE n'.format(label))


def delete_all():
    print('delete all')
    graph.run('match (n) detach delete n')


def count_label(label):
    return len(graph.nodes.match(label))


def bench_create1():
    print('Using py2neo one by one')
    delete_label('test')
    start = time.time()
    tx = graph.begin()
    for i in range(100000):
        n = Node('test', id=i)
        tx.create(n)
    tx.commit()
    print(time.time() - start)
    print(count_label('test'))
    delete_label('test')


def bench_create2():
    print('Using cypher one by one')
    delete_label('test')
    start = time.time()
    tx = graph.begin()
    for i in range(100000):
        tx.run('create (n:test {id: $id})', id=i)
        if i and i % 1000 == 0:
            tx.commit()
            tx = graph.begin()
    tx.commit()
    print(time.time() - start)
    print(count_label('test'))
    delete_label('test')


def bench_create3():
    print('Using Subgraph')
    delete_label('test')
    start = time.time()
    tx = graph.begin()
    nodes = []
    for i in range(100000):
        nodes.append(Node('test', id=i))
    s = Subgraph(nodes=nodes)
    tx.create(s)
    tx.commit()
    print(time.time() - start)
    print(count_label('test'))
    delete_label('test')


def bench_create4():
    print('Using unwind')
    delete_label('test')
    start = time.time()
    tx = graph.begin()
    ids = list(range(100000))
    tx.run('unwind $ids as id create (n:test {id: id})', ids=ids)
    tx.commit()
    print(time.time() - start)
    print(count_label('test'))
    delete_label('test')


def bench_create():
    create_tests = [bench_create1, bench_create2, bench_create3, bench_create4]

    print('testing create')
    for i in create_tests:
        i()


if __name__ == '__main__':
    bench_create()
```

Apparently, using cypher with `unwind` keyword is the fastest way to batch insert nodes.

```text
testing create
Using py2neo one by one
96.09799289703369
100000
Using cypher one by one
9.493892192840576
100000
Using Subgraph
7.638832092285156
100000
Using unwind
2.511630058288574
100000
```

The above result is based on `http` protocol. A very interesting result is that, `bolt` protocol will decrease the time of the first method, but double the time of sencond method. That's wired, maybe `py2neo` has some special opitimization when doing batch insert on `bolt` protocol? But I have no idea why insert one by one with cypher is 2x slower. Here is the result of `bolt` protocol.

```text
testing create
Using py2neo one by one
51.73185706138611
100000
Using cypher one by one
22.051995992660522
100000
Using Subgraph
8.81674599647522
100000
Using unwind
2.8623900413513184
100000
```
