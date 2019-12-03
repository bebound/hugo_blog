+++
title = "Jaeger Code Structure"
author = ["KK"]
date = 2019-09-22T17:07:00+08:00
lastmod = 2019-11-29T00:29:11+08:00
tags = ["Jaeger"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

Here is the main logic for jaeger agent and jaeger collector. (Based on [jaeger](https://github.com/jaegertracing/jaeger) 1.13.1)

{{< figure src="/images/jaeger.svg" width="600" >}}


## Jaeger Agent {#jaeger-agent}

Collect UDP packet from 6831 port, convert it to `model.Span`, send to collector by gRPC


## Jaeger Collector {#jaeger-collector}

Process gRPC or process packet from Zipkin(port 9411).


## Jaeger Query {#jaeger-query}

Listen gRPC and HTTP request from 16686.
