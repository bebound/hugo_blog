+++
title = "Difference between Value and Pointer variable in Defer in Go"
author = ["KK"]
date = 2019-12-19T22:33:00+08:00
lastmod = 2019-12-19T23:19:42+08:00
tags = ["Go", "Defer"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

`defer` is a useful function to do cleanup, as it will execute in LIFO order before the surrounding function returns. If you don't know how it works, sometimes the execution result may confuse you.


## How it Works and Why Value or Pointer Receiver Matters {#how-it-works-and-why-value-or-pointer-receiver-matters}

I found an interesting code on [Stack Overflow](https://stackoverflow.com/questions/28893586/golang-defer-clarification):

```go
type X struct {
    S string
}

func (x X) Close() {
    fmt.Println("Value-Closing", x.S)
}

func (x *X) CloseP() {
    fmt.Println("Pointer-Closing", x.S)
}

func main() {
    x := X{"Value-X First"}
    defer x.Close()
    x = X{"Value-X Second"}
    defer x.Close()

    x2 := X{"Value-X2 First"}
    defer x2.CloseP()
    x2 = X{"Value-X2 Second"}
    defer x2.CloseP()

    xp := &X{"Pointer-X First"}
    defer xp.Close()
    xp = &X{"Pointer-X Second"}
    defer xp.Close()

    xp2 := &X{"Pointer-X2 First"}
    defer xp2.CloseP()
    xp2 = &X{"Pointer-X2 Second"}
    defer xp2.CloseP()
}
```

The output is:

{{< highlight text "linenos=table, linenostart=1" >}}
Pointer-Closing Pointer-X2 Second
Pointer-Closing Pointer-X2 First
Value-Closing Pointer-X Second
Value-Closing Pointer-X First
Pointer-Closing Value-X2 Second
Pointer-Closing Value-X2 Second
Value-Closing Value-X Second
Value-Closing Value-X First
{{< /highlight >}}

Take a look at line 5-6, why `Pointer-Closing Value-X2 Second` was printed twice? According to [Effective Go](https://golang.org/doc/effective%5Fgo.html#defer), "**The arguments to the deferred function (which include the receiver if the function is a method) are evaluated when the defer executes, not when the call executes.**". And the function's parameters will **saved anew** when evaluated.

As `x2` is value and the defer function `CloseP`'s receiver is a pointer, once defer is called, it will create a pointer which point to `x2` as function's caller. In the following defer, it will create a pointer which point to `x2` again. Although `x2.S` change to "Second", `x2`'s address never changes. Finally, when these two defer is called, the same content was printed again.


## How to Exit Program and Run all Defer {#how-to-exit-program-and-run-all-defer}

From [Golang Runtime](https://golang.org/pkg/runtime/#Goexit):

> `runtime.Goexit()` terminates the goroutine that calls it. No other goroutine is affected. Goexit runs all deferred calls before terminating the goroutine. Because Goexit is not a panic, any recover calls in those deferred functions will return nil.
>
> Calling Goexit from the main goroutine terminates that goroutine without func main returning. Since func main has not returned, the program continues execution of other goroutines. If all other goroutines exit, the program crashes.

If you want the program to exit normally, just add `defer os.Exit(0)` at the top of `main` function. Here is the example code:

```go
package main

import (
  "fmt"
  "os"
  "runtime"
  "time"
)

func subGoroutine() {
  defer fmt.Println("exit sub routine")
  for {
    fmt.Println("sub goroutine running")
    time.Sleep(1 * time.Second)
  }
}

func main() {
  defer os.Exit(0)
  defer fmt.Println("calling os.Exit")

  go subGoroutine()

  time.Sleep(2 * time.Second)
  runtime.Goexit()
}
```

Output:

```nil
sub goroutine running
sub goroutine running
sub goroutine running
calling os.Exit

Process finished with exit code 0
```

The defer code in `subGoroutine` will not execute.


## Ref: {#ref}

1.  [面向信仰编程 defer](https://draveness.me/golang/keyword/golang-defer.html)
2.  [Golang defer clarification](https://stackoverflow.com/questions/28893586/golang-defer-clarification)
3.  [How to exit a go program honoring deferred calls?](https://stackoverflow.com/questions/27629380/how-to-exit-a-go-program-honoring-deferred-calls/39755730)
4.  [Effective Go](https://golang.org/doc/effective%5Fgo.html#defer)
5.  [Golang Runtime](https://golang.org/pkg/runtime/#Goexit)
