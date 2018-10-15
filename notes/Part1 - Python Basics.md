# Part1 - Python Basics
Adapted the parts I found relevant from [GitHub - jakevdp/PythonDataScienceHandbook: Python Data Science Handbook: full text in Jupyter Notebooks](https://github.com/jakevdp/PythonDataScienceHandbook)
## 1. Accessing help / documentation 
## Help
#### Accessing pythons help/documentations - `?`
```python
help(len)
len?
len??
```
#### Adding documentation/ help to functions
```python
  def square(a):
    """Return the square of a."""
    return a ** 2
  
  square?
  Help on function square in module __main__:

  square(a)
    Return the square of a.
```

#### Accessing Source Code - `??`
For functions implemented in Python, you can access source code. If you can't access source code, it is implemented in another language, e.g. C.

```python
square??
Signature: square(a)
Source:   
def square(a):
  """Return the square of a."""
  return a ** 2
File:      /content/<ipython-input-11-1aa21ec0328f>
Type:      function
```

```python
len??
*Returns the same as len?*
```

### Autocompletion - `[].<TAB>`
```python
L = [1,2,3]
L._<TAB>
L.__add__           L.__gt__            L.__reduce__
L.__class__         L.__hash__          L.__reduce_ex__
```

## Wildcard Matching - `*[]?`
For example, we can use this to list every object in the namespace that ends with Warning:
```python
In [10]: *Warning?
BytesWarning                  RuntimeWarning
DeprecationWarning            SyntaxWarning
FutureWarning                 UnicodeWarning
ImportWarning                 UserWarning
PendingDeprecationWarning     Warning
ResourceWarning
```

******
## 2. Timing and Profiling- `timeit / time / prun`
### Timing
#### Timeit
```python
%%timeit #multiline 
%timeit #singleline
```
Runs multiple loops of lines of code and gives general timing info - best and worst runs.
```python
%%timeit
L = []
for n in range(1000):
  L.append(n ** 2)  
1000 loops, best of 3: 373 µs per loop
```
#### Time
Timeit repeats actions to get average time and this could be misleading - e.g. when sorting a list, sorting a sorted list is faster then sorting an unsorted list.
```python
import random
L = [random.random() for i in range(100000)]
print("sorting an unsorted list:")
%time L.sort()
```
```
sorting an unsorted list:
CPU times: user 40.6 ms, sys: 896 µs, total: 41.5 ms
Wall time: 41.5 ms
```

### Profiling
Profiling can be useful to find what is causing a bottleneck in an application execution. 

You can profile each function call or you can profile line by line.

#### Example - Profiling Function calls
```python
def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total
```
```python
%prun sum_of_lists(1000000)
```
Output
```
  Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        5    0.599    0.120    0.599    0.120 <ipython-input-19>:4(<listcomp>)
        5    0.064    0.013    0.064    0.013 {built-in method sum}
        1    0.036    0.036    0.699    0.699 <ipython-input-19>:1(sum_of_lists)
        1    0.014    0.014    0.714    0.714 <string>:1(<module>)
        1    0.000    0.000    0.714    0.714 {built-in method exec}
```
******
## 3.IPython's In and Out Objects
You can access previous inputs and outputs, using the `In` and `Out` objects.

```python
import math

math.sin(2)
0.9092974268256817

math.cos(2)
-0.4161468365471424

print(In)
[``, `import math`, `math.sin(2)`, `math.cos(2)`, `print(In)`]

Out
{2: 0.9092974268256817, 3: -0.4161468365471424}
```
Input 2, resulted in the output 0.90923.

### Supressing out - ```;``` 
Add ```;``` to end of line, no output.

****************
## 4. Errors and Debugging
### Errors
Different exceptions modes, with different details of information :  ```Plain```,```Context```and  ```Verbose```.

Example

```python
def func1(a, b):
    return a / b

def func2(x):
    a = x
    b = x - 1
    return func1(a, b)
  
func2(1)
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-2-b2e110f6fc8f> in <module>()
----> 1 func2(1)

<ipython-input-1-d849e34d61fb> in func2(x)
      5     a = x
      6     b = x - 1
----> 7     return func1(a, b)

<ipython-input-1-d849e34d61fb> in func1(a, b)
      1 def func1(a, b):
----> 2     return a / b
      3 
      4 def func2(x):
      5     a = x

ZeroDivisionError: division by zero
```  
```python
%xmode Plain
Exception reporting mode: Plain
  
func2(1)
Traceback (most recent call last):

  File "<ipython-input-8-7cb498ea7ed1>", line 1, in <module>
    func2(1)

  File "<ipython-input-1-586ccabd0db3>", line 7, in func2
    return func1(a, b)

  File "<ipython-input-1-586ccabd0db3>", line 2, in func1
    return a / b

ZeroDivisionError: division by zero
```

### Debug - `%debug`
After an exception, if you run `%debug`, the debugger will open in interactive mode at the point of exception. You can then run commands and output variables and see what caused the exception.
```python
%debug
```
```
> <ipython-input-1-586ccabd0db3>(2)func1()
      1 def func1(a, b):
----> 2     return a / b
      3 
      4 def func2(x):
      5     a = x
```

