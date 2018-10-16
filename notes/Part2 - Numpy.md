# Part 2 - Numpy
Adapted from [GitHub - jakevdp/PythonDataScienceHandbook: Python Data Science Handbook: full text in Jupyter Notebooks](https://github.com/jakevdp/PythonDataScienceHandbook),[NumPy Tutorial](https://www.tutorialspoint.com/numpy), [NumPy Reference — NumPy v1.13 Manual](https://docs.scipy.org/doc/numpy-1.13.0/reference/) and personal insights. 
## Data Types in Python
Pthon uses dynamic typing, therefore we can create heterogeneous lists.

Example Heterogeneous Lists
```python
L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]
```
Output
```
[bool, str, float, int]
```

### Dynamic Type vs Fixed Type lists
Flexibility vs Efficiency
![array_vs_list.png](https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/be23269c7eb119e093a6d5ce91e464f5e686d9ab/notebooks/figures/array_vs_list.png)
From: [PythonDataScienceHandbook/02.01-Understanding-Data-Types.ipynb at be23269c7eb119e093a6d5ce91e464f5e686d9ab · jakevdp/PythonDataScienceHandbook · GitHub](https://github.com/jakevdp/PythonDataScienceHandbook/blob/be23269c7eb119e093a6d5ce91e464f5e686d9ab/notebooks/02.01-Understanding-Data-Types.ipynb)
*************
## Numpy Arrays
- Efficient storage of array-based data
- Efficient operations on the data

### 2.1 Creating Numpy Arrays
#### From python lists
- From python lists - no explicit type
```python 
np.array([1, 4, 2, 5, 3])
```
- From python lists - explicit type
```python 
np.array([1, 2, 3, 4], dtype=`float32`)
```

#### From scratch
- 1-d
```python
np.zeros(10, dtype=int)
```
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```
- 3 * 5 array
```python
np.ones((3, 5), dtype=float)
```
```
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.]])
```
- Create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1
 ```python
 np.random.normal(0, 1, (3, 3))
 ```
 ```
 array([[ 1.51772646,  0.39614948, -0.10634696],
       [ 0.25671348,  0.00732722,  0.37783601],
       [ 0.68446945,  0.15926039, -0.70744073]])`
 ```
 - Identity Matrix
 ```
 np.eye(3)
 ```
 ```
 array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
 ```

***********
### 2.2 NumPy Array Attributes
```
python 
import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
```
Example Attributes
```python
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")
```
```
x3 ndim:  3
x3 shape: (3, 4, 5)
x3 size:  60
dtype: int64
itemsize: 8 bytes
nbytes: 480 bytes
```
***********
### 2.3.1 Numpy Array Manipulations 
#### 2.3.1 Array Indexing
- 1d array
```python
np.random.seed(0)
x1 = np.random.randint(10, size=6)  
x1
```
```
array([5, 0, 3, 3, 7, 9])
```
- Multi-dimensional arrays - 2d array
```python
np.random.seed(0)
x2 = np.random.randint(10, size=(3, 4)) 
```
```
array([[3, 5, 2, 4],
       [7, 6, 8, 8],
       [1, 6, 7, 7]])
```
```python
x2[2, -1]
```
```
7
```

#### 2.3.2 Array Slicing 
General Formulae
`x[start:stop:step]`
#### 1d
```
x = np.arange(10)
x
```
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```
-Select first 5 elements/ last 5
```python
print("First 5:",x[:5])
print("Last 5 :",x[5:])
```
```
First 5: [0 1 2 3 4]
Last 5 : [5 6 7 8 9]
```
- Sub arrays
```python
x[4:7]  # middle sub-array
```
```
array([4, 5, 6])
```
#### Multidimensional Arrays
- 2d Sub arrays
```python
np.random.seed(0)
x2 = np.random.randint(10, size=(3, 4))
x2
```
```
array([[5, 0, 3, 3],
       [7, 9, 3, 5],
       [2, 4, 7, 6]])
```

```python
x2[:2, :3] 
```
```
array([[5, 0, 3],
       [7, 9, 3]])
```
- Printing columns - column 0
```python
print(x[:,0])
```
- Printing rows - row 0
```python
print(x[0,:])
```
#### 2.3.3 Reshaping Arrays
- Convert 1d array into 2d row vector
```python
x = np.array([1, 2, 3])
print("Using reshape",x.reshape((1, 3)))
print("Using newaxis",x[np.newaxis, :])
```
```
Using reshape [[1 2 3]]
Using newaxis [[1 2 3]]
```

- Convert 1d array into or 2d column vector
```python
x = np.array([1, 2, 3])
print("Using reshape \n",x.reshape((3,1)))
print("Using newaxis \n",x[ :, np.newaxis])
```
```
Using reshape 
 [[1]
 [2]
 [3]]
Using newaxis 
 [[1]
 [2]
 [3]]
```
**Intitution** Newaxis adds dimension where it is placed. e.g. 
```
x.shape -> (3,1) 
y = x[np.newaxis,:]
y.shape -> (1,3,1) 
```

#### 2.3.4 Concatenation of arrays
```python
x = np.array([1,2,3])
y = np.array([3, 2, 1])
z = [99,99,99]
np.concatenate([x,y,z])
```
```
array([1, 2, 3, 3, 2, 1,99,99,99])
```
#### Concatenation accross a certain axis
```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
# concatenate along the first axis - axis 0 by default
axis0Concat = np.concatenate([grid, grid])
axis1Concat = np.concatenate([grid, grid],axis=1)
print("Concat on axis 0: \n",axis0Concat)
print("Concat on axis 1: \n",axis1Concat)
```
```
Concat on axis 0: 
 [[1 2 3]
 [4 5 6]
 [1 2 3]
 [4 5 6]]
Concat on axis 1: 
 [[1 2 3 1 2 3]
 [4 5 6 4 5 6]]
```
**Intution** You are concating on chosen axis, which is represented by dimention of the object.
```
grid.shape -> (2,3)
axis0Concat.shape -> (4,3) - concat on axis 0, 2 + 2 = 4
axis1Concat.shape -> (2,6) - concat on axis 1, 3 + 3 = 6
```

#### Vstack 
`Equivalent to np.concatenate(tup, axis=0) if tup contains arrays that are at least 2-dimensional.`
```python
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])
```
```
array([[1, 2, 3],
       [9, 8, 7],
       [6, 5, 4]])
```
**Note** When vstack or np.concatenate(tup, axis=0), axis 1 needs to be the same for both arrays that are stacked. 
e.g. x.shape  -> (4,**3**), y.shape-> (2,**3**). 
#### Hstack
`Equivalent to np.concatenate(tup, axis=1) if tup contains arrays that are at least 2-dimensional.`
```python
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])
```
```
array([[ 9,  8,  7, 99],
       [ 6,  5,  4, 99]])
```
**Note** When hstack or np.concatenate(tup, axis=1), axis 0 needs to be the same for both arrays that are stacked. 
e.g. x.shape  -> (**3**,3), y.shape-> (**3**,2). 

#### 2.3.5 Splitting Arrays - `numpy.split(ary, indices_or_sections, axis)`
Split at indexs 3 and 5.
```python
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
```
```
[1 2 3] [99 99] [3 2 1]
```
Variants - np.hsplit and np.vsplit also exist.

************
### 2.4 Functions - UFuns in Python 
Functions that operate on arrays, element by element. These functions are **vectorized** and hence are often optimized well.


The following table lists the arithmetic operators implemented in NumPy:

| Operator	    | Equivalent ufunc    | Description                           |
|---------------|---------------------|---------------------------------------|
|``+``          |``np.add``           |Addition (e.g., ``1 + 1 = 2``)         |
|``-``          |``np.subtract``      |Subtraction (e.g., ``3 - 2 = 1``)      |
|``-``          |``np.negative``      |Unary negation (e.g., ``-2``)          |
|``*``          |``np.multiply``      |Multiplication (e.g., ``2 * 3 = 6``)   |
|``/``          |``np.divide``        |Division (e.g., ``3 / 2 = 1.5``)       |
|``//``         |``np.floor_divide``  |Floor division (e.g., ``3 // 2 = 1``)  |
|``**``         |``np.power``         |Exponentiation (e.g., ``2 ** 3 = 8``)  |
|``%``          |``np.mod``           |Modulus/remainder (e.g., ``9 % 4 = 1``)|

From [PythonDataScienceHandbook/02.03-Computation-on-arrays-ufuncs.ipynb at master · jakevdp/PythonDataScienceHandbook · GitHub](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.03-Computation-on-arrays-ufuncs.ipynb)

#### 2.4.1 Array Arithmetic
```python
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)
```
```
x     = [0 1 2 3]
x + 5 = [5 6 7 8]
x - 5 = [-5 -4 -3 -2]
x * 2 = [0 2 4 6]
x / 2 = [ 0.   0.5  1.   1.5]
x // 2 = [0 0 1 1]
-x     =  [ 0 -1 -2 -3]
x ** 2 =  [0 1 4 9]
x % 2  =  [0 1 0 1]
```
These operations are wrappers for specific functions, e.g.  the `+` operator is a wrapper for the `add` function.

#### 2.4.2 Absolute Value - `np.absolute(x)` or  `np.abs(x)`
```
x = np.array([-2, -1, 0, 1, 2])
abs(x)
```
```
array([2, 1, 0, 1, 2])
```

#### 2.4.3 Exponents and logarithms
Exponents
```python
x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))
```
```
x     = [1, 2, 3]
e^x   = [ 2.71828183  7.3890561  20.08553692]
2^x   = [2. 4. 8.]
3^x   = [ 3  9 27]
```
Logarithms
```python
x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))
```
```
x        = [1, 2, 4, 10]
ln(x)    = [0.         0.69314718 1.38629436 2.30258509]
log2(x)  = [0.         1.         2.         3.32192809]
log10(x) = [0.         0.30103    0.60205999 1.        ]
```
#### 2.4.4 Aggregations

```python
L = np.random.random(100)
print("Sum:",np.sum(L))
print("Min:",np.max(L))
print("Max:",np.min(L))
```
Example of possible output
```
Sum: 51.97939595817096
Min: 0.9885847781325164
Max: 0.00045139488468282085
```

Other Aggregations
The following table provides a list of useful aggregation functions available in NumPy:

|Function Name      |   NaN-safe Version  | Description                                   |
|-------------------|---------------------|-----------------------------------------------|
| ``np.sum``        | ``np.nansum``       | Compute sum of elements                       |
| ``np.prod``       | ``np.nanprod``      | Compute product of elements                   |
| ``np.mean``       | ``np.nanmean``      | Compute mean of elements                      |
| ``np.std``        | ``np.nanstd``       | Compute standard deviation                    |
| ``np.var``        | ``np.nanvar``       | Compute variance                              |
| ``np.min``        | ``np.nanmin``       | Find minimum value                            |
| ``np.max``        | ``np.nanmax``       | Find maximum value                            |
| ``np.argmin``     | ``np.nanargmin``    | Find index of minimum value                   |
| ``np.argmax``     | ``np.nanargmax``    | Find index of maximum value                   |
| ``np.median``     | ``np.nanmedian``    | Compute median of elements                    |
| ``np.percentile`` | ``np.nanpercentile``| Compute rank-based statistics of elements     |
| ``np.any``        | N/A                 | Evaluate whether any elements are true        |
| ``np.all``        | N/A                 | Evaluate whether all elements are true        |

From: [PythonDataScienceHandbook/02.04-Computation-on-arrays-aggregates.ipynb at be23269c7eb119e093a6d5ce91e464f5e686d9ab · jakevdp/PythonDataScienceHandbook · GitHub](https://github.com/jakevdp/PythonDataScienceHandbook/blob/be23269c7eb119e093a6d5ce91e464f5e686d9ab/notebooks/02.04-Computation-on-arrays-aggregates.ipynb)

#### 2.4.5 Comparisons -`==` / `<` / `>`
```python
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
x
x < 6
```
Possible output
```
array([[5, 0, 3, 3],
       [7, 9, 3, 5],
       [2, 4, 7, 6]])
array([[ True,  True,  True,  True],
       [False, False,  True,  True],
       [ True,  True, False, False]], dtype=bool)
```

#### 2.4.6 Sorting Arrays - `np.sort` and `np.argsort`
Numpy sort - `np.sort` or `x.sort()``
```python
x = np.array([2, 1, 4, 3, 5])
np.sort(x)
```
```
array([1, 2, 3, 4, 5])
```
Argsort - returns indexs of sorted elements i.e. indexs where element in array should be, if the array was sorted.
```python
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
```
```
[1 0 3 2 4]
```
Hence, element at index 0 (number 2), will be sorted(ascending) if it was at index 1.

##### Sort by column or row
```python
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)
```
```
[[6 3 7 4 6 9]
 [2 6 7 4 3 7]
 [7 2 5 4 1 7]
 [5 1 4 0 9 5]]
```
```python
# sort each column of X
print("Sort each column:\n",np.sort(X, axis=0))
# sort each row of X
print("Sort each row:\n",np.sort(X, axis=1))
```
```
Sort each column:
 [[2 1 4 0 1 5]
 [5 2 5 4 3 7]
 [6 3 7 4 6 7]
 [7 6 7 4 9 9]]
Sort each row:
 [[3 4 6 6 7 9]
 [2 3 4 6 7 7]
 [1 2 4 5 7 7]
 [0 1 4 5 5 9]]

```
