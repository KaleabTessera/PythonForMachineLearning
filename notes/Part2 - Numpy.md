# Part 2 - Numpy
Adapted from [GitHub - jakevdp/PythonDataScienceHandbook: Python Data Science Handbook: full text in Jupyter Notebooks](https://github.com/jakevdp/PythonDataScienceHandbook)
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
![array_vs_list.png](https://raw.githubusercontent.com/KaleabTessera/PythonDataScienceHandbook/be23269c7eb119e093a6d5ce91e464f5e686d9ab/notebooks/figures/array_vs_list.png)
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
### 2.3 Array Indexing
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

### 2.4 Array Slicing 
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
### 2.5 Reshaping Arrays
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
Newaxis adds dimension where it is placed. e.g. 
```
x.shape -> (3,1) 
y = x[np.newaxis,:]
y.shape -> (1,3,1) 
```
