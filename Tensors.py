import numpy as np

BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024

class Tensor:
    def __init__(self, data):
        "Create a New Tensor"
        self.data = np.array(data, dtype= np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        "String representation of tensor which is helpful for debugging"
        return f"Tensor(data = {self.data}, shape = {self.shape})"

    def __str__(self):
        "Human readable Tensor representation"
        return f"Tensor({self.data})"

    def numpy(self):
        "Return the underlying Numpy array"
        return self.data

    def memory_footprints(self):
        "Calculate the size of Tensor"
        return self.data.nbytes

    def __add__(self, other):
        "Element-wise addition with broadcasting support"
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        "Element-wise subtraction with broadcasting support"
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        "Element-wise multiplication with broadcasting support"
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        "Element-wise division of tensors"
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def _validate_matmul_shapes(self, other):
        pass

    def matmul(self, other):
        pass

    def __matmul__(self, other):
        pass

    def __getitem__(self, key):
        pass

    def reshape(self, *shape):
        pass

    def transpose(self, dim0 = None, dim1 = None):
        pass

    def sum(self, axis = None, keepdims = False):
        pass

    def mean(self, axis = None, keepdims = False):
        pass

    def max(self, axis = None, keepdims = False):
        pass

