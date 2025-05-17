import numpy as np
import math
cimport numpy as np

# DTYPE = np.int
# ctypedef np.int_t DTYPE_t

DTYPE = np.int_
ctypedef np.intp_t DTYPE_t

def find_worst_coverage(np.ndarray[DTYPE_t, ndim=1] arr, double delta):
    assert arr.dtype == DTYPE
    cdef Py_ssize_t length = arr.shape[0]
    cdef Py_ssize_t min_size = int(np.ceil(delta * length))
    cdef float min_coverage = 1.0
    cdef float current_sum = 0.0
    cdef float begin_sum = 0.0
    
    cdef DTYPE_t min_begin = 0
    cdef DTYPE_t min_end = 0
    
    cdef Py_ssize_t index, begin, end
    
    for index in range(min_size-1):
        begin_sum += arr[index]
    
    for begin in range(length-min_size+1):
        current_sum = begin_sum 
        for end in range(begin+min_size, length+1):
            current_sum += arr[end-1]
            if current_sum / (end - begin) < min_coverage:
                min_coverage = current_sum / (end - begin)
                min_begin = begin
                min_end = end - 1
            
        begin_sum -= arr[begin]
        begin_sum += arr[begin + min_size - 1]
        
    return min_begin, min_end

cdef class MaximumDensityStructure():
    # Not available in Python-space:
    
    # Available in Python-space:
    cdef public Py_ssize_t n
    cdef public DTYPE_t i0 #max_start
    cdef public DTYPE_t[:] L #min_index starting from i

    
    cdef DTYPE_t[:] S_lengths
    cdef DTYPE_t[:] S_starts
    cdef DTYPE_t[:] S_values

    cdef DTYPE_t[:] p

    cdef np.float_t[:] cum_array
    cdef np.float_t[:] cum_weight
    cdef DTYPE_t l, u, b
    

    def __init__(self, 
                np.ndarray[np.float_t, ndim=1] array, np.ndarray[np.float_t, ndim=1] weight,
                np.float_t L):
        self.n = array.shape[0]
        self.p = np.zeros(self.n, dtype=DTYPE)
        self.cum_array = np.concatenate(([0], array.cumsum()))
        self.cum_weight = np.concatenate(([0], weight.cumsum()))
        
        self.S_lengths = np.zeros(self.n, dtype=DTYPE)
        self.S_starts = np.zeros(self.n, dtype=DTYPE)
        self.S_values = np.zeros(self.n, dtype=DTYPE)
        self.initialize_min_lengths(L)
        
    cdef mu_avg(self, DTYPE_t begin, DTYPE_t end):
        return (
                    self.cum_array[end+1] - self.cum_array[begin]
                ) / (
                    self.cum_weight[end+1] - self.cum_weight[begin]
                )
    
    cdef initialize_min_lengths(self, np.float_t L):
        cdef DTYPE_t index_i, index_j
        index_i = 0
        index_j = 0
        self.L = np.full(self.n, -1, dtype=DTYPE)

        while index_i < self.n:
            while (
                index_j < self.n
            ) and (self.cum_weight[index_j+1] - self.cum_weight[index_i] < L):
                index_j += 1
            if index_j < self.n:
                self.L[index_i] = index_j
                self.i0 = index_i
            index_i += 1
    
    cdef LMatchInitialize(self, DTYPE_t x, DTYPE_t y):
        cdef Py_ssize_t index
        
        for index in range(y,x,-1):
            self.p[index] = index
            while (self.p[index] < y) and (
                self.mu_avg(index, self.p[index]) <= self.mu_avg(
                self.p[index]+1, self.p[self.p[index]+1])
            ):
                self.p[index] = self.p[self.p[index]+1]

            self.S_lengths[self.p[index]] += 1
    
        self.S_starts = np.concatenate(([0],np.cumsum(self.S_lengths)[:-1]))
        cdef np.ndarray[DTYPE_t, ndim=1] current_indices = np.copy(self.S_starts)
        
        for index in range(x+1, y+1, 1):
            self.S_values[current_indices[self.p[index]]] = index
            current_indices[self.p[index]] += 1  
#         S[k] = S_values[S_starts[k]:( S_starts[k] +  S_lengths[k])]
        self.l = y
        self.u = y
        self.b = y

    cdef extract_and_remove_from_S(self, DTYPE_t u, DTYPE_t l):
        cdef DTYPE_t begin = self.S_starts[u]
        cdef DTYPE_t end = self.S_starts[u] + self.S_lengths[u] - 1
        cdef DTYPE_t current
        
        while begin != end - 1:
            current = math.ceil((begin + end) / 2.0)
            if self.S_values[current] <= l:
                begin = current
            else:
                end = current
        
        if self.S_values[end] <=l:
            begin = end
            
        cdef DTYPE_t b = self.S_values[begin]
        self.S_lengths[u] = begin - self.S_starts[u] + 1
        return b
        
    cdef LMatchFind(self, DTYPE_t i, DTYPE_t x):
        assert(self.L[i] != -1)
        
        while self.l > (1 + max(self.L[i], x)):
            self.l -= 1
            if self.p[self.l] >= self.u:
                self.b = self.l
            
        while self.u >= self.l and (
            self.mu_avg(i, self.b - 1) > self.mu_avg(i, self.p[self.b])
        ):
            self.u = self.b - 1
            if self.u >= self.l:
                self.b = self.extract_and_remove_from_S(self.u, self.l)    
        return self.u
    
def MaximumDensitySegment(np.ndarray[np.float_t, ndim=1] array, np.ndarray[np.float_t, ndim=1] weight,
                np.float_t L):
    max_density_structure = MaximumDensityStructure(
        array, weight, L
    )
    max_density_structure.LMatchInitialize(
        0, max_density_structure.n - 1)
    
    cdef np.ndarray[DTYPE_t, ndim=1] g = np.full(
        max_density_structure.n, -1, dtype=DTYPE)
    cdef DTYPE_t index
    
    
    for index in range(max_density_structure.i0, -1, -1):
        if max_density_structure.L[index] == max_density_structure.n - 1:
            g[index] = max_density_structure.n - 1
        else:
            g[index] = max_density_structure.LMatchFind(index, 0)
            
    cdef DTYPE_t max_index = 0
    cdef np.float_t max_density = - np.inf
    
    for index in range(0, max_density_structure.i0+1):
        if max_density_structure.mu_avg(index, g[index]) > max_density:
            max_index = index
            max_density = max_density_structure.mu_avg(index, g[index])
    
    return max_index, g[max_index]
