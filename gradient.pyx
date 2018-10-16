import numpy as np
import math
cimport cython
cimport numpy as np
from libc.math cimport sqrt

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.ndarray[np.uint8_t, ndim=2] get_x_gradients(char[:,:] img):
    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]
    cdef int x
    cdef int y
    
    cdef char[:,:] x_gradients = np.zeros([height,width], dtype=np.uint8)

    for y in range(height):
        x_gradients[y][0] = img[y][1] - img[y][0]

        for x in range(1, width - 1):
            x_gradients[y][x] = (img[y][x+1] - img[y][x-1]) / 2

        x_gradients[y][width - 1] = img[y][width - 1] - img[y][width - 2]

    return np.asarray(x_gradients)

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double[:,:] get_magnitude_matrix(char[:,:] x_gradients, char[:,:] y_gradients):
    cdef Py_ssize_t height = x_gradients.shape[0]
    cdef Py_ssize_t width = x_gradients.shape[1]
    cdef double gx
    cdef double gy
    cdef double[:,:] magnitudes = np.zeros([height,width], dtype=np.double)

    for y in range(height):
        for x in range(width):
            gx = x_gradients[y][x]
            gy = y_gradients[y][x]
            magnitudes[y][x] = sqrt((gx * gx) + (gy * gy))

    return magnitudes

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef normalize_gradients(char[:,:] x_gradients, char[:,:] y_gradients, double[:,:] normalized_x, double[:,:] normalized_y):
    cdef Py_ssize_t height = x_gradients.shape[0]
    cdef Py_ssize_t width = x_gradients.shape[1]
    cdef double[:,:] magnitudes = get_magnitude_matrix(x_gradients, y_gradients)

    cdef double stdMagnGrad = np.std(magnitudes)
    cdef double meanMagnGrad = np.mean(magnitudes)

    cdef int x
    cdef int y
    cdef double gx
    cdef double gy
    cdef double magnitude

    cdef double threshold = 0.5 * stdMagnGrad + meanMagnGrad

    # normalize gradients by removing all where the magnitude is below the threshold
    for y in range(height):
        for x in range(width):
            gx = x_gradients[y][x]
            gy = y_gradients[y][x]
            magnitude = magnitudes[y][x]
            
            if magnitude < threshold:
                normalized_x[y][x] = 0
                normalized_y[y][x] = 0
            else:
                normalized_x[y][x] = gx / magnitude
                normalized_y[y][x] = gy / magnitude
