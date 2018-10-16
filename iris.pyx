import numpy as np
import math
import cv2
import gradient
cimport cython
cimport numpy as np
from libc.math cimport sqrt
import gradient

cdef double[:,:] scores

@cython.cdivision(True)
cpdef double[:] find_center(np.ndarray[np.uint8_t, ndim=2] gray, int face_width, int scale_to_pixels):
    cdef int fw = face_width

    cdef Py_ssize_t height = gray.shape[0]
    cdef Py_ssize_t width = gray.shape[1]

    cdef double resize_factor = 1
    cdef np.ndarray[np.uint8_t, ndim=2] resized = gray

    cdef int x, y
    cdef double[:] result
    cdef int is_resized = 0

    if width > scale_to_pixels:
        resize_factor = scale_to_pixels / int(width)
        resized = cv2.resize(gray, (0,0), fx=resize_factor, fy=resize_factor)
        height = resized.shape[0]
        width = resized.shape[1]
        is_resized = 1

    cdef char[:,:] x_gradients = gradient.get_x_gradients(resized)
    cdef char[:,:] y_gradients = gradient.get_x_gradients(resized.transpose()).transpose()

    cdef double[:,:] normalized_x = np.zeros([height, width], dtype=np.double)
    cdef double[:,:] normalized_y = np.zeros([height, width], dtype=np.double)
    gradient.normalize_gradients(x_gradients, y_gradients, normalized_x, normalized_y)

    cdef np.ndarray[np.uint8_t, ndim=2] blurred = cv2.GaussianBlur(resized,(5,5),0.005 * fw);
    scores = get_center_scores(blurred, normalized_x, normalized_y)    
    cdef double[:,:] post_processed = post_process(scores)

    result = choose_center(post_processed, 3)

    if is_resized == 1:
        result[0] = result[0] / resize_factor
        result[1] = result[1] / resize_factor
    
    return result


# Calculate scores for all possible centers
@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double[:,:] get_center_scores(unsigned char[:,:] weights, double[:,:] x_gradients, double[:,:] y_gradients):
    global scores
    cdef Py_ssize_t height = x_gradients.shape[0]
    cdef Py_ssize_t width = x_gradients.shape[1]
    cdef int cx
    cdef int cy
    scores = np.zeros([height, width], dtype=np.double)

    cdef int xmax = 0
    cdef int ymax = 0
    cdef double gx
    cdef double gy

    for y in range(height):
        for x in range(width):
            gx = x_gradients[y][x]
            gy = y_gradients[y][x]
            if gx == 0 and gy == 0:
                continue
            test_possible_centers(x, y, weights, gx, gy, height, width, scores)

    return scores

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.ndarray[np.uint8_t, ndim=2] get_heatmap():
    global scores
    cdef Py_ssize_t height = scores.shape[0]
    cdef Py_ssize_t width = scores.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] result = np.zeros([height, width], dtype=np.uint8)
    cdef double max = np.asarray(scores).max()

    for y in range(height):
        for x in range(width):
            result[y][x] = int (scores[y][x] * 255 / max)

    return result


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double[:,:] post_process(double[:,:] scores):
    cdef Py_ssize_t height = scores.shape[0]
    cdef Py_ssize_t width = scores.shape[1]

    cdef double max = np.max(scores)
    cdef double threshold = max * 0.97
    cdef double value
    cdef double[:,:] normalized = np.zeros([height, width], dtype=np.double)
    cdef int x
    cdef int y

    #cdef set flood = set()

    for y in range(height):
        for x in range(width):
            value = scores[y][x]
            if (value > threshold):
                normalized[y][x] = value

    return normalized

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double test_possible_centers(int x, int y, unsigned char[:,:] weights, double gx, double gy, Py_ssize_t height, Py_ssize_t width, double[:,:] scores):
    cdef int cx
    cdef int cy
    cdef double dx
    cdef double dy
    cdef double magnitude
    cdef double dotProduct
    cdef unsigned char weight

    cdef int N = height * width

    for cy in range(height):
        for cx in range(width):
            if x == cx and y == cy:
                continue

            # create a vector d from the possible center to the gradient origin
            dx = x - cx
            dy = y - cy

            # normalize d 
            magnitude = sqrt((dx * dx) + (dy * dy))
            dx = dx / magnitude
            dy = dy / magnitude

            dotProduct = (dx * gx + dy * gy)

            if dotProduct > 0:
                weight = 255 - weights[cy][cx]
                dotProduct = dotProduct * dotProduct * weight / N  
                scores[cy][cx] = scores[cy][cx] + dotProduct


cdef double[:] choose_center(double[:,:] scores, int mask_size):
    cdef Py_ssize_t height = scores.shape[0]
    cdef Py_ssize_t width = scores.shape[1]

    cdef int index = 0
    cdef int x = 0
    cdef int y = 0

    index = np.argmax(scores)
    x = index % width
    y = (index - x) / width

    return score_centroid_around_max(scores, 1, x, y)

# calculate centroid for the scores around the maximum
# mask size is given as number of pixels from the known center in all directions
@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double[:] score_centroid_around_max(double[:,:] scores, int mask_size, int max_x, int max_y):
    cdef Py_ssize_t height = scores.shape[0]
    cdef Py_ssize_t width = scores.shape[1]

    cdef double[:] result = np.zeros([2], dtype = np.double)
    result[0] = max_x * 1.0
    result[1] = max_y * 1.0

    # validate image borders
    if max_x + mask_size >= width or max_x - mask_size < 0:
        return result
    if max_y + mask_size >= height or max_y - mask_size < 0:
        return result

    # calculate volume
    cdef double m00 = 0 # weighted volume
    cdef double n10 = 0 # normalized centroid x
    cdef double n01 = 0 # normalized centroid y

    cdef int i = 0
    cdef int j = 0

    for j in range(max_y - mask_size, max_y + mask_size + 1):
        for i in range(max_x - mask_size, max_x + mask_size + 1):
            m00 = m00 + scores[j][i]
            n10 = n10 + (scores[j][i] * i) 
            n01 = n01 + (scores[j][i] * j) 

    if m00 > 0:
        result[0] = n10 / m00
        result[1] = n01 / m00

    return result