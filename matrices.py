# Create a matrix of zeros with dimensions rows x cols
def matrix_zeros(rows, cols):
    return [[0] * cols for i in range(rows)]

# Create a copy of the matrix M
def copy_matrix(M):
    return [rows[:] for rows in M]

# Print the matrix in such a format:
#   [[6, 4], 
#    [4, 8], 
#    [3, 5]]
def print_matrix(mat):

    # Check if mat is a 1D matrix
    if any(isinstance(elem, int) for elem in mat):
        print(mat)

    else:
        print('[', end='')

        # Iterate through rows of mat
        for i, row in enumerate(mat):

            # Add a space if this is NOT the first row
            if i != 0:
                print(' ', end='')

            print('[', end='') # Add left bracket for start of row

            # Iterate through elements of row
            for j, col in enumerate(row):

                print(col, end='') # print element

                # Add a comma and space if this is NOT the last element of the row
                #   else add right bracket
                if j != (len(row) - 1):
                    print(', ', end='')
                else:
                    print(']', end = '')

            if i != (len(mat) - 1):
                print(',')

        print(']')

def matrix_add(A, B):
    if (matrix_shape(A) != matrix_shape(B)):
       raise ValueError('Cannot add arrays with different shapes.') 
    C = []
#    C = matrix_zeros(len(A), len(A[0]))
#    for i in range(len(A)):
#        for j in range(len(A[0])):
#            C[i][j] = A[i][j] + B[i][j]
    return [C.append(row + col)]

def matrix_multiply(A, B):
    C = matrix_zeros(len(A), len(B[0]))
    for k in range(len(B[0])):
        for i in range(len(A)):
            mult = 0
            for j in range(len(A[0])):
                mult += A[i][j] * B[j][k]
            C[i][k] = mult
    return C

def matrix_transpose(A):
    C = matrix_zeros(len(A[0]), len(A))
    for i in range(len(A[0])):
        for j in range(len(A)):
            C[i][j] = A[j][i]
    return C

def matrix_flatten(A):
    C = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            C.append(A[i][j])
    return C

def matrix_concat(A, B):

    if len(A) != len(B):
       raise ValueError('Cannot concatonate arrays with different number of rows.') 

    C = matrix_zeros(len(A), len(A[0]) + len(B[0]))
    for i, row in enumerate(A):
        for j, num in enumerate(row):
            C[i][j] = num
    for i, row in enumerate(A):
        for j, num in enumerate(row):
            C[i][j + len(A[0])] = num
    return C

def matrix_shape(A):
    shape = []
    while isinstance(A, list):
        shape.append(len(a))
        A = A[0]
    return shape

def transpose_matrix(matrix):
    transpose = []
    for i in range(len(matrix[0])): # num col
        transpose.append([matrix[j][i] for j in range(len(matrix))]) # num rows
               
    return transpose
