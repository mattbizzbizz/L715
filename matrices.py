#def matrix_zeros(row, col):
#    M = []
#    for i in range(row):
#        M.append([])
#        for j in range(cols):
#            M[i].append(0)
#    return M

def matrix_zeros(rows, cols):
    M = [
            [0 for i in range(cols)]
                    for j in range(rows)]
    return M

#def copy_matrix(M_in):
#    M_out = matrix_zeros(len(M_in), len(M_in[0]))
#    for i in range(len(M_in)):
#        for j in range(len(M_in[0])):
#            M_out[i][j] = M_in[i][j]
#    return M_out

def copy_matrix(M):
    copy = [rows for rows in M]
    return copy

def print_matrix(M):
    print('[', end = '')
    for i, row in enumerate(M):
        if type(row) == list:
            if i == 0:
                print('[', end = '')
            else:
                print(' [', end = '')
            for j, num in enumerate(row):
                print(num, end = '')
                if j != (len(row) - 1):
                    print(', ', end = '')
            if i != (len(M) - 1):
                print('],')
            else:
                print(']', end = '')
        else:
            print(row, end = '')
            if i != (len(M) - 1):
                print(', ', end = '')
    print(']')

def matrix_add(A, B):
    C = matrix_zeros(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
    return C

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

B = [[6, 4], 
     [4, 8], 
     [3, 5]]
print_matrix(B)
print_matrix(matrix_flatten(B))
