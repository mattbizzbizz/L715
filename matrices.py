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
    print("[", end = '')
    for row in M:
        print(row, end = '')
    print("]")

#print(matrix_zeros(4, 2))

N = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]]

#print(copy_matrix(N))

print_matrix(N)
