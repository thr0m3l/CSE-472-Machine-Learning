import numpy as np

#Take m,n as input
m = int(input("Enter the number of rows: "))
n = int(input("Enter the number of columns: "))

#Generate a random matrix of size m x n
mat = np.random.randint(10000, size=(m,n))

#Perform SVD
U, D, V = np.linalg.svd(mat)
print(U)
print(D)
print(V)

D_plus = np.zeros((m,n)).T
D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))

A_plus = np.dot(V.T, np.dot(D_plus, U.T))
print(A_plus)

#Check if the calculated pseudo-inverse is the same as the one calculated by numpy
print("Are the matrices the same? ", np.allclose(A_plus, np.linalg.pinv(mat)))

