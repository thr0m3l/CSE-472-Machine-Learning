import numpy as np

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

n = int(input("Enter the size of the matrix: "))

#Generate a random invertible matrix    
M = np.random.randint(10000, size=(n,n))

while not is_invertible(M):
    M = np.random.randint(10000, size=(n,n))
    if (is_invertible(M)):
        break
M = np.dot(M, M.T); #Make the matrix symmetric


w, v = np.linalg.eig(M)
print("Eigenvalues: ", w)
print("Eigenvectors: ", v)

#Reconstruct the matrix from the eigenvalues and eigenvectors
w = np.diag(w)
vinv = np.linalg.inv(v)
M_reconstructed = np.dot(v, np.dot(w, vinv))

print("Original matrix: ", M)
print("Reconstructed matrix: ", M_reconstructed)

#Check if the reconstructed matrix is the same as the original matrix
print("Are the matrices the same? ", np.allclose(M, M_reconstructed))