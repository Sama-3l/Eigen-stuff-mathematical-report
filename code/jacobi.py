import numpy as np
import time

import matplotlib.pyplot as plt

class Jacobi:

    def jacobi_algorithm(self, A, max_iterations, tolerance):
        n = A.shape[0]
        eigenvectors = np.eye(n)
        
        # Iterate until convergence or maximum number of iterations
        for k in range(max_iterations):
            
            max_off_diag = 0
            p, q = 0, 0
            for i in range(n):
                for j in range(i+1, n):
                    if np.abs(A[i,j]) > max_off_diag:
                        max_off_diag = np.abs(A[i,j])
                        p, q = i, j
            
            if max_off_diag < tolerance:
                break
            
            a_ip = A[p,p]
            a_iq = A[p,q]
            a_qq = A[q,q]
            phi = 0.5 * np.arctan2(2*a_iq, a_qq - a_ip)
            c = np.cos(phi)
            s = np.sin(phi)
            J = np.eye(n)
            J[p,p] = c
            J[q,q] = c
            J[p,q] = -s
            J[q,p] = s
            
            A = J.T.dot(A).dot(J)
            eigenvectors = eigenvectors.dot(J)
        
        eigenvalues = np.diag(A)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        return eigenvalues, eigenvectors
    
    def jacobi_error(self, A, max_iterations, tolerance):
        n = A.shape[0]
        eigenvectors = np.eye(n)
        maxEigValues = []
        iterations = []
        
        for k in range(max_iterations):
            max_off_diag = 0
            p, q = 0, 0
            for i in range(n):
                for j in range(i+1, n):
                    if np.abs(A[i,j]) > max_off_diag:
                        max_off_diag = np.abs(A[i,j])
                        p, q = i, j
            if max_off_diag < tolerance:
                break
            a_ip = A[p,p]
            a_iq = A[p,q]
            a_qq = A[q,q]
            phi = 0.5 * np.arctan2(2*a_iq, a_qq - a_ip)
            c = np.cos(phi)
            s = np.sin(phi)
            J = np.eye(n)
            J[p,p] = c
            J[q,q] = c
            J[p,q] = -s
            J[q,p] = s
            A = J.T.dot(A).dot(J)
            eigenvectors = eigenvectors.dot(J)
            eigenvalues = np.diag(A)
            maxEigValues.append(max(eigenvalues))
            iterations.append(k)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        return maxEigValues, iterations
    
    def get_time_taken(self, A, max_iterations, tolerance):
        start = time.time()
        eigenvalues, eigenvectors = self.jacobi_algorithm(A, max_iterations, tolerance)
        end = time.time()
        return eigenvalues, eigenvectors, round(end - start, 3)