import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import time

class PowerMethod:

    def power_method(self, A, max_iterations, tolerance):
            n = A.shape[0]
            x = np.random.rand(n)
            x /= np.linalg.norm(x)
            eigenvalue = 0
            
            for i in range(max_iterations):
                
                y = A.dot(x)
                
                eigenvalue_new = np.max(np.abs(y))
                x_new = y / eigenvalue_new
                
                if np.linalg.norm(x - x_new) < tolerance:
                    break
                eigenvalue = eigenvalue_new
                x = x_new
            
            eigenvector = x / np.linalg.norm(x)
            return eigenvalue, eigenvector
    
    def get_time_taken(self, A, max_iterations, tolerance):
        start = time.time()
        eigenvalue, eigenvector = self.power_method(A, max_iterations, tolerance)
        end = time.time()
        return eigenvalue, eigenvector, round(end - start, 3)
    
    def get_power_error(self, A, max_iterations, tolerance):
            n = A.shape[0]
            x = np.random.rand(n)
            x /= np.linalg.norm(x)
            eigenvalue = 0
            eigenvalues = []
            iterations = []
            
            for i in range(max_iterations):
                # Multiply A by x
                y = A.dot(x)
                
                eigenvalue_new = np.max(np.abs(y))
                x_new = y / eigenvalue_new
                
                if np.linalg.norm(x - x_new) < tolerance:
                    break
                eigenvalue = eigenvalue_new
                x = x_new
                if i > 10:
                    eigenvalues.append(eigenvalue)
                    iterations.append(i)
            
            eigenvector = x / np.linalg.norm(x)
            return eigenvalues, iterations