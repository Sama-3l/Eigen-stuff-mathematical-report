import numpy as np
import time
import math
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from power import PowerMethod
from jacobi import Jacobi

class EigenResults:
    
    def numpy_results(A):
        numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
        print(max(numpy_eigenvalues))
        print(numpy_eigenvectors[np.array(numpy_eigenvalues).argmax()])

    def jacobi(A):
        eigenvalues, eigenvectors = Jacobi().jacobi_algorithm(A, 5000, 1e-16)
        max_value = max(eigenvalues)
        max_vector = eigenvectors[np.array(eigenvalues).argmax()]
        print(max_value, '\n', max_vector)

    def Power(A):
        eigenvalue, eigenvector = PowerMethod().power_method(A, 5000, 1e-16)
        print(eigenvalue)
        print(eigenvector)

    def NN(A):
        eigenvalue, eigenvector, number_of_steps = NeuralNetwork().run_neural_net(A)
        print(eigenvalue)
        print(eigenvector)

class TimeResults:

    def numpy_time(A):
        start = time.time()
        numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
        end = time.time()
        print(round(end- start, 3))
        return round(end- start, 3)

    def jacobi_time(A):
        eigenvalues, eigenvectors, time = Jacobi().get_time_taken(A, 5000, 1e-16)
        max_value = max(eigenvalues)
        max_vector = eigenvectors[np.array(eigenvalues).argmax()]
        print(time)
        return time

    def power_time(A):
        eigenvalue, eigenvector, time = PowerMethod().get_time_taken(A, 5000, 1e-16)
        # print(time)
        return time

    def nn_time(A):
        eigenvalue, eigenvector, number_of_steps, time = NeuralNetwork().get_time_taken_nn(A)
        print(time)
        return time
    
class ErrorResults:


    def jacobi_error_analysis(axs1, A, x, y):
        numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
        eigenvalues, iterations = Jacobi().jacobi_error(A, 5000, 1e-16)
        numpy_eig = []
        for i in range(len(iterations)):
            numpy_eig.append(max(numpy_eigenvalues))

        axs1[x][y].set_title(f'dim {A.shape[0]}')
        axs1[x][y].plot(iterations, numpy_eig)
        axs1[x][y].plot(iterations, eigenvalues)

    def power_error_analysis(axs2, A, x, y):
        numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
        eigenvalues, iterations = PowerMethod().get_power_error(A, 5000, 1e-16)
        numpy_eig = []
        for i in range(len(iterations)):
            numpy_eig.append(max(numpy_eigenvalues))

        axs2[x][y].set_title(f'dim {A.shape[0]}')
        axs2[x][y].plot(iterations, numpy_eig)
        axs2[x][y].plot(iterations, eigenvalues)
    
    def nn_error_analysis(axs3, A, x, y):
        numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
        eigenvalues, iterations = NeuralNetwork().run_nn_for_error(A)
        numpy_eig = []
        for i in range(len(iterations)):
            numpy_eig.append(max(numpy_eigenvalues))

        axs3[x][y].set_title(f'dim {A.shape[0]}')
        axs3[x][y].plot(iterations, numpy_eig)
        axs3[x][y].plot(iterations, eigenvalues)

def sep():
    for i in range(50):
        print('-', end='')
    print('\n')

def random_symmetric (matrix_size):
        
    A = np.random.rand (matrix_size, matrix_size)
    A = (np.transpose(A) + A) / 2
    return A

def get_power_time_vs_dim(number_of_dimensions):
    times = []
    dim = []
    for i in range(2, number_of_dimensions):
        times.append(TimeResults.power_time(random_symmetric(i)))
        dim.append(i)
    print(times)
    plt.plot(dim, times)
    plt.show()  

def get_eigen_results(number_of_dimensions):
    eigen = EigenResults()
    for i in range(2, number_of_dimensions+1):
        A = random_symmetric(i)
        print("Numpy values\n")
        eigen.numpy_results(A)
        print("\n\nJacobi Results\n")
        eigen.jacobi(A)
        print("\n\nPower Results\n")
        eigen.Power(A)
        print("\n\nNeural net Results\n")
        eigen.NN(A)
        sep()
    
def get_time_results(number_of_dimensions):
    time = TimeResults
    times = [[], [], [], []]
    dim = []
    for i in range(2, number_of_dimensions):
        A = random_symmetric(i)
        dim.append(i)
        print("Numpt time\n")
        times[0].append(time.numpy_time(A))
        print("\n\nJacobi Results\n")
        times[1].append(time.jacobi_time(A))
        print("\n\nPower Results\n")
        times[2].append(time.power_time(A))
        print("\n\nNeural net Results\n")
        times[3].append(time.nn_time(A))
        sep()

    for i in range(4):
        plt.plot(dim, times[i], label=i)

    plt.legend()
    plt.show()

def get_error_results(number_of_dimensions):

    def get_subplot_coordinates(dimension):
        x = math.floor(dimension / 16) % 4
        y = math.floor(dimension / 4) % 4
        return x, y

    errorAnalysis = ErrorResults
    fig1, axs1 = plt.subplots(4, 4, constrained_layout=True, num=1)
    fig1.suptitle('Jacobi Method')
    fig2, axs2 = plt.subplots(nrows=4, ncols=4, constrained_layout=True)
    fig2.suptitle('Power Method')
    fig3, axs3 = plt.subplots(nrows=4, ncols=4, constrained_layout=True)
    fig3.suptitle('Neural Network')
    for i in range(2, number_of_dimensions+1, 4):
        A = random_symmetric(i)
        x, y = get_subplot_coordinates(i)
        errorAnalysis.jacobi_error_analysis(axs1, A, x, y)
        errorAnalysis.power_error_analysis(axs2, A, x, y)
        errorAnalysis.nn_error_analysis(axs3, A, x, y)

    plt.show()