import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import time

class NeuralNetwork:

    tf.disable_v2_behavior()

    def f_x(self, x, A, matrix_size):
        xTxA = (tf.tensordot(tf.transpose(x), x, axes=1)*A)
        # (1- xTAx)*I
        xTAxI = (1- tf.tensordot(tf.transpose(x), tf.tensordot(A, x, axes=1), axes=1))*np.eye(matrix_size)
        # (xTx*A - (1- xTAx)*I)*x
        f = tf.tensordot((xTxA + xTAxI), x, axes=1)
        return f
    
    def printout(self, i, verbose, loss, max_iterations, eigenvalue):
        if verbose:
            if i % 100 == 0:
                l = loss.eval()
                print("Step:", i, "/",max_iterations, "loss: ", l, "Eigenvalue:" , eigenvalue)

    def get_error_at_every_step(self, A, eigenvalue):
        numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
        return np.min(abs(numpy_eigenvalues - eigenvalue))

    def NN_Eigenvalue(self, matrix_size, A, max_iterations, nn_structure, eigen_guess, eigen_lr, delta_threshold):
        
        # Defining the 6x6 identity matrix
        I = np.identity(matrix_size)

        # Create a vector of random numbers and then normalize it
        # This is the beginning trial solution eigenvector
        x0 = np.random.rand(matrix_size)
        x0 = x0/np.sqrt(np.sum(x0*x0))
        # Reshape the trial eigenvector into the format for Tensorflow
        x0 = np.reshape(x0, (1, matrix_size))
        # Convert the above matrix and vector into tensors that can be used by
        # Tensorflow
        I_tf = tf.convert_to_tensor(I)
        x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)
        # Set up the neural network with the specified architecture
        error = []
        steps = []

        with tf.variable_scope('dnn'):
            num_hidden_layers = np.size(nn_structure)
            # x0 is the input to the neural network
            previous_layer = x0_tf
            # Hidden layers
            for l in range(num_hidden_layers):
                current_layer = tf.keras.layers.Dense(units=nn_structure[l], activation=tf.nn.relu)(previous_layer)
                previous_layer = current_layer
                # Output layer
                dnn_output = tf.keras.layers.Dense(units=matrix_size)(previous_layer)

        # Define the loss function
        with tf.name_scope('loss'):
            # trial eigenvector is the output of the neural network
            x_trial = tf.transpose(dnn_output) 
            # f(x)
            f_trial = tf.transpose(self.f_x(x_trial, A, matrix_size))
            # eigenvalue calculated using the trial eigenvector using the 
            # Rayleigh quotient formula
            eigenvalue_trial = tf.transpose(x_trial)@A@x_trial/(tf.transpose(x_trial)@x_trial)

            x_trial = tf.transpose(x_trial)
            # Define the loss function
            loss = tf.losses.mean_squared_error(f_trial, x_trial) + \
            eigen_lr*tf.losses.mean_squared_error([[eigen_guess]], eigenvalue_trial)

        # Define the training algorithm and loss function
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)
            ## Execute the Tensorflow session
            with tf.Session() as sess: 
                # Initialize the Tensorflow variables
                init = tf.global_variables_initializer()
                init.run()
                # Define for calculating the change between successively calculated
                # eigenvalues
                old_eig = 0
                for i in range(max_iterations):
                    sess.run(training_op)
                    # Compute the eigenvalue using the Rayleigh quotient
                    eigenvalue = (x_trial.eval() @ (A @ x_trial.eval().T)
                    /(x_trial.eval() @ x_trial.eval().T))[0,0]
                    eigenvector = x_trial.eval()
                    # Calculate the change between the currently calculated eigenvalue
                    # and the previous one
                    delta = np.abs(eigenvalue-old_eig)
                    old_eig = eigenvalue

                    # Print useful information every 100 steps
                    error.append(self.get_error_at_every_step(A, eigenvalue))
                    steps.append(i)

                    # Kill the loop if the change in eigenvalues is less than the 
                    # given threshold

                    if delta < delta_threshold: 
                        break
        # Return the converged eigenvalue and eigenvector
        
        return eigenvalue, eigenvector, i
    
    def Error_NN_Eigenvalue(self, matrix_size, A, max_iterations, nn_structure, eigen_guess, eigen_lr, delta_threshold):
        
        # Defining the 6x6 identity matrix
        I = np.identity(matrix_size)

        # Create a vector of random numbers and then normalize it
        # This is the beginning trial solution eigenvector
        x0 = np.random.rand(matrix_size)
        x0 = x0/np.sqrt(np.sum(x0*x0))
        # Reshape the trial eigenvector into the format for Tensorflow
        x0 = np.reshape(x0, (1, matrix_size))
        # Convert the above matrix and vector into tensors that can be used by
        # Tensorflow
        I_tf = tf.convert_to_tensor(I)
        x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)
        # Set up the neural network with the specified architecture
        iterative_eigenvalues = []
        iterations = []

        with tf.variable_scope('dnn'):
            num_hidden_layers = np.size(nn_structure)
            # x0 is the input to the neural network
            previous_layer = x0_tf
            # Hidden layers
            for l in range(num_hidden_layers):
                current_layer = tf.keras.layers.Dense(units=nn_structure[l], activation=tf.nn.relu)(previous_layer)
                previous_layer = current_layer
                # Output layer
                dnn_output = tf.keras.layers.Dense(units=matrix_size)(previous_layer)

        # Define the loss function
        with tf.name_scope('loss'):
            # trial eigenvector is the output of the neural network
            x_trial = tf.transpose(dnn_output) 
            # f(x)
            f_trial = tf.transpose(self.f_x(x_trial, A, matrix_size))
            # eigenvalue calculated using the trial eigenvector using the 
            # Rayleigh quotient formula
            eigenvalue_trial = tf.transpose(x_trial)@A@x_trial/(tf.transpose(x_trial)@x_trial)

            x_trial = tf.transpose(x_trial)
            # Define the loss function
            loss = tf.losses.mean_squared_error(f_trial, x_trial) + \
            eigen_lr*tf.losses.mean_squared_error([[eigen_guess]], eigenvalue_trial)

        # Define the training algorithm and loss function
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)
            ## Execute the Tensorflow session
            with tf.Session() as sess: 
                # Initialize the Tensorflow variables
                init = tf.global_variables_initializer()
                init.run()
                # Define for calculating the change between successively calculated
                # eigenvalues
                old_eig = 0
                for i in range(max_iterations):
                    sess.run(training_op)
                    # Compute the eigenvalue using the Rayleigh quotient
                    eigenvalue = (x_trial.eval() @ (A @ x_trial.eval().T)
                    /(x_trial.eval() @ x_trial.eval().T))[0,0]
                    eigenvector = x_trial.eval()
                    # Calculate the change between the currently calculated eigenvalue
                    # and the previous one
                    delta = np.abs(eigenvalue-old_eig)
                    old_eig = eigenvalue

                    iterative_eigenvalues.append(eigenvalue)
                    iterations.append(i)

                    if delta < delta_threshold: 
                        break
        
        return iterative_eigenvalues, iterations

    def random_symmetric (self, matrix_size):

        # Create a matrix filled with random numbers
        A = np.random.rand (matrix_size, matrix_size)
        # Ensure that matrix A is symmetric
        A = (np.transpose(A) + A) / 2
        return A

    def pairing_model_4p4h (self, g, delta):

        A = np.array(
            [[2*delta-g, -0.5*g, -0.5*g, -0.5*g, -0.5*g, 0.],
            [ -0.5*g, 4*delta-g, -0.5*g, -0.5*g, 0., -0.5*g ], 
            [ -0.5*g, -0.5*g, 6*delta-g, 0., -0.5*g, -0.5*g ], 
            [ -0.5*g, -0.5*g, 0., 6*delta-g, -0.5*g, -0.5*g ], 
            [ -0.5*g, 0., -0.5*g, -0.5*g, 8*delta-g, -0.5*g ], 
            [ 0., -0.5*g, -0.5*g, -0.5*g, -0.5*g, 10*delta-g ]])
        return A
    
    def run_nn_for_error(self, A):
        matrix_size = A.shape[0] # Size of the matrix
        max_iterations = 5000 # Maximum number of iterations
        nn_structure = [100,100] # Number of hidden neurons in each layer
        eigen_guess = 70 # Guess for the eigenvalue (see the header of NN_Eigenvalue)
        eigen_lr = 0.01 # Eigenvalue learnign rate (see the header of NN_Eigenvalue)
        delta_threshold = 1e-16 # Kill condition
        verbose = True # True to display state of neural network evrey 100th iteration
        number_of_steps = 0

        tf.reset_default_graph()
        # Calcualte the estimate of the eigenvalue and the eigenvector
        eigenvalues, iterations  = self.Error_NN_Eigenvalue(matrix_size, A, max_iterations, nn_structure, eigen_guess, eigen_lr, delta_threshold)

        return eigenvalues, iterations


    def run_neural_net(self, A):

        # Defining variables
        matrix_size = A.shape[0] # Size of the matrix
        max_iterations = 5000 # Maximum number of iterations
        nn_structure = [100,100] # Number of hidden neurons in each layer
        eigen_guess = 70 # Guess for the eigenvalue (see the header of NN_Eigenvalue)
        eigen_lr = 0.01 # Eigenvalue learnign rate (see the header of NN_Eigenvalue)
        delta_threshold = 1e-16 # Kill condition
        verbose = True # True to display state of neural network evrey 100th iteration
        number_of_steps = 0

        # Reset the Tensorflow graph, causes an error if this is not here
        # Since the above cells are not re-ran every time this one is, they are not 
        # reset. This line is needed to reset the Tensorflow computational graph to
        # avoid variable redefinition errors. 
        tf.reset_default_graph()
        # Calcualte the estimate of the eigenvalue and the eigenvector
        eigenvalue, eigenvector, number_of_steps  = self.NN_Eigenvalue(matrix_size, A, max_iterations, nn_structure, eigen_guess, eigen_lr, delta_threshold)

        return eigenvalue, eigenvector, number_of_steps
    
    def get_time_taken_nn(self, A):
        start = time.time()
        eigenvalue, eigenvector, number_of_steps = self.run_neural_net(A)
        end = time.time()
        return eigenvalue, eigenvector, number_of_steps, round(end - start, 3)