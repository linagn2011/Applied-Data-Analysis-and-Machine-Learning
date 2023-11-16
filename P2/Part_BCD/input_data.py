import autograd.numpy as np



def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4



def batch_generator(X, y, batch_size, shuffle=True):
    """
    Generate batches from input data X and target data y.

    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n, m), where n is the number of samples and m is the number of features.
    - y (numpy.ndarray): Target data vector of shape (n,).
    - batch_size (int): Size of each batch.
    - shuffle (bool): Whether to shuffle the input data before generating batches. Default is True.

    Yields:
    - X_batch (numpy.ndarray): Batch of input data of shape (batch_size, m).
    - y_batch (numpy.ndarray): Batch of target data of shape (batch_size,).
    """
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    num_batches = n // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_indices = indices[start_idx:end_idx]

        X_batch = X[batch_indices, :]
        y_batch = y[batch_indices]

        yield X_batch, y_batch

    # If there's a remainder, yield the last batch
    if n % batch_size != 0:
        start_idx = num_batches * batch_size
        batch_indices = indices[start_idx:]

        X_batch = X[batch_indices, :]
        y_batch = y[batch_indices]

        yield X_batch, y_batch