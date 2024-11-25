#define TILE_DIM 16
#define COARSE_FACTOR 2

__global__ void mm_tiled_coarse_kernel(float* A, float* B, float* C, unsigned int M,
    unsigned int N, unsigned int K) {
    // Define shared memory tiles
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM * COARSE_FACTOR];

    // Compute the row and starting column indices
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    // Initialize sum array
    float sum[COARSE_FACTOR];
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        sum[c] = 0.0f;
    }

    // Compute the number of tiles over the shared dimension N
    unsigned int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    // Loop over tiles
    for (unsigned int tile = 0; tile < numTiles; ++tile) {
        // Load A_s
        unsigned int A_col = tile * TILE_DIM + threadIdx.x;
        if (row < M && A_col < N) {
            A_s[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        } else {
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B_s for all coarse factors
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int col = colStart + c * blockDim.x;
            unsigned int B_row = tile * TILE_DIM + threadIdx.y;
            if (B_row < N && col < K) {
                B_s[threadIdx.y][threadIdx.x + c * blockDim.x] = B[B_row * K + col];
            } else {
                B_s[threadIdx.y][threadIdx.x + c * blockDim.x] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        if (row < M) {
            for (unsigned int i = 0; i < TILE_DIM; ++i) {
                float a_element = A_s[threadIdx.y][i];
                for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
                    float b_element = B_s[i][threadIdx.x + c * blockDim.x];
                    sum[c] += a_element * b_element;
                }
            }
        }

        __syncthreads();
    }

    // Write results back to C
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int col = colStart + c * blockDim.x;
        if (row < M && col < K) {
            C[row * K + col] = sum[c];
        }
    }
}

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Derivative of ReLU
__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Sigmoid Activation Function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivative of Sigmoid
__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Kernel to add biases and apply ReLU activation
__global__ void add_bias_relu_kernel(float* Z, float* b, float* A_out, unsigned int M, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = M * N;
    if (idx < total_elements) {
        unsigned int col = idx % N;
        A_out[idx] = relu(Z[idx] + b[col]);
    }
}

// Kernel to add biases and apply Sigmoid activation
__global__ void add_bias_sigmoid_kernel(float* Z, float* b, float* A_out, unsigned int M, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = M * N;
    if (idx < total_elements) {
        unsigned int col = idx % N;
        A_out[idx] = sigmoid(Z[idx] + b[col]);
    }
}

// Kernel to compute derivative of activation function during backpropagation
__global__ void activation_derivative_kernel(float* A, float* dA, unsigned int M, unsigned int N, bool is_relu) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = M * N;
    if (idx < total_elements) {
        if (is_relu) {
            dA[idx] *= relu_derivative(A[idx]);
        } else {
            dA[idx] *= sigmoid_derivative(A[idx]);
        }
    }
}

__global__ void threshold_kernel(float* outputs, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outputs[idx] = outputs[idx] >= 0.5f ? 1.0f : 0.0f;
    }
}

// Forward pass function
void forward_pass(
    float* d_X,    // Input data
    float* d_W1,   // Weights between input and hidden layer
    float* d_b1,   // Biases for hidden layer
    float* d_W2,   // Weights between hidden and output layer
    float* d_b2,   // Biases for output layer
    float* d_Z1,   // Linear outputs of hidden layer
    float* d_A1,   // Activations of hidden layer
    float* d_Z2,   // Linear outputs of output layer
    float* d_A2,   // Output predictions
    unsigned int batch_size, // Number of samples
    unsigned int N1, // Input size
    unsigned int N2, // Hidden layer size
    unsigned int N3 // Output size
) {
    // Dimensions
    unsigned int M = batch_size; // Number of samples 

    // Matrix multiplication parameters
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((N2 + (TILE_DIM * COARSE_FACTOR) - 1) / (TILE_DIM * COARSE_FACTOR),
        (M + TILE_DIM - 1) / TILE_DIM);

    // Compute Z1 = X * W1
    mm_tiled_coarse_kernel<<<dimGrid, dimBlock>>>(d_X, d_W1, d_Z1, M, N1, N2);

    // Add biases and apply ReLU activation: A1 = ReLU(Z1 + b1)
    unsigned int total_elements = M * N2;
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    add_bias_relu_kernel<<<num_blocks, threads_per_block>>>(d_Z1, d_b1,d_A1, M, N2);

    // Compute Z2 = A1 * W2
    dim3 dimGrid2((N3 + (TILE_DIM * COARSE_FACTOR) - 1) / (TILE_DIM * COARSE_FACTOR), (M + TILE_DIM - 1) / TILE_DIM);
    mm_tiled_coarse_kernel<<<dimGrid2, dimBlock>>>(d_A1, d_W2, d_Z2, M, N2, N3);

    // Add biases and apply Sigmoid activation: A2 = Sigmoid(Z2 + b2)
    total_elements = M * N3;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    add_bias_sigmoid_kernel<<<num_blocks, threads_per_block>>>(d_Z2, d_b2, d_A2, M, N3);  
}

// Kernel to compute dZ2 = A2 - Y
__global__ void compute_dZ2_kernel(float* A2, float* Y, float* dZ2, unsigned int M) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        dZ2[idx] = A2[idx] - Y[idx];
    }
}

// Kernel to compute sum over rows (for biases gradients)
__global__ void sum_over_rows_kernel(float* dZ, float* db, unsigned int M, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < M; ++i) {
            sum += dZ[i * N + idx];
        }
        db[idx] = sum;
    }
}

__global__ void transpose_matrix_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (x < width && y < height) {
        unsigned int pos_in = y * width + x;     // Position in the input matrix
        unsigned int pos_out = x * height + y;   // Position in the output (transposed) matrix
        output[pos_out] = input[pos_in];
    }
}

__global__ void update_weights_kernel(
    float* W, float* dW, float learning_rate, unsigned int size, unsigned int batch_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= learning_rate * dW[idx] / batch_size;
    }
}

__global__ void update_biases_kernel(
    float* b, float* db, float learning_rate, unsigned int size, unsigned int batch_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] -= learning_rate * db[idx] / batch_size;
    }
}


// Backward pass function
void backward_pass(
    float* d_X,    // Input data
    float* d_Y,    // True labels
    float* d_W1,   // Weights between input and hidden layer
    float* d_W2,   // Weights between hidden and output layer
    float* d_b1,   // Biases for hidden layer
    float* d_b2,   // Biases for output layer
    float* d_Z1,   // Linear outputs of hidden layer
    float* d_A1,   // Activations of hidden layer
    float* d_Z2,   // Linear outputs of output layer
    float* d_A2,   // Output predictions
    float* d_dW1,  // Gradients for W1
    float* d_db1,  // Gradients for b1
    float* d_dW2,  // Gradients for W2
    float* d_db2,  // Gradients for b2
    float* d_dZ1,  // Gradient at Z1
    float* d_dZ2,  // Gradient at Z2
    unsigned int batch_size,
    unsigned int N1, // Input size
    unsigned int N2, // Hidden layer size
    unsigned int N3 // Output size

) {
    // Dimensions
    unsigned int M = batch_size; // Number of samples

    // Compute dZ2 = A2 - Y
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = (M + threads_per_block - 1) / threads_per_block;
    compute_dZ2_kernel<<<num_blocks, threads_per_block>>>(d_A2, d_Y, d_dZ2, M);

    // Compute dW2 = A1^T * dZ2
    dim3 dimBlockTrans(TILE_DIM, TILE_DIM);
    dim3 dimGridTrans((N3 + TILE_DIM - 1) / TILE_DIM, (N2 + TILE_DIM - 1) / TILE_DIM);

    // Transpose A1 and use mm_tiled_coarse_kernel
    // We need to allocate a transposed version of A1
    float* d_A1_T;
    cudaMalloc((void**)&d_A1_T, M * N2 * sizeof(float));

    transpose_matrix_kernel<<<dimGridTrans, dimBlockTrans>>>(d_A1, d_A1_T, M, N2);

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid(
        (N3 + (TILE_DIM * COARSE_FACTOR) - 1) / (TILE_DIM * COARSE_FACTOR),
        (N2 + TILE_DIM - 1) / TILE_DIM
    );

    // Compute dW2 = A1^T * dZ2
    mm_tiled_coarse_kernel<<<dimGrid, dimBlock>>>(d_A1_T, d_dZ2, d_dW2, N2, M, N3);

    // Compute db2 = sum over dZ2
    num_blocks = (N3 + threads_per_block - 1) / threads_per_block;
    sum_over_rows_kernel<<<num_blocks, threads_per_block>>>(d_dZ2, d_db2, M, N3);

    // Compute dA1 = dZ2 * W2^T
    // Since W2 is of shape (N2, N3), W2^T is (N3, N2)
    float* d_W2_T;
    cudaMalloc((void**)&d_W2_T, N2 * N3 * sizeof(float));

    dim3 dimGridTransW2((N2 + TILE_DIM - 1) / TILE_DIM, (N3 + TILE_DIM - 1) / TILE_DIM);

    // Transpose W2
    transpose_matrix_kernel<<<dimGridTransW2, dimBlockTrans>>>(d_W2, d_W2_T, N2, N3);

    // Compute dA1 = dZ2 * W2^T
    dim3 dimGrid_dA1((N2 + (TILE_DIM * COARSE_FACTOR) - 1) / (TILE_DIM * COARSE_FACTOR), (M + TILE_DIM - 1) / TILE_DIM);
    mm_tiled_coarse_kernel<<<dimGrid_dA1, dimBlock>>>(d_dZ2, d_W2_T, d_dZ1, M, N3, N2);

    // Apply derivative of ReLU: dZ1 = dA1 * ReLU'(Z1)
    unsigned int total_elements = M * N2;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    activation_derivative_kernel<<<num_blocks, threads_per_block>>>(d_Z1, d_dZ1, M, N2, true);

    float* d_X_T;
    cudaMalloc((void**)&d_X_T, N1 * M * sizeof(float));  // Transposed X (size N1 x M)

    // Transpose X
    dim3 dimGridTransX((M + TILE_DIM - 1) / TILE_DIM, (N1 + TILE_DIM - 1) /TILE_DIM );
    transpose_matrix_kernel<<<dimGridTransX, dimBlockTrans>>>(d_X, d_X_T, M, N1);


    // Compute dW1 = X^T * dZ1
    dim3 dimGrid_dW1((N2 + (TILE_DIM * COARSE_FACTOR) - 1) / (TILE_DIM * COARSE_FACTOR), (N1 + TILE_DIM - 1) / TILE_DIM);
    mm_tiled_coarse_kernel<<<dimGrid_dW1, dimBlock>>>(d_X_T, d_dZ1, d_dW1, N1, M, N2);

    // Compute db1 = sum over dZ1
    num_blocks = (N2 + threads_per_block - 1) / threads_per_block;
    sum_over_rows_kernel<<<num_blocks, threads_per_block>>>(d_dZ1, d_db1, M, N2);

    // Free temporary memory
    cudaFree(d_A1_T);
    cudaFree(d_W2_T);
    cudaFree(d_X_T);
}