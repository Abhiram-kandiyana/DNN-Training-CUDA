// kernel.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define TILE_DIM 16
#define COARSE_FACTOR 2

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

// Kernel to add biases and apply activation
__global__ void add_bias_activate_kernel(float* Z, float* b, float* A_out, unsigned int M, unsigned int N, bool is_relu) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = M * N;
    if (idx < total_elements) {
        unsigned int col = idx % N;
        float z = Z[idx] + b[col];
        A_out[idx] = is_relu ? relu(z) : sigmoid(z);
    }
}

// Kernel to compute derivative of activation function during backpropagation
__global__ void activation_derivative_kernel(float* Z, float* dA, unsigned int size, bool is_relu) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        float grad = is_relu ? relu_derivative(z) : sigmoid_derivative(z);
        dA[idx] *= grad;
    }
}

// Kernel to compute element-wise subtraction (A2 - Y)
__global__ void compute_dZ2_kernel(float* A2, float* Y, float* dZ2, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ2[idx] = A2[idx] - Y[idx];
    }
}

// Kernel to update parameters
__global__ void update_parameters_kernel(float* param, float* grad, float learning_rate, unsigned int size, unsigned int batch_size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= (learning_rate / batch_size) * grad[idx];
    }
}

// Forward pass function
void forward_pass(
    cublasHandle_t handle,
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
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute Z1 = W1^T * X
    // Note: cuBLAS uses column-major order
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N2, batch_size, N1,
        &alpha,
        d_W1, N1,
        d_X, N1,
        &beta,
        d_Z1, N2
    );

    // Add biases and apply ReLU activation: A1 = ReLU(Z1 + b1)
    unsigned int total_elements = batch_size * N2;
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    add_bias_activate_kernel<<<num_blocks, threads_per_block>>>(d_Z1, d_b1, d_A1, batch_size, N2, true);

    // Compute Z2 = W2^T * A1
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N3, batch_size, N2,
        &alpha,
        d_W2, N2,
        d_A1, N2,
        &beta,
        d_Z2, N3
    );

    // Add biases and apply Sigmoid activation: A2 = Sigmoid(Z2 + b2)
    total_elements = batch_size * N3;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    add_bias_activate_kernel<<<num_blocks, threads_per_block>>>(d_Z2, d_b2, d_A2, batch_size, N3, false);
}
// Optimized reduction kernel to sum over rows for bias gradients
__global__ void sum_over_rows_kernel(float* dZ, float* db, unsigned int M, unsigned int N) {
    extern __shared__ float shared_data[];

    unsigned int col = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned int i = tid; i < M; i += blockDim.x) {
        sum += dZ[i * N + col];
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Perform tree-based reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        db[col] = shared_data[0];
    }
}

// Function to compute bias gradients using optimized reduction
void compute_bias_gradients(float* d_dZ, float* d_db, unsigned int batch_size, unsigned int N) {
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = N;
    size_t shared_memory_size = threads_per_block * sizeof(float);

    sum_over_rows_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(d_dZ, d_db, batch_size, N);
}

// Backward pass function
void backward_pass(
    cublasHandle_t handle,
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
    float* d_dA1,  // Gradient at A1
    float* d_dZ1,  // Gradient at Z1
    float* d_dZ2,  // Gradient at Z2
    unsigned int batch_size,
    unsigned int N1, // Input size
    unsigned int N2, // Hidden layer size
    unsigned int N3 // Output size
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute dZ2 = A2 - Y
    unsigned int total_elements = batch_size * N3;
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    compute_dZ2_kernel<<<num_blocks, threads_per_block>>>(d_A2, d_Y, d_dZ2, total_elements);

    // Compute dW2 = A1 * dZ2^T
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N2, N3, batch_size,
        &alpha,
        d_A1, N2,
        d_dZ2, N3,
        &beta,
        d_dW2, N2
    );

    // Compute db2 = sum over dZ2
    compute_bias_gradients(d_dZ2, d_db2, batch_size, N3);

    // Compute dA1 = W2 * dZ2
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N2, batch_size, N3,
        &alpha,
        d_W2, N2,
        d_dZ2, N3,
        &beta,
        d_dA1, N2
    );

    // Compute dZ1 = dA1 * ReLU'(Z1)
    total_elements = batch_size * N2;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    cudaMemcpy(d_dZ1, d_dA1, total_elements * sizeof(float), cudaMemcpyDeviceToDevice);
    activation_derivative_kernel<<<num_blocks, threads_per_block>>>(d_Z1, d_dZ1, total_elements, true);

    // Compute dW1 = X * dZ1^T
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N1, N2, batch_size,
        &alpha,
        d_X, N1,
        d_dZ1, N2,
        &beta,
        d_dW1, N1
    );

    // Compute db1 = sum over dZ1
    compute_bias_gradients(d_dZ1, d_db1, batch_size, N2);
}

// Function to compute loss (cross-entropy loss)
float compute_loss(float* d_Y, float* d_A2, unsigned int size) {
    float h_loss = 0.0f;
    float* h_Y = new float[size];
    float* h_A2 = new float[size];

    cudaMemcpy(h_Y, d_Y, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A2, d_A2, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < size; ++i) {
        float y = h_Y[i];
        float a2 = h_A2[i];
        a2 = fminf(fmaxf(a2, 1e-7f), 1.0f - 1e-7f); // Clipping
        h_loss += - (y * logf(a2) + (1 - y) * logf(1 - a2));
    }
    h_loss /= size;

    delete[] h_Y;
    delete[] h_A2;

    return h_loss;
}

// Function to compute accuracy
float compute_accuracy(float* d_Y, float* d_A2, unsigned int size) {
    float* h_Y = new float[size];
    float* h_A2 = new float[size];

    cudaMemcpy(h_Y, d_Y, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A2, d_A2, size * sizeof(float), cudaMemcpyDeviceToHost);

    unsigned int correct_predictions = 0;
    for (unsigned int i = 0; i < size; ++i) {
        float predicted = h_A2[i] >= 0.5f ? 1.0f : 0.0f;
        if (predicted == h_Y[i]) {
            ++correct_predictions;
        }
    }
    delete[] h_Y;
    delete[] h_A2;

    return (float)correct_predictions / size * 100.0f;
}
