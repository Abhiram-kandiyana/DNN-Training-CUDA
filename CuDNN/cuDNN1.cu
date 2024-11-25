#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define NUM_EPOCHS 10000

// Xavier initialization
void xavier_init(int in_dim, int out_dim, float *W) {
    float limit = sqrtf(6.0f / (in_dim + out_dim));
    for (int i = 0; i < in_dim * out_dim; ++i) {
        W[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
    }
}

// Generate parity data
void generate_parity_data(int N, int num_samples, float *X, float *Y) {
    for (int i = 0; i < num_samples; ++i) {
        int value = i;
        int parity = 0;
        for (int j = 0; j < N; ++j) {
            int bit = (value >> j) & 1;
            X[i * N + j] = (float)bit;
            parity ^= bit;
        }
        Y[i] = (float)parity;
    }
}

// CUDA kernel for weight updates
__global__ void update_weights(float *W, float *dW, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= learning_rate * dW[idx];
    }
}

// Helper function to create and set tensor descriptors
void create_and_set_tensor_descriptor(cudnnTensorDescriptor_t *desc, int n, int c, int h, int w) {
    cudnnCreateTensorDescriptor(desc);
    cudnnSetTensor4dDescriptor(*desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
}

// Helper function to create and set filter descriptors
void create_and_set_filter_descriptor(cudnnFilterDescriptor_t *desc, int k, int c, int h, int w) {
    cudnnCreateFilterDescriptor(desc);
    cudnnSetFilter4dDescriptor(*desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w);
}

int main() {
    srand(time(NULL));

    for (int N = 2; N <= 25; ++N) {
        int input_size = N;
        int hidden_size = 20;
        int output_size = 1;
        int num_samples = 1 << N; // 2^N samples

        // Memory limitation check
        if (num_samples > (1 << 20)) { // Limit samples to avoid memory issues
            printf("Skipping N = %d due to memory constraints.\n", N);
            continue;
        }

        printf("Training for input size N = %d\n", N);

        // Allocate host memory
        float *h_X = (float *)malloc(num_samples * input_size * sizeof(float));
        float *h_Y = (float *)malloc(num_samples * sizeof(float));

        // Generate parity data
        generate_parity_data(N, num_samples, h_X, h_Y);

        // Allocate device memory
        float *d_X, *d_Y;
        cudaMalloc((void **)&d_X, num_samples * input_size * sizeof(float));
        cudaMalloc((void **)&d_Y, num_samples * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_X, h_X, num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y, num_samples * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize weights and biases
        float *h_W1 = (float *)malloc(input_size * hidden_size * sizeof(float));
        float *h_b1 = (float *)calloc(hidden_size, sizeof(float));
        float *h_W2 = (float *)malloc(hidden_size * output_size * sizeof(float));
        float *h_b2 = (float *)calloc(output_size, sizeof(float));

        xavier_init(input_size, hidden_size, h_W1);
        xavier_init(hidden_size, output_size, h_W2);

        // Allocate device memory for weights and biases
        float *d_W1, *d_b1, *d_W2, *d_b2;
        cudaMalloc((void **)&d_W1, input_size * hidden_size * sizeof(float));
        cudaMalloc((void **)&d_b1, hidden_size * sizeof(float));
        cudaMalloc((void **)&d_W2, hidden_size * output_size * sizeof(float));
        cudaMalloc((void **)&d_b2, output_size * sizeof(float));

        // Copy weights and biases to device
        cudaMemcpy(d_W1, h_W1, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2, output_size * sizeof(float), cudaMemcpyHostToDevice);

        // cuDNN handle
        cudnnHandle_t cudnn;
        cudnnCreate(&cudnn);

        // Tensor descriptors
        cudnnTensorDescriptor_t input_desc, hidden_desc, output_desc;
        create_and_set_tensor_descriptor(&input_desc, num_samples, input_size, 1, 1);
        create_and_set_tensor_descriptor(&hidden_desc, num_samples, hidden_size, 1, 1);
        create_and_set_tensor_descriptor(&output_desc, num_samples, output_size, 1, 1);

        // Activation descriptor
        cudnnActivationDescriptor_t activation_desc;
        cudnnCreateActivationDescriptor(&activation_desc);
        cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0);

        // Convolution descriptor (reused for both layers)
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc,
                                        0, 0, // pad_h, pad_w
                                        1, 1, // stride_h, stride_w
                                        1, 1, // dilation_h, dilation_w
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT);

        // Filter descriptors
        cudnnFilterDescriptor_t filter_desc1, filter_desc2;
        create_and_set_filter_descriptor(&filter_desc1, hidden_size, input_size, 1, 1);
        create_and_set_filter_descriptor(&filter_desc2, output_size, hidden_size, 1, 1);

        // Workspace for convolution
        size_t workspace_bytes_fwd1 = 0, workspace_bytes_fwd2 = 0;
        size_t workspace_bytes_bwd_filter1 = 0, workspace_bytes_bwd_filter2 = 0;
        size_t workspace_bytes_bwd_data1 = 0, workspace_bytes_bwd_data2 = 0;
        size_t workspace_bytes = 0;
        void *d_workspace = NULL;

        // Get workspace sizes
        cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                input_desc,
                                                filter_desc1,
                                                conv_desc,
                                                hidden_desc,
                                                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                &workspace_bytes_fwd1);

        cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                hidden_desc,
                                                filter_desc2,
                                                conv_desc,
                                                output_desc,
                                                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                &workspace_bytes_fwd2);

        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                                                       input_desc,
                                                       hidden_desc,
                                                       conv_desc,
                                                       filter_desc1,
                                                       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                                       &workspace_bytes_bwd_filter1);

        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                                                       hidden_desc,
                                                       output_desc,
                                                       conv_desc,
                                                       filter_desc2,
                                                       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                                       &workspace_bytes_bwd_filter2);

        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn,
                                                     filter_desc2,
                                                     output_desc,
                                                     conv_desc,
                                                     hidden_desc,
                                                     CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                                     &workspace_bytes_bwd_data2);

        // Find the maximum required workspace size
        workspace_bytes = max(max(workspace_bytes_fwd1, workspace_bytes_fwd2),
                              max(max(workspace_bytes_bwd_filter1, workspace_bytes_bwd_filter2),
                                  workspace_bytes_bwd_data2));

        if (workspace_bytes > 0) {
            cudaMalloc(&d_workspace, workspace_bytes);
        }

        // Allocate device memory for activations
        float *d_hidden, *d_output;
        cudaMalloc((void **)&d_hidden, num_samples * hidden_size * sizeof(float));
        cudaMalloc((void **)&d_output, num_samples * output_size * sizeof(float));

        // Allocate device memory for gradients
        float *d_delta_output, *d_delta_hidden;
        cudaMalloc((void **)&d_delta_output, num_samples * output_size * sizeof(float));
        cudaMalloc((void **)&d_delta_hidden, num_samples * hidden_size * sizeof(float));

        float *d_dW1, *d_db1, *d_dW2, *d_db2;
        cudaMalloc((void **)&d_dW1, input_size * hidden_size * sizeof(float));
        cudaMalloc((void **)&d_db1, hidden_size * sizeof(float));
        cudaMalloc((void **)&d_dW2, hidden_size * output_size * sizeof(float));
        cudaMalloc((void **)&d_db2, output_size * sizeof(float));

        // Create bias descriptors
        cudnnTensorDescriptor_t bias_desc1, bias_desc2;
        create_and_set_tensor_descriptor(&bias_desc1, 1, hidden_size, 1, 1);
        create_and_set_tensor_descriptor(&bias_desc2, 1, output_size, 1, 1);

        // OpTensor descriptor for computing delta_output
        cudnnOpTensorDescriptor_t op_desc;
        cudnnCreateOpTensorDescriptor(&op_desc);
        cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_SUB, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

        // Start timing
        clock_t start = clock();

        // Training loop
        float alpha = 1.0f;
        float beta = 0.0f;
        float learning_rate = 0.1f;

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            // Forward pass: Input -> Hidden
            cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_desc,
                                    d_X,
                                    filter_desc1,
                                    d_W1,
                                    conv_desc,
                                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                    d_workspace,
                                    workspace_bytes,
                                    &beta,
                                    hidden_desc,
                                    d_hidden);

            // Add bias
            cudnnAddTensor(cudnn,
                           &alpha,
                           bias_desc1,
                           d_b1,
                           &alpha,
                           hidden_desc,
                           d_hidden);

            // Sigmoid activation
            cudnnActivationForward(cudnn,
                                   activation_desc,
                                   &alpha,
                                   hidden_desc,
                                   d_hidden,
                                   &beta,
                                   hidden_desc,
                                   d_hidden);

            // Forward pass: Hidden -> Output
            cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    hidden_desc,
                                    d_hidden,
                                    filter_desc2,
                                    d_W2,
                                    conv_desc,
                                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                    d_workspace,
                                    workspace_bytes,
                                    &beta,
                                    output_desc,
                                    d_output);

            // Add bias
            cudnnAddTensor(cudnn,
                           &alpha,
                           bias_desc2,
                           d_b2,
                           &alpha,
                           output_desc,
                           d_output);

            // Sigmoid activation
            cudnnActivationForward(cudnn,
                                   activation_desc,
                                   &alpha,
                                   output_desc,
                                   d_output,
                                   &beta,
                                   output_desc,
                                   d_output);

            // Compute delta_output = d_output - d_Y
            cudnnOpTensor(cudnn,
                          op_desc,
                          &alpha,
                          output_desc,
                          d_output,
                          &alpha,
                          output_desc, // Reuse output_desc instead of label_desc
                          d_Y,
                          &beta,
                          output_desc,
                          d_delta_output);

            // Backward pass through output activation
            cudnnActivationBackward(cudnn,
                                    activation_desc,
                                    &alpha,
                                    output_desc,
                                    d_output,
                                    output_desc,
                                    d_delta_output,
                                    output_desc,
                                    d_output, // x (not used for sigmoid)
                                    &beta,
                                    output_desc,
                                    d_delta_output);

            // Compute gradients w.r.t W2 and b2
            cudnnConvolutionBackwardFilter(cudnn,
                                           &alpha,
                                           hidden_desc,
                                           d_hidden,
                                           output_desc,
                                           d_delta_output,
                                           conv_desc,
                                           CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                           d_workspace,
                                           workspace_bytes,
                                           &beta,
                                           filter_desc2,
                                           d_dW2);

            cudnnConvolutionBackwardBias(cudnn,
                                         &alpha,
                                         output_desc,
                                         d_delta_output,
                                         &beta,
                                         bias_desc2,
                                         d_db2);

            // Compute delta_hidden
            cudnnConvolutionBackwardData(cudnn,
                                         &alpha,
                                         filter_desc2,
                                         d_W2,
                                         output_desc,
                                         d_delta_output,
                                         conv_desc,
                                         CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                         d_workspace,
                                         workspace_bytes,
                                         &beta,
                                         hidden_desc,
                                         d_delta_hidden);

            // Backward pass through hidden activation
            cudnnActivationBackward(cudnn,
                                    activation_desc,
                                    &alpha,
                                    hidden_desc,
                                    d_hidden,
                                    hidden_desc,
                                    d_delta_hidden,
                                    hidden_desc,
                                    d_hidden, // x (not used for sigmoid)
                                    &beta,
                                    hidden_desc,
                                    d_delta_hidden);

            // Compute gradients w.r.t W1 and b1
            cudnnConvolutionBackwardFilter(cudnn,
                                           &alpha,
                                           input_desc,
                                           d_X,
                                           hidden_desc,
                                           d_delta_hidden,
                                           conv_desc,
                                           CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                           d_workspace,
                                           workspace_bytes,
                                           &beta,
                                           filter_desc1,
                                           d_dW1);

            cudnnConvolutionBackwardBias(cudnn,
                                         &alpha,
                                         hidden_desc,
                                         d_delta_hidden,
                                         &beta,
                                         bias_desc1,
                                         d_db1);

            // Update weights and biases
            int threads_per_block = 256;
            int blocks_per_grid;

            blocks_per_grid = (input_size * hidden_size + threads_per_block - 1) / threads_per_block;
            update_weights<<<blocks_per_grid, threads_per_block>>>(d_W1, d_dW1, learning_rate, input_size * hidden_size);

            blocks_per_grid = (hidden_size * output_size + threads_per_block - 1) / threads_per_block;
            update_weights<<<blocks_per_grid, threads_per_block>>>(d_W2, d_dW2, learning_rate, hidden_size * output_size);

            blocks_per_grid = (hidden_size + threads_per_block - 1) / threads_per_block;
            update_weights<<<blocks_per_grid, threads_per_block>>>(d_b1, d_db1, learning_rate, hidden_size);

            blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;
            update_weights<<<blocks_per_grid, threads_per_block>>>(d_b2, d_db2, learning_rate, output_size);
        }

        // End timing
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time taken for N = %d: %f seconds\n", N, time_spent);

        // Cleanup
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_W1);
        cudaFree(d_b1);
        cudaFree(d_W2);
        cudaFree(d_b2);
        cudaFree(d_hidden);
        cudaFree(d_output);
        cudaFree(d_delta_output);
        cudaFree(d_delta_hidden);
        cudaFree(d_dW1);
        cudaFree(d_db1);
        cudaFree(d_dW2);
        cudaFree(d_db2);
        if (d_workspace)
            cudaFree(d_workspace);

        free(h_X);
        free(h_Y);
        free(h_W1);
        free(h_b1);
        free(h_W2);
        free(h_b2);

        cudnnDestroyOpTensorDescriptor(op_desc);
        cudnnDestroyTensorDescriptor(bias_desc1);
        cudnnDestroyTensorDescriptor(bias_desc2);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(hidden_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(filter_desc1);
        cudnnDestroyFilterDescriptor(filter_desc2);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn);
    }

    return 0;
}
