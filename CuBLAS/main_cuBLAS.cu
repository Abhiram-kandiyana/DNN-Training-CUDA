// main.cu

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel_2.cu"

int main() {
    // Seed for random number generation
    srand(42); // Fixed seed for reproducibility


    unsigned int N1;   // Input size (number of bits)
    unsigned int N2 = 20;  // Hidden layer size (adjusted for better learning)
    unsigned int N3 = 1;
    float learning_rate = 0.1f;
    unsigned int epochs = 10000;
    unsigned int threads_per_block = 256;

    // Open the output file
    FILE *output_file = fopen("gpu_cublas_output_1.txt", "a");
    if (output_file == NULL) {
        std::cerr << "Failed to open output file." << std::endl;
        return -1;
    }
    setvbuf(output_file, NULL, _IOLBF, 0);

    for(int i=11;i<26;i=i+1)
    {
        N1=i;

        // Hyperparameters
        unsigned long long total_samples = pow(2,N1);  // For 6 bits, 2^6 combinations
        unsigned long long train_size = total_samples*0.75;     // 75% for training
        // unsigned int test_size = total_samples - train_size;
       

        // Dimensions
        // Output size

        // Allocate host memory for data
        float *h_X = (float *)malloc(total_samples * N1 * sizeof(float));    // Input data
        float *h_Y = (float *)malloc(total_samples * sizeof(float));         // Labels

        // Generate dataset
        for (unsigned long long i = 0; i < total_samples; ++i) {
            unsigned long long num_ones = 0;
            for (unsigned long long j = 0; j < N1; ++j) {
                unsigned int bit = (i >> j) & 1;
                h_X[i * N1 + j] = (float)bit;
                num_ones += bit;
            }
            h_Y[i] = (num_ones % 2 == 1) ? 1.0f : 0.0f;
        }

        // Split data into training and test sets
        float *h_X_train = (float *)malloc(train_size * N1 * sizeof(float));
        float *h_Y_train = (float *)malloc(train_size * sizeof(float));
        // float h_X_test[test_size * N1];
        // float h_Y_test[test_size];
        if (h_X_train == NULL || h_Y_train == NULL) {
            std::cerr << "Failed to allocate host memory for training data." << std::endl;
            free(h_X); free(h_Y);
            return -1;
        }

        // Simple split
        memcpy(h_X_train, h_X, train_size * N1 * sizeof(float));
        memcpy(h_Y_train, h_Y, train_size * sizeof(float));
        // memcpy(h_X_test, h_X + train_size * N1, test_size * N1 * sizeof(float));
        // memcpy(h_Y_test, h_Y + train_size, test_size * sizeof(float));

        // Initialize weights and biases
        float *h_W1 = (float *)malloc(N1 * N2 * sizeof(float));          // Weights between input and hidden layer
        float *h_b1 = (float *)malloc(N2 * sizeof(float));               // Biases for hidden layer
        float *h_W2 = (float *)malloc(N2 * N3 * sizeof(float));          // Weights between hidden and output layer
        float *h_b2 = (float *)malloc(N3 * sizeof(float));                  // Biases for output layer

        if (h_W1 == NULL || h_b1 == NULL || h_W2 == NULL || h_b2 == NULL) {
            std::cerr << "Failed to allocate host memory for weights and biases." << std::endl;
            free(h_X); free(h_Y);
            free(h_X_train); free(h_Y_train);
            return -1;
        }

        // Xavier initialization
        float limit = sqrtf(6.0f / (N1 + N2));
        for (unsigned int i = 0; i < N1 * N2; ++i) {
            h_W1[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
        }
        limit = sqrtf(6.0f / (N2 + N3));
        for (unsigned int i = 0; i < N2 * N3; ++i) {
            h_W2[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
        }
        memset(h_b1, 0, N2 * sizeof(float));
        memset(h_b2, 0, N3 * sizeof(float));

        // Allocate device memory for training data
        float *d_X, *d_Y, *d_W1, *d_b1, *d_W2, *d_b2;
        float *d_Z1, *d_A1, *d_Z2, *d_A2;
        float *d_dW1, *d_db1, *d_dW2, *d_db2;
        float *d_dA1, *d_dZ1, *d_dZ2;

        cudaMalloc((void**)&d_X, train_size * N1 * sizeof(float));
        cudaMalloc((void**)&d_Y, train_size * sizeof(float));
        cudaMalloc((void**)&d_W1, N1 * N2 * sizeof(float));
        cudaMalloc((void**)&d_b1, N2 * sizeof(float));
        cudaMalloc((void**)&d_W2, N2 * N3 * sizeof(float));
        cudaMalloc((void**)&d_b2, N3 * sizeof(float));

        cudaMalloc((void**)&d_Z1, train_size * N2 * sizeof(float));
        cudaMalloc((void**)&d_A1, train_size * N2 * sizeof(float));
        cudaMalloc((void**)&d_Z2, train_size * N3 * sizeof(float));
        cudaMalloc((void**)&d_A2, train_size * N3 * sizeof(float));

        cudaMalloc((void**)&d_dW1, N1 * N2 * sizeof(float));
        cudaMalloc((void**)&d_db1, N2 * sizeof(float));
        cudaMalloc((void**)&d_dW2, N2 * N3 * sizeof(float));
        cudaMalloc((void**)&d_db2, N3 * sizeof(float));

        cudaMalloc((void**)&d_dA1, train_size * N2 * sizeof(float));
        cudaMalloc((void**)&d_dZ1, train_size * N2 * sizeof(float));
        cudaMalloc((void**)&d_dZ2, train_size * N3 * sizeof(float));

        // Copy training data to device
        cudaMemcpy(d_X, h_X_train, train_size * N1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y_train, train_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W1, h_W1, N1 * N2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1, N2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2, N2 * N3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2, N3 * sizeof(float), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Training loop
        for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
            // Zero gradients
            cudaMemset(d_dW1, 0, N1 * N2 * sizeof(float));
            cudaMemset(d_db1, 0, N2 * sizeof(float));
            cudaMemset(d_dW2, 0, N2 * N3 * sizeof(float));
            cudaMemset(d_db2, 0, N3 * sizeof(float));

            // Forward pass
            forward_pass(handle, d_X, d_W1, d_b1, d_W2, d_b2, d_Z1, d_A1, d_Z2, d_A2, train_size, N1, N2, N3);

            // Backward pass
            backward_pass(handle, d_X, d_Y, d_W1, d_W2, d_b1, d_b2, d_Z1, d_A1, d_Z2, d_A2,
                        d_dW1, d_db1, d_dW2, d_db2, d_dA1, d_dZ1, d_dZ2,
                        train_size, N1, N2, N3);

            // Update weights and biases on the device
            unsigned int num_blocks_W1 = (N1 * N2 + threads_per_block - 1) / threads_per_block;
            unsigned int num_blocks_b1 = (N2 + threads_per_block - 1) / threads_per_block;
            unsigned int num_blocks_W2 = (N2 * N3 + threads_per_block - 1) / threads_per_block;
            unsigned int num_blocks_b2 = (N3 + threads_per_block - 1) / threads_per_block;

            update_parameters_kernel<<<num_blocks_W1, threads_per_block>>>(d_W1, d_dW1, learning_rate, N1 * N2, train_size);
            update_parameters_kernel<<<num_blocks_W2, threads_per_block>>>(d_W2, d_dW2, learning_rate, N2 * N3, train_size);
            update_parameters_kernel<<<num_blocks_b1, threads_per_block>>>(d_b1, d_db1, learning_rate, N2, train_size);
            update_parameters_kernel<<<num_blocks_b2, threads_per_block>>>(d_b2, d_db2, learning_rate, N3, train_size);

            // Optionally print loss and training accuracy every 1000 epochs
            // if ((epoch + 1) % 1000 == 0) {
            //     float h_loss = compute_loss(d_Y, d_A2, train_size);
            //     float training_accuracy = compute_accuracy(d_Y, d_A2, train_size);
            //     std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << h_loss
            //               << ", Training Accuracy: " << training_accuracy << "%" << std::endl;
            // }
    }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float trainingTime;
        cudaEventElapsedTime(&trainingTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("Training for %d epochs completed\n", epochs);
        // printf("******** GPU: Inputs = %d; Total Training time = %0.4f seconds ********\n",N1, trainingTime / 1000);
        fprintf(output_file, "******** GPU: Inputs = %d; Total Training time = %0.4f seconds ********\n",N1, trainingTime / 1000);
        fflush(output_file);

        // // Compute final training accuracy
        // float training_accuracy = compute_accuracy(d_Y, d_A2, train_size);
        // std::cout << "Final Training Accuracy: " << training_accuracy << "%" << std::endl;

        // // Testing the trained model
        // // Allocate device memory for test data
        // float *d_X_test, *d_Y_test, *d_Z1_test, *d_A1_test, *d_Z2_test, *d_A2_test;
        // cudaMalloc((void**)&d_X_test, test_size * N1 * sizeof(float));
        // cudaMalloc((void**)&d_Y_test, test_size * sizeof(float));
        // cudaMalloc((void**)&d_Z1_test, test_size * N2 * sizeof(float));
        // cudaMalloc((void**)&d_A1_test, test_size * N2 * sizeof(float));
        // cudaMalloc((void**)&d_Z2_test, test_size * N3 * sizeof(float));
        // cudaMalloc((void**)&d_A2_test, test_size * N3 * sizeof(float));

        // // Copy test data to device
        // cudaMemcpy(d_X_test, h_X_test, test_size * N1 * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_Y_test, h_Y_test, test_size * sizeof(float), cudaMemcpyHostToDevice);

        // // Forward pass on test data
        // forward_pass(handle, d_X_test, d_W1, d_b1, d_W2, d_b2, d_Z1_test, d_A1_test, d_Z2_test, d_A2_test,
        //              test_size, N1, N2, N3);

        // // Compute test accuracy
        // float test_accuracy = compute_accuracy(d_Y_test, d_A2_test, test_size);
        // std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;

        // Clean up cuBLAS handle
        cublasDestroy(handle);

        // Free device memory
        cudaFree(d_X); cudaFree(d_Y);
        cudaFree(d_W1); cudaFree(d_b1);
        cudaFree(d_W2); cudaFree(d_b2);
        cudaFree(d_Z1); cudaFree(d_A1);
        cudaFree(d_Z2); cudaFree(d_A2);
        cudaFree(d_dW1); cudaFree(d_db1);
        cudaFree(d_dW2); cudaFree(d_db2);
        cudaFree(d_dA1); cudaFree(d_dZ1); cudaFree(d_dZ2);
    
        free(h_X); free(h_Y);
        free(h_X_train); free(h_Y_train);
        free(h_W1); free(h_b1);
        free(h_W2); free(h_b2);

        // cudaFree(d_X_test); cudaFree(d_Y_test);
        // cudaFree(d_Z1_test); cudaFree(d_A1_test);
        // cudaFree(d_Z2_test); cudaFree(d_A2_test);
    }

    fclose(output_file);

    return 0;
}
