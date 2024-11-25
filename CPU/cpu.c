#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Activation Functions
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Forward Pass Function
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
    unsigned int M = batch_size;

    // Compute Z1 = X * W1
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < N2; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < N1; ++k) {
                sum += d_X[i * N1 + k] * d_W1[k * N2 + j];
            }
            d_Z1[i * N2 + j] = sum;
        }
    }

    // Add biases and apply ReLU activation: A1 = ReLU(Z1 + b1)
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < N2; ++j) {
            d_A1[i * N2 + j] = relu(d_Z1[i * N2 + j] + d_b1[j]);
        }
    }

    // Compute Z2 = A1 * W2
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < N3; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < N2; ++k) {
                sum += d_A1[i * N2 + k] * d_W2[k * N3 + j];
            }
            d_Z2[i * N3 + j] = sum;
        }
    }

    // Add biases and apply Sigmoid activation: A2 = Sigmoid(Z2 + b2)
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < N3; ++j) {
            d_A2[i * N3 + j] = sigmoid(d_Z2[i * N3 + j] + d_b2[j]);
        }
    }
}

// Backward Pass Function
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
    unsigned int M = batch_size;

    // Compute dZ2 = A2 - Y
    for (unsigned int i = 0; i < M * N3; ++i) {
        d_dZ2[i] = d_A2[i] - d_Y[i];
    }

    // Compute dW2 = A1^T * dZ2
    for (unsigned int i = 0; i < N2; ++i) {
        for (unsigned int j = 0; j < N3; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < M; ++k) {
                sum += d_A1[k * N2 + i] * d_dZ2[k * N3 + j];
            }
            d_dW2[i * N3 + j] = sum;
        }
    }

    // Compute db2 = sum over dZ2
    for (unsigned int j = 0; j < N3; ++j) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < M; ++i) {
            sum += d_dZ2[i * N3 + j];
        }
        d_db2[j] = sum;
    }

    // Compute dA1 = dZ2 * W2^T
    float *d_dA1 = (float*) malloc(M * N2 * sizeof(float));
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < N2; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < N3; ++k) {
                sum += d_dZ2[i * N3 + k] * d_W2[j * N3 + k];
            }
            d_dA1[i * N2 + j] = sum;
        }
    }

    // Apply derivative of ReLU: dZ1 = dA1 * ReLU'(Z1)
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < N2; ++j) {
            d_dZ1[i * N2 + j] = d_dA1[i * N2 + j] * relu_derivative(d_Z1[i * N2 + j] + d_b1[j]);
        }
    }
    free(d_dA1);

    // Compute dW1 = X^T * dZ1
    for (unsigned int i = 0; i < N1; ++i) {
        for (unsigned int j = 0; j < N2; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < M; ++k) {
                sum += d_X[k * N1 + i] * d_dZ1[k * N2 + j];
            }
            d_dW1[i * N2 + j] = sum;
        }
    }

    // Compute db1 = sum over dZ1
    for (unsigned int j = 0; j < N2; ++j) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < M; ++i) {
            sum += d_dZ1[i * N2 + j];
        }
        d_db1[j] = sum;
    }
}

int main() {
    // Seed for random number generation
    srand(42); // Fixed seed for reproducibility

    // Dimensions
    // unsigned int N1 = 5;
    unsigned int N1;   // Input size (number of bits)
    unsigned int N2 = 6;  // Hidden layer size
    unsigned int N3 = 1; // Output size
     float learning_rate = 0.1f;
    unsigned int epochs = 10000;

    FILE *output_file = fopen("cpu_output.txt", "w");
    if (output_file == NULL) {
        printf("Failed to open output file.");
        return -1;
    }  
    setvbuf(output_file, NULL, _IOLBF, 0);
   
    for(int i=10;i<26;i=i+1)
    {
   
        N1=i;
        // Hyperparameters
        unsigned int total_samples = pow(2,N1);  // For 5 bits, 2^5 combinations
        // printf("total samples %d\n",total_samples);
        unsigned int train_size = total_samples*0.75;    // 75% for training
        // printf("train size %d\n",train_size);
        // unsigned int test_size = total_samples - train_size;
    

        // Allocate host memory for data
        float *h_X = (float *)malloc(total_samples * N1 * sizeof(float));    // Input data
        float *h_Y = (float *)malloc(total_samples * sizeof(float));         // Labels

        if (h_X == NULL || h_Y == NULL) {
        fprintf(stderr,"Failed to allocate host memory for data.");
        return -1;
        }

        // Generate dataset
        for (unsigned int i = 0; i < total_samples; ++i) {
            unsigned int num_ones = 0;
            for (unsigned int j = 0; j < N1; ++j) {
                unsigned int bit = (i >> j) & 1;
                h_X[i * N1 + j] = (float)bit;
                num_ones += bit;
            }
            h_Y[i] = (num_ones % 2 == 1) ? 1.0f : 0.0f;
        }

        // Split data into training and test sets
        float *h_X_train = (float *)malloc(train_size * N1 * sizeof(float));
        float *h_Y_train = (float *)malloc(train_size * sizeof(float));

        if (h_X_train == NULL || h_Y_train == NULL) {
        fprintf(stderr,"Failed to allocate host memory for training data.");
        free(h_X); free(h_Y);
        return -1;
        }
        // float h_X_test[test_size * N1];
        // float h_Y_test[test_size];

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
            fprintf(stderr,"Failed to allocate host memory for weights and biases.");
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

        // Allocate memory for training data
        float *d_X = (float*) malloc(train_size * N1 * sizeof(float));
        float *d_Y = (float*) malloc(train_size * sizeof(float));
        float *d_W1 = (float*) malloc(N1 * N2 * sizeof(float));
        float *d_b1 = (float*) malloc(N2 * sizeof(float));
        float *d_W2 = (float*) malloc(N2 * N3 * sizeof(float));
        float *d_b2 = (float*) malloc(N3 * sizeof(float));
        float *d_Z1 = (float*) malloc(train_size * N2 * sizeof(float));
        float *d_A1 = (float*) malloc(train_size * N2 * sizeof(float));
        float *d_Z2 = (float*) malloc(train_size * N3 * sizeof(float));
        float *d_A2 = (float*) malloc(train_size * N3 * sizeof(float));
        float *d_dW1 = (float*) malloc(N1 * N2 * sizeof(float));
        float *d_db1 = (float*) malloc(N2 * sizeof(float));
        float *d_dW2 = (float*) malloc(N2 * N3 * sizeof(float));
        float *d_db2 = (float*) malloc(N3 * sizeof(float));
        float *d_dZ1 = (float*) malloc(train_size * N2 * sizeof(float));
        float *d_dZ2 = (float*) malloc(train_size * N3 * sizeof(float));

        // Copy training data
        memcpy(d_X, h_X_train, train_size * N1 * sizeof(float));
        memcpy(d_Y, h_Y_train, train_size * sizeof(float));
        memcpy(d_W1, h_W1, N1 * N2 * sizeof(float));
        memcpy(d_b1, h_b1, N2 * sizeof(float));
        memcpy(d_W2, h_W2, N2 * N3 * sizeof(float));
        memcpy(d_b2, h_b2, N3 * sizeof(float));

        // Timing
        clock_t start_time = clock();

        // Training loop
        for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
            // Zero gradients
            memset(d_dW1, 0, N1 * N2 * sizeof(float));
            memset(d_db1, 0, N2 * sizeof(float));
            memset(d_dW2, 0, N2 * N3 * sizeof(float));
            memset(d_db2, 0, N3 * sizeof(float));

            // Forward pass
            forward_pass(d_X, d_W1, d_b1, d_W2, d_b2, d_Z1, d_A1, d_Z2, d_A2, train_size, N1, N2, N3);

            // Backward pass
            backward_pass(d_X, d_Y, d_W1, d_W2, d_b1, d_b2, d_Z1, d_A1, d_Z2, d_A2,
                        d_dW1, d_db1, d_dW2, d_db2, d_dZ1, d_dZ2, train_size, N1, N2, N3);

            // Update weights and biases
            for (unsigned int i = 0; i < N1 * N2; ++i) {
                d_W1[i] -= learning_rate * d_dW1[i] / train_size;
            }
            for (unsigned int i = 0; i < N2; ++i) {
                d_b1[i] -= learning_rate * d_db1[i] / train_size;
            }
            for (unsigned int i = 0; i < N2 * N3; ++i) {
                d_W2[i] -= learning_rate * d_dW2[i] / train_size;
            }
            for (unsigned int i = 0; i < N3; ++i) {
                d_b2[i] -= learning_rate * d_db2[i] / train_size;
            }

            // // Optionally print loss and training accuracy every 100 epochs
            // if ((epoch + 1) % 100 == 0) {
            //     float loss = 0.0f;
            //     unsigned int correct_predictions = 0;
            //     for (unsigned int i = 0; i < train_size; ++i) {
            //         float y = d_Y[i];
            //         float a2 = d_A2[i];
            //         a2 = fminf(fmaxf(a2, 1e-7f), 1.0f - 1e-7f); // Clipping
            //         loss += - (y * logf(a2) + (1 - y) * logf(1 - a2));

            //         // Threshold the output
            //         float predicted = a2 >= 0.5f ? 1.0f : 0.0f;
            //         if (predicted == y) {
            //             ++correct_predictions;
            //         }
            //     }
            //     loss /= train_size;
            //     float training_accuracy = (float)correct_predictions / train_size * 100.0f;
            //     printf("Epoch [%d/%d], Loss: %f, Training Accuracy: %f%%\n", epoch + 1, epochs, loss, training_accuracy);
            // }
        }

        clock_t end_time = clock();
        float training_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Training for %d epochs completed\n", epochs);
        // printf("******** CPU: inputs = %d, Total Training time = %0.4f seconds ********\n",N1,training_time);
        fprintf(output_file, "******** CPU: inputs = %d, Total Training time = %0.4f seconds ********\n",N1,training_time);
        fflush(output_file);



        // Final Training Accuracy
        // Apply thresholding to training outputs
        // for (unsigned int i = 0; i < train_size; ++i) {
        //     d_A2[i] = d_A2[i] >= 0.5f ? 1.0f : 0.0f;
        // }

        // // Calculate training accuracy
        // unsigned int correct_predictions_train = 0;
        // for (unsigned int i = 0; i < train_size; ++i) {
        //     if (d_A2[i] == d_Y[i]) {
        //         ++correct_predictions_train;
        //     }
        // }
        // float training_accuracy = (float)correct_predictions_train / train_size * 100.0f;
        // printf("Final Training Accuracy: %f%%\n", training_accuracy);

        // Testing the trained model
        // Allocate memory for test data
        // float *d_X_test = (float*) malloc(test_size * N1 * sizeof(float));
        // float *d_Y_test = (float*) malloc(test_size * sizeof(float));
        // float *d_Z1_test = (float*) malloc(test_size * N2 * sizeof(float));
        // float *d_A1_test = (float*) malloc(test_size * N2 * sizeof(float));
        // float *d_Z2_test = (float*) malloc(test_size * N3 * sizeof(float));
        // float *d_A2_test = (float*) malloc(test_size * N3 * sizeof(float));

        // Copy test data
        // memcpy(d_X_test, h_X_test, test_size * N1 * sizeof(float));
        // memcpy(d_Y_test, h_Y_test, test_size * sizeof(float));

        // Forward pass on test data
        // forward_pass(d_X_test, d_W1, d_b1, d_W2, d_b2, d_Z1_test, d_A1_test, d_Z2_test, d_A2_test, test_size, N1, N2, N3);

        // Apply thresholding to test outputs
        // for (unsigned int i = 0; i < test_size; ++i) {
        //     d_A2_test[i] = d_A2_test[i] >= 0.5f ? 1.0f : 0.0f;
        // }

        // Calculate test accuracy
        // unsigned int correct_predictions = 0;
        // for (unsigned int i = 0; i < test_size; ++i) {
        //     if (d_A2_test[i] == d_Y_test[i]) {
        //         ++correct_predictions;
        //     }
        // }
        // float test_accuracy = (float)correct_predictions / test_size * 100.0f;
        // printf("Test Accuracy: %f%%\n", test_accuracy);

        // Free allocated memory
        free(d_X);
        free(d_Y);
        free(d_W1);
        free(d_b1);
        free(d_W2);
        free(d_b2);
        free(d_Z1);
        free(d_A1);
        free(d_Z2);
        free(d_A2);
        free(d_dW1);
        free(d_db1);
        free(d_dW2);
        free(d_db2);
        free(d_dZ1);
        free(d_dZ2);
        // free(d_X_test);
        // free(d_Y_test);
        // free(d_Z1_test);
        // free(d_A1_test);
        // free(d_Z2_test);
        // free(d_A2_test);
    }

    fclose(output_file);

    return 0;
}
