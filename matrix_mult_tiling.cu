// Program by Arthur Alves Araujo Ferreira - All rights reserved
// ITESM ID: A01022593
// nvcc -o test matrix_mult_tiling.cu -std=c++11

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_SIZE 32

using namespace std;

// Function that multiplies 2 matrices using gpu tiling
__global__ void matrixMultiplyGPUTiling(double *A, double *B, double *C, const int n) {
    __shared__ double tileA[TILE_SIZE * TILE_SIZE];
    __shared__ double tileB[TILE_SIZE * TILE_SIZE];

    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;

    unsigned int col = x + blockIdx.x * blockDim.x;
    unsigned int row = y + blockIdx.y * blockDim.y;

    double sum = 0;
    for (int i = 0; i < (n + TILE_SIZE - 1)/TILE_SIZE; i++) {
        if (row < n && i * TILE_SIZE + x < n)
            tileA[y*TILE_SIZE+x] = A[row*n + i*TILE_SIZE+x];
        else
            tileA[y*TILE_SIZE+x] = 0.0f;

        if (col < n && i * TILE_SIZE + y < n)
            tileB[y*TILE_SIZE+x] = B[(i*TILE_SIZE+y)*n + col];
        else
            tileB[y*TILE_SIZE+x] = 0.0f;

        __syncthreads();
        for (int idx = 0; idx < TILE_SIZE; idx++) {
            sum += tileA[y*TILE_SIZE+idx] * tileB[idx*TILE_SIZE+x];
        }
        __syncthreads();
    }
    if (col < n && row < n) {
        C[col*n+row] += sum;
    }
}

// Function that multiplies 2 matrices using gpu
__global__ void matrixMultiplyGPU(double *A, double *B, double *C, const int n) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < n && iy < n) {
        for(int k = 0; k < n; k++) {
            C[iy * n + ix] += A[iy * n + k] * B[k * n + ix];
        }
    }
}

// Function that multiplies 2 matrices on cpu
void matrixMultiply(double *A, double *B, double *C, const int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[j + k * n];
            }
        }
    }
}

// Function that runs through a matrix and prints it
void printMatrix(double *matrix, const int n) {
	// Prints the matrix with numbers shortened to 2 decimals and tabs separating them
	int size = n*n;
    for (int i = 0; i < size; i++) {
        std::cout << matrix[i] << " ";
        if (i != 0 && i % n == 0)
    		  std::cout << std::endl;
	}
	return;
}

// Funtion that compares two matrices and returns boolean
bool matrixCompare(double *m_A, double *m_B, const int n) {
    bool result = true;
    int size = n*n;
    for (int i = 0; i < size; i++) {
        if (m_A[i] != m_B[i]) {
            result = false;
            break;
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
	// Set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using Device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  // Matrix information
  int n = 1000;
	int bytes = n*n * sizeof(double *);

  // Host matrices
  double *h_A = (double *)malloc(bytes);
  double *h_B = (double *)malloc(bytes);

  // Results
  double *hostRef = (double *)malloc(bytes);
  double *gpuRef = (double *)malloc(bytes);

	// Fill result matrices w zeros
	memset(hostRef, 0, bytes);
	memset(gpuRef, 0, bytes);

  // Fill input matrices with random nums between 1 and 10
  for (int i = 0; i < n; i++) {
      h_A[i] = 1 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/9));
      h_B[i] = 1 + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/9));
  }

  // Set up device
	double *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, bytes);
  cudaMalloc((void **)&d_B, bytes);
  cudaMalloc((void **)&d_C, bytes);

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, bytes);

	dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  cout <<"grid.x "<<grid.x<<" grid.y "<<grid.y<<" block.x "<<block.x<<" block.y "<<block.y<< endl;

  // Multiply and time CPU
	auto start = std::chrono::high_resolution_clock::now();
	matrixMultiply(h_A, h_B, hostRef, n);
	auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration_ms = end - start;
  double totalTime = duration_ms.count();
  cout << "Time for multiplying on cpu: " << totalTime << endl;

  // Multiply matrices in gpu
  start = std::chrono::high_resolution_clock::now();
  matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, n);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost);
  cout << (matrixCompare(hostRef, gpuRef, n) ? "Correctly multiplied both matrices (comparing GPU and CPU)" : "Incorrect GPU multiplication") << endl;

  duration_ms = end - start;
  totalTime = duration_ms.count();
  cout << "Time for multiplying on gpu: " << totalTime << endl;

  // Multiply matrices in gpu with tiling
  start = std::chrono::high_resolution_clock::now();
  matrixMultiplyGPUTiling<<<grid, block>>>(d_A, d_B, d_C, n);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost);
  cout << (matrixCompare(hostRef, gpuRef, n) ? "Correctly multiplied both matrices (comparing GPU with tiling and CPU)" : "Incorrect GPU tiling multiplication") << endl;

  duration_ms = end - start;
  totalTime = duration_ms.count();
  cout << "Time for multiplying on gpu with tiling: " << totalTime << endl;

  // Free memory that was allocated for matrixes
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  cudaDeviceReset();
  return 0;
}
