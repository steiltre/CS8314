#include <stdio.h>

#include "dense_mat.h"

#define THREADSPERBLOCK 32

/**
  * @brief Dense matrix-vector multiplication on GPU
  *
  * @param a Matrix
  * @param x Input vector
  * @param(out) y Output vector
  */
__global__ void matvec(
    float *a,
    float *x,
    float *y,
    int n);

__global__ void matvec2(
    float *a,
    float *x,
    float *y,
    int n);

__global__ void matvec3(
    float *a,
    float *x,
    float *y,
    int n);

__global__ void matvec4(
    float *a,
    float *x,
    float *y,
    int n);

/**
 * @brief Set all entries of vector to zero
 *
 * @param x Vector
 * @param n Length of vector
 */
__global__ void zerovec(
    float *x,
    int n);

int main(
    int argc,
    char ** argv)
{
  char * ifname = argv[1];
  char * ofname = NULL;
  if(argc > 2) {
    ofname = argv[2];
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsedTime;

  dense_mat * mat = dense_mat_load(ifname);
  dense_mat * x = (dense_mat*) malloc( sizeof(*x) );
  dense_mat * y = (dense_mat*) malloc( sizeof(*y) );

  x->nrows = mat->ncols;
  x->ncols = 1;
  y->nrows = mat->nrows;
  y->ncols = 1;

  x->vals = (float*) malloc( mat->ncols * sizeof(*x) );
  y->vals = (float*) malloc( mat->ncols * sizeof(*y) );

  for (int i=0; i<mat->ncols; i++) {
    x->vals[i] = i%16;
  }

  float *dev_a, *dev_x, *dev_y;

  int mat_size = mat->nrows * mat->ncols * sizeof(*dev_a);
  int row_size = mat->ncols * sizeof(*dev_a);
  int col_size = mat->nrows * sizeof(*dev_a);

  dim3 grid( ceil( float( 1*mat->nrows)/(float(THREADSPERBLOCK)) ), ceil( float(mat->nrows)/1 ) );  // Splits row into blocks have 1 x THREADSPERBLOCK threads
  dim3 block( THREADSPERBLOCK/float(1), 1 );

  dim3 grid2( ceil( float( mat->nrows)/float(THREADSPERBLOCK)) );
  dim3 block2( THREADSPERBLOCK );

  cudaMalloc((void **) &dev_a, mat_size);
  cudaMalloc((void **) &dev_x, row_size);
  cudaMalloc((void **) &dev_y, col_size);

  cudaMemcpy( dev_a, mat->vals, mat_size, cudaMemcpyHostToDevice );
  cudaMemcpy( dev_x, x->vals, row_size, cudaMemcpyHostToDevice );

  zerovec<<<ceil(float(mat->nrows)/float(THREADSPERBLOCK)), THREADSPERBLOCK>>>(dev_y,mat->nrows);

  cudaEventRecord(start);

  //matvec<<< grid, block >>> (dev_a, dev_x, dev_y, mat->nrows);
  //matvec2<<< grid, block, block.y * sizeof(float) >>> (dev_a, dev_x, dev_y, mat->nrows);
  //matvec3<<< grid2, block2 >>> (dev_a, dev_x, dev_y, mat->nrows);
  matvec4<<< grid2, block2, block2.x * sizeof(float) >>> (dev_a, dev_x, dev_y, mat->nrows);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time : %f ms\n", elapsedTime);
  printf("Effective bandwidth (GB/s): %f\n", (3*4*mat->nrows*mat->ncols * 0.000000001) / (elapsedTime*0.001));
  printf("Seems to be %f GFlops\n", ((2*mat->nrows) * mat->ncols * 0.000000001) / (elapsedTime*0.001));

  cudaMemcpy( y->vals, dev_y, col_size, cudaMemcpyDeviceToHost );

  if(ofname) {
      dense_mat_write(y, ofname);
  }

  cudaFree(dev_a);
  cudaFree(dev_x);
  cudaFree(dev_y);
}

__global__ void matvec(
    float *a,
    float *x,
    float *y,
    int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tid = row*n + col;

  if (row < n && col < n) {
    atomicAdd(&(y[row]), a[global_tid] * x[col]);
  }
}

__global__ void matvec2(
    float *a,
    float *x,
    float *y,
    int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tid = row*n + col;

  //extern __shared__ float temp[];
  __shared__ float temp;

  temp = 0;

  if (row < n && col < n) {
    //atomicAdd( &(temp[threadIdx.y]), a[global_tid] * x[col] );
    atomicAdd( &temp, a[global_tid] * x[col] );
    //atomicAdd(&(y[row]), a[global_tid] * x[col]);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    //atomicAdd( &(y[row]), temp[threadIdx.y] );
    atomicAdd( &(y[row]), temp );
  }
}

__global__ void zerovec(
    float *x,
    int n)
{
  x[ threadIdx.x + blockDim.x * blockIdx.x ] = 0;
}

__global__ void matvec3(
        float *a,
        float *x,
        float *y,
        int n)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int sum = 0;

  for (int i=0; i<n; i++) {
      sum += a[ row*n + i ] * x[i];
  }

  y[row] = sum;

}

__global__ void matvec4(
    float *a,
    float *x,
    float *y,
    int n)
{
  extern __shared__ float temp[];
  int row = blockIdx.x*blockDim.x + threadIdx.x;

  temp[threadIdx.x] = 0;

  for (int i=0; i<n; i++) {
      temp[threadIdx.x] += a[ row*n + i ] * x[i];
  }

  y[row] = temp[threadIdx.x];
}
