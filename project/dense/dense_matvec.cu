#include <stdio.h>

#include "dense_mat.h"

#define THREADSPERBLOCK 16

/**
  * @brief Dense matrix-vector multiplication on GPU
  *
  * @param a Matrix
  * @param x Input vector
  * @param(out) y Output vector
  */
__global__ void matvec(
    double *a,
    double *x,
    double *y,
    int n);

/**
 * @brief Set all entries of vector to zero
 *
 * @param x Vector
 * @param n Length of vector
 */
__global__ void zerovec(
    double *x,
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

  dense_mat * mat = dense_mat_load(ifname);

  double * x = (double*) malloc( mat->ncols * sizeof(*x) );
  double * y;

  for (int i=0; i<mat->ncols; i++) {
    x[i] = i;
  }

  double *dev_a, *dev_x, *dev_y;

  int mat_size = mat->nrows * mat->ncols * sizeof(*dev_a);
  int row_size = mat->ncols * sizeof(*dev_a);
  int col_size = mat->nrows * sizeof(*dev_a);

  cudaMalloc((void **) &dev_a, mat_size);
  cudaMalloc((void **) &dev_x, row_size);
  cudaMalloc((void **) &dev_y, col_size);

  cudaMemcpy( dev_a, mat, mat_size, cudaMemcpyHostToDevice );
  cudaMemcpy( dev_x, x, row_size, cudaMemcpyHostToDevice );

  zerovec<<<ceil(double(mat->nrows)/double(THREADSPERBLOCK)), THREADSPERBLOCK>>>(y,mat->nrows);

  matvec<<<ceil(double(mat->nrows * mat->nrows)/double(THREADSPERBLOCK)), THREADSPERBLOCK>>> (dev_a, dev_x, dev_y, mat->nrows);

  cudaMemcpy( dev_y, y, col_size, cudaMemcpyDeviceToHost );

  cudaFree(dev_a);
  cudaFree(dev_x);
  cudaFree(dev_y);
}

__global__ void matvec(
    double *a,
    double *x,
    double *y,
    int n)
{
  //double sum = 0;

  int global_tid = threadIdx.x + blockDim.x * blockIdx.x;
  int row = global_tid / n;
  int col = global_tid - row * n;

  y[row] += a[global_tid + col] * x[col];
}

__global__ void zerovec(
    double *x,
    int n)
{
  x[ threadIdx.x + blockDim.x * blockIdx.x ] = 0;
}

