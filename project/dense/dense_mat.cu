#include <stdio.h>

#include "dense_mat.h"

/**
  * @brief Dense matrix-vector multiplication on GPU
  *
  * @param a Matrix
  * @param x Input vector
  * @param(out) y Output vector
  */
__global__ void matvec(double *a, double *x, double *y, int n);

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

}

