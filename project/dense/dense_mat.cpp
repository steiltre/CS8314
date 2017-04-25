

/* ensure we have `getline()` */
#ifndef _POSITX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "dense_mat.h"

dense_mat * dense_mat_load(
    char const * const ifname)
{
  FILE * fin = fopen(ifname, "r");
  if(!fin) {
    fprintf(stderr, "ERROR: could not open '%s' for reading.\n", ifname);
    return NULL;
  }

  dense_mat * mat = (dense_mat*) malloc(sizeof(*mat));

  /* read matrix dimensions */
  fscanf(fin, "%u", &(mat->nrows));
  fscanf(fin, "%u", &(mat->ncols));
  fscanf(fin, "\n"); /* make sure we process the newline, too. */

  mat->vals = (float*) malloc( mat->nrows * mat->ncols * sizeof(mat->vals));

  for(int i=0; i<mat->nrows*mat->ncols; i++) {
    fscanf(fin, "%f", mat->vals + i);
  }

  return mat;
}

void dense_mat_write(
    dense_mat * mat,
    char const * const ofname)
{
  FILE * fout = fopen(ofname, "w");

  fprintf(fout, "%i ", mat->nrows);
  fprintf(fout, "%i\n", mat->ncols);

  for(int i=0; i<mat->nrows; i++) {
    for(int j=0; j<mat->ncols; j++) {
      fprintf(fout, "%0.3f ", mat->vals[ i*mat->ncols + j]);
    }
    fprintf(fout, "\n");
  }

}

void dense_mat_free(
    dense_mat * const mat)
{
  free(mat->vals);
}
