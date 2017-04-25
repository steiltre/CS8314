#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

/**
 * @brief Structure for a dense matrix
 */
typedef struct
{
  /** Size of matrix */
  int nrows;
  int ncols;

  /** Values in matrix */
  double *vals;
} dense_mat;

/**
 * @brief Load matrix from file
 *
 * @param ifname The name of the file to read.
 *
 * @return A pointer to the matrix
 */
dense_mat * dense_mat_load(
    char const * const ifname);

/**
 * @brief Free all memory allocated by `dense_mat_load()`.
 *
 * @param mat The matrix to free.
 */
void dense_mat_free(
    dense_mat * const mat);


#endif
