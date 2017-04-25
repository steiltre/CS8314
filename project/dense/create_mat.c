#include <stdio.h>

int main(
    int argc,
    char ** argv)
{
  char * ofname = "col_large.mat";
  int n = 8192;

  FILE * fout = fopen(ofname, "w");

  fprintf(fout, "%i %i\n", n, n);

  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) {
      //if ( i==j ) {
        //fprintf(fout, "%0.3f ", 1.0);
        fprintf(fout, "%0.3f ", (double) (j%16));
      //}
      //else {
        //fprintf(fout, "%0.3f ", 0.0);
      //}
    }
    fprintf(fout, "\n");
  }

  fclose(fout);
}
