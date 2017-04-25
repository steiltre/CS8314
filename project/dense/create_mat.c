#include <stdio.h>

int main(
    int argc,
    char ** argv)
{
  char * ofname = "Id16.mat";
  int n = 16;

  FILE * fout = fopen(ofname, "w");

  fprintf(fout, "%i %i\n", n, n);

  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) {
      if ( i==j ) {
        fprintf(fout, "%0.3e ", 1.0);
      }
      else {
        fprintf(fout, "%0.3e ", 0.0);
      }
    }
    fprintf(fout, "\n");
  }

  fclose(fout);
}
