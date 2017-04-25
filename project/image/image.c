

#include "image.h"
#include "csr_mat.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))


/**
* @brief Ensure a value is in the range [0, 255]. Value may be nearby after
*        scaling due to floating point rounding.
*
* @param v The value to round.
*
* @return The rounded value.
*/
static inline float p_round_val(
    float const v)
{
  if(v < 0) {
    return 0.;
  }
  if(v > 255.) {
    return 255.;
  }
  return v;
}


/**
* @brief Some pixels can escape the [0, 255] range. This maps all values back
*        into the valid range.
*
* @param image The image to scale.
*/
static void p_scale_image(
    image_t * const image)
{
  int const width  = image->width;
  int const height = image->height;

  float * const restrict red   = image->red;
  float * const restrict green = image->green;
  float * const restrict blue  = image->blue;

  float minv = 255.;
  float maxv = 0.;

  #pragma omp parallel for schedule(static) \
      reduction(min: minv) reduction(max: maxv)
  for(int x=0; x < width * height; ++x) {
    minv = MIN(minv, red[x]);
    minv = MIN(minv, green[x]);
    minv = MIN(minv, blue[x]);

    maxv = MAX(maxv, red[x]);
    maxv = MAX(maxv, green[x]);
    maxv = MAX(maxv, blue[x]);
  }

  if(minv >= 0. && maxv <= 255.) {
    return;
  }

  float const scale = 255. / (maxv - minv);

  #pragma omp parallel for schedule(static)
  for(int x=0; x < width * height; ++x) {
    red[x]   = (red[x]   - minv) * scale;
    green[x] = (green[x] - minv) * scale;
    blue[x]  = (blue[x]  - minv) * scale;

    red[x] = p_round_val(red[x]);
    green[x] = p_round_val(green[x]);
    blue[x] = p_round_val(blue[x]);
  }
}

/**
 * @brief Use RGB values to create grayscale image
 *
 * @param image The image to grayscale
 *
 * @return output Grayscale image
 */
image_t * grayscale(
    image_t const * const image)
{
  image_t * output = image_alloc( image->width, image->height );

  float temp;

  for (int i=0; i < image->width * image->height; i++) {
    temp = sqrt( image->red[i]*image->red[i] + image->green[i]*image->green[i] + image->blue[i]*image->blue[i] );

    output->red[i] = temp;
    output->green[i] = temp;
    output->blue[i] = temp;
  }

  return output;
}


csr_mat * create_weight(
    image_t const * const image,
    float radius)
{

  /* Create sparse matrix assuming all pixels have 4 * radius^2 neighbors (overestimate, may want to fix later) */
  csr_mat * wgt = csr_alloc( image->height * image->width * 4 * radius * radius, image->height * image->width );

  int ind = 0;

  wgt->ptr[0] = 0;

  for (int i=0; i<image->height; i++) {
    for (int j=0; j<image->width; j++) {

      float brightness = image->red[i*image->width + j];

      for (int row = i-radius; row <= i+radius; row++) {
        for (int col = j-radius; col <= j+radius; col++) {

          float dist2 = (row-i)*(row-i) + (col-j)*(col-j);

          if (dist2 < radius*radius && row>0 && row<image->height && col>0 && col<image->width) {
            wgt->vals[ind] = exp( -1 * dist2 - ( brightness - image->red[row*image->width + col] ) * (brightness - image->red[row*image->width + col] ) );
            wgt->cols[ind] = j;
            ind++;
          }
        }
      }
      wgt->ptr[i*image->width + j + 1] = ind;
    }
  }

}


size_t image_stencil_flops(
    image_t const * const image,
    int const stencil_dim)
{
  size_t const flops_per_pixel = stencil_dim * stencil_dim * 2;
  size_t npixels = 3 * image->width * image->height; /* 3 accounts for RGB */

  /* subtract the boundary since we skip it for simplicity */
  npixels -= image->width * 2;
  npixels -= image->height * 2;

  return flops_per_pixel * npixels;
}


image_t * image_load(
    char const * const filename)
{
  int width;
  int height;
  int nchannels;
  unsigned char * stb_image = stbi_load(filename, &width, &height, &nchannels,
      3);
  if(stb_image == NULL) {
    stbi_failure_reason();
    return NULL;
  }

  /* Now split stb_image into red/green/blue channels. */
  image_t * im = image_alloc(width, height);
  for(int x=0; x < width * height; ++x) {
    im->red[x]   = (float) stb_image[(x * nchannels) + 0];
    im->green[x] = (float) stb_image[(x * nchannels) + 1];
    im->blue[x]  = (float) stb_image[(x * nchannels) + 2];
  }

  /* clean up the loaded data */
  stbi_image_free(stb_image);

  return im;
}


void image_write_bmp(
    char const * const filename,
    image_t * const image)
{
  p_scale_image(image);

  /* First merge the red/green/blue channels. */
  int const im_size = image->width * image->height;
  unsigned char * image_bytes = malloc(im_size * 3 * sizeof(*image_bytes));
  for(int x=0; x < im_size; ++x) {
    image_bytes[(x * 3) + 0] = (unsigned char) image->red[x];
    image_bytes[(x * 3) + 1] = (unsigned char) image->green[x];
    image_bytes[(x * 3) + 2] = (unsigned char) image->blue[x];
  }

  /* Now write to file. */
  int success = stbi_write_bmp(filename, image->width, image->height, 3, image_bytes);
  if(!success) {
    fprintf(stderr, "ERROR writing to '%s'\n", filename);
    stbi_failure_reason();
  }

  free(image_bytes);
}



image_t * image_alloc(
    int width,
    int height)
{
  image_t * im = malloc(sizeof(*im));

  im->width  = width;
  im->height = height;

  im->red   = malloc(width * height * sizeof(*im->red));
  im->green = malloc(width * height * sizeof(*im->green));
  im->blue  = malloc(width * height * sizeof(*im->blue));

  return im;
}


void image_free(
    image_t * im)
{
  if(im == NULL) {
    return;
  }
  free(im->red);
  free(im->green);
  free(im->blue);
  free(im);
}


