/*******************************************************************************
 * This file is part of Sedulus.
 * Copyright (c) 2022 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#ifndef FFT_H
#define FFT_H

#include <complex.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <math.h>

#include "fft_types.h"
#include "distributed_grid.h"

/* A structure for calculating kernel functions */
struct kernel {
    /* Wavevector in internal inverse length units */
    double kx,ky,kz;
    double k;
    /* The physical grid spacing dx = L/N */
    double spacing;
    /* Value of the kernel at this k */
    double complex kern;
    /* Optional extra parameters */
    const void *params;
};

static inline long int row_major_index(int i, int j, int k, long int N, long int Nz) {
    return i*N*Nz + j*Nz + k;
}

static inline long int row_major_half_transposed(int i, int j, int k, long int N, long int Nz_half) {
    return j*Nz_half*N + i*Nz_half + k;
}

static inline long long int wrap_ll(long long int i, long long int N) {
    return ((i)%(N)+(N))%(N);
}

static inline long long int row_major(long long i, long long j, long long k, long long N) {
    i = wrap_ll(i,N);
    j = wrap_ll(j,N);
    k = wrap_ll(k,N);
    return (long long int) i*N*N + j*N + k;
}

static inline long long int row_major_half(long long i, long long j, long long k, long long N) {
    i = wrap(i,N);
    j = wrap(j,N);
    k = wrap(k,N/2+1);
    return (long long int) i*(N/2+1)*N + j*(N/2+1) + k;
}

static inline void inverse_row_major(long long int id, int *x, int *y, int *z, int N) {
    int i = id % N;
    int j = (id - i)/N % N;
    int k = (id - i - j*N)/(N*N) % N;

    *z = i;
    *y = j;
    *x = k;
}

static inline double hypot3(double x, double y, double z) {
    return hypot(x, hypot(y, z));
}

/* General and FFTW wrapper functions */
void fft_wavevector(int x, int y, int z, int N, double delta_k, double *kx,
                    double *ky, double *kz, double *k);
void fft_execute(FourierPlanType plan);
GridFloatType* fft_alloc_real(size_t n);
GridComplexType* fft_alloc_complex(size_t n);
void fft_free(void *ptr);
ptrdiff_t fft_mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                MPI_Comm comm, ptrdiff_t *local_n0,
                                ptrdiff_t *local_0_start);

/* Functions for ordinary contiguous arrays */
int fft_normalize_r2c(GridComplexType *arr, int N, double boxlen);
int fft_normalize_c2r(double *arr, int N, double boxlen);
int fft_apply_kernel(GridComplexType *write, const GridComplexType *read, int N,
                     double boxlen, void (*compute)(struct kernel* the_kernel),
                     const void *params);

/* Functions for distributed grids */
int fft_r2c_dg(struct distributed_grid *dg);
int fft_c2r_dg(struct distributed_grid *dg);
int fft_apply_kernel_dg(struct distributed_grid *dg_write,
                        const struct distributed_grid *dg_read,
                        void (*compute)(struct kernel* the_kernel),
                        const void *params);


#endif
