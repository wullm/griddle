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

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/fft.h"

/* Compute the 3D wavevector (kx,ky,kz) and its length k */
void fft_wavevector(int x, int y, int z, int N, double delta_k, double *kx,
                    double *ky, double *kz, double *k) {
    *kx = (x > N/2) ? (x - N)*delta_k : x*delta_k;
    *ky = (y > N/2) ? (y - N)*delta_k : y*delta_k;
    *kz = (z > N/2) ? (z - N)*delta_k : z*delta_k;
    *k = sqrt((*kx)*(*kx) + (*ky)*(*ky) + (*kz)*(*kz));
}

/* Normalize the complex array after transforming to momentum space */
int fft_normalize_r2c(GridComplexType *arr, int N, double boxlen) {
    const double boxvol = boxlen*boxlen*boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                arr[row_major_half(x, y, z, N)] *= boxvol/((long long)N*N*N);
            }
        }
    }

    return 0;
}

/* Normalize the real array after transforming to configuration space */
int fft_normalize_c2r(double *arr, int N, double boxlen) {
    const double boxvol = boxlen*boxlen*boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                arr[row_major(x, y, z, N)] /= boxvol;
            }
        }
    }

    return 0;
}

/* (Distributed grid version) Normalize the complex array after transforming
 * to momentum space */
int fft_normalize_r2c_dg(struct distributed_grid *dg) {
    const int N = dg->N;
    const double boxlen = dg->boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double fac = boxvol / ((double) N * N * N);
    for (long int i = 0; i < dg->local_complex_size; i++) {
        dg->fbox[i] *= fac;
    }
    return 0;
}

/* (Distributed grid version) Normalize the real array after transforming
 * to configuration space */
int fft_normalize_c2r_dg(struct distributed_grid *dg) {
    const double boxlen = dg->boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double inv_boxvol = 1.0 / boxvol;
    for (long int i = 0; i < dg->local_real_size; i++) {
        dg->box[i] *= inv_boxvol;
    }
    return 0;
}


/* Execute an FFTW plan */
void fft_execute(FourierPlanType plan) {
#ifdef SINGLE_PRECISION_FFTW
    fftwf_execute(plan);
#else
    fftw_execute(plan);
#endif
}

/* Allocate a real domain grid using FFTW routines */
GridFloatType* fft_alloc_real(size_t n) {
#ifdef SINGLE_PRECISION_FFTW
    return fftwf_alloc_real(n);
#else
    return fftw_alloc_real(n);
#endif
}

/* Allocate a complex domain grid using FFTW routines */
GridComplexType* fft_alloc_complex(size_t n) {
#ifdef SINGLE_PRECISION_FFTW
    return fftwf_alloc_complex(n);
#else
    return fftw_alloc_complex(n);
#endif
}

/* Free memory using FFTW routines */
void fft_free(void *ptr) {
#ifdef SINGLE_PRECISION_FFTW
    fftwf_free(ptr);
#else
    fftw_free(ptr);
#endif
}

ptrdiff_t fft_mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                MPI_Comm comm, ptrdiff_t *local_n0,
                                ptrdiff_t *local_0_start) {
#ifdef SINGLE_PRECISION_FFTW
    return fftwf_mpi_local_size_3d(n0, n1, n2, comm, local_n0, local_0_start);
#else
    return fftw_mpi_local_size_3d(n0, n1, n2, comm, local_n0, local_0_start);
#endif
}


/* Apply a kernel to a 3D array after transforming to momentum space */
int fft_apply_kernel(GridComplexType *write, const GridComplexType *read, int N,
                     double boxlen, void (*compute)(struct kernel* the_kernel),
                     const void *params) {
    const double dk = 2 * M_PI / boxlen;
    const double fac = boxlen / N;

    #pragma omp parallel for
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                double kx,ky,kz,k;
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Compute the kernel */
                struct kernel the_kernel = {kx, ky, kz, k, fac, 0.f, params};
                compute(&the_kernel);

                /* Apply the kernel */
                const long long int id = row_major_half(x,y,z,N);
                write[id] = read[id] * the_kernel.kern;
            }
        }
    }

    return 0;
}

/* (Distributed grid version) Perform an r2c Fourier transform and normalize */
int fft_r2c_dg(struct distributed_grid *dg) {
    /* Create MPI FFTW plan */
    FourierPlanType r2c_mpi;
#ifdef SINGLE_PRECISION_FFTW
    r2c_mpi = fftwf_mpi_plan_dft_r2c_3d(dg->N, dg->N, dg->N, dg->box,
                                        dg->fbox, dg->comm, FFTW_ESTIMATE);
#else
    r2c_mpi = fftw_mpi_plan_dft_r2c_3d(dg->N, dg->N, dg->N, dg->box,
                                        dg->fbox, dg->comm, FFTW_ESTIMATE);
#endif

    /* Execute the Fourier transform and normalize */
    fft_execute(r2c_mpi);
    fft_normalize_r2c_dg(dg);

    /* Destroy the plan */
#ifdef SINGLE_PRECISION_FFTW
    fftwf_destroy_plan(r2c_mpi);
#else
    fftw_destroy_plan(r2c_mpi);
#endif

    /* Flip the flag for bookkeeping */
    dg->momentum_space = 1;

    return 0;
}

/* (Distributed grid version) Perform a c2r Fourier transform and normalize */
int fft_c2r_dg(struct distributed_grid *dg) {
    /* Create MPI FFTW plan */
    FourierPlanType c2r_mpi;
#ifdef SINGLE_PRECISION_FFTW
    c2r_mpi = fftwf_mpi_plan_dft_c2r_3d(dg->N, dg->N, dg->N, dg->fbox,
                                        dg->box, dg->comm, FFTW_ESTIMATE);
#else
    c2r_mpi = fftw_mpi_plan_dft_c2r_3d(dg->N, dg->N, dg->N, dg->fbox,
                                   dg->box, dg->comm, FFTW_ESTIMATE);
#endif

    /* Execute the Fourier transform and normalize */
    fft_execute(c2r_mpi);
    fft_normalize_c2r_dg(dg);

    /* Destroy the plan */
#ifdef SINGLE_PRECISION_FFTW
    fftwf_destroy_plan(c2r_mpi);
#else
    fftw_destroy_plan(c2r_mpi);
#endif

    /* Flip the trigger for bookkeeping */
    dg->momentum_space = 0;

    return 0;
}

/* (Distrbuted grid version) Apply a kernel to a complex 3D array */
int fft_apply_kernel_dg(struct distributed_grid *dg_write,
                        const struct distributed_grid *dg_read,
                        void (*compute)(struct kernel* the_kernel),
                        const void *params) {

    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 + 1) */
    const int N = dg_read->N;
    const int NX = dg_read->NX;
    const int X0 = dg_read->X0; //the local portion starts at X = X0
    const double boxlen = dg_read->boxlen;
    const double dk = 2 * M_PI / boxlen;
    const double fac = boxlen / N;

    if (dg_read->NX != dg_write->NX || dg_read->N != dg_write->N) {
        printf("Error: non-matching grid dimensions between read/write.\n");
        return 1;
    }

    if (dg_read->momentum_space != 1) {
        printf("Error: input field is not in momentum space.\n");
        return 2;
    }

    #pragma omp parallel for
    for (int x=X0; x<X0 + NX; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                double kx,ky,kz,k;
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Compute the kernel */
                struct kernel the_kernel = {kx, ky, kz, k, fac, 0.f, params};
                compute(&the_kernel);

                /* Apply the kernel */
                *point_row_major_half_dg(x, y, z, dg_write) = *point_row_major_half_dg(x, y, z, dg_read) * the_kernel.kern;
            }
        }
    }

    /* The output field is now in momentum space */
    dg_write->momentum_space = 1;

    return 0;
}
