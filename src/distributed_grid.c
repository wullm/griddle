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

#include <stdlib.h>
#include "../include/distributed_grid.h"
#include "../include/fft.h"

int alloc_local_grid(struct distributed_grid *dg, int N, double boxlen, MPI_Comm comm) {
    /* Determine the size of the local portion */
    dg->local_size = fftw_mpi_local_size_3d(N, N, N/2+1, comm, &dg->NX, &dg->X0);

    /* Store a reference to the communicator */
    dg->comm = comm;

    /* Store basic attributes */
    dg->N = N;
    dg->boxlen = boxlen;

    /* Allocate memory for the complex and real arrays */
    dg->fbox = fftw_alloc_complex(dg->local_size);
    dg->box = fftw_alloc_real(2*dg->local_size);

    /* This flag will be flipped each time we do a Fourier transform */
    dg->momentum_space = 0;

    /* By default, we do not allocate buffers */
    dg->buffer_size = 0;

    return 0;
}

int free_local_grid(struct distributed_grid *dg) {
    free_local_real_grid(dg);
    free_local_complex_grid(dg);
    if (dg->buffer_size > 0) {
        free_local_buffers(dg);
    }
    return 0;
}

int free_local_real_grid(struct distributed_grid *dg) {
    fftw_free(dg->box);
    return 0;
}

int free_local_complex_grid(struct distributed_grid *dg) {
    fftw_free(dg->fbox);
    return 0;
}

int alloc_local_buffers(struct distributed_grid *dg, int buffer_size) {
    dg->buffer_size = buffer_size;
    dg->buffer_left = fftw_alloc_real(buffer_size * dg->N * dg->N);
    dg->buffer_right = fftw_alloc_real(buffer_size * dg->N * dg->N);
    return 0;
}

int free_local_buffers(struct distributed_grid *dg) {
    fftw_free(dg->buffer_left);
    fftw_free(dg->buffer_right);
    return 0;
}

/* Make copies of the edges of the local grids and communicate them to
 * neighbouring MPI ranks. Assumes memory has already been allocated. */
int create_local_buffers(struct distributed_grid *dg) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(dg->comm, &rank);
    MPI_Comm_size(dg->comm, &MPI_Rank_Count);

    /* Is there anything to do? */
    if (MPI_Rank_Count < 2) return 0;

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Sizes of the buffers */
    long long int size = dg->buffer_size * dg->N * dg->N;

    /* Allocate temporary arrays for the buffers */
    double *send_buffer_left = fftw_alloc_real(size);
    double *send_buffer_right = fftw_alloc_real(size);

    /* Create the right buffer of the left neighbour */
    int first_x = dg->X0;
    for (int i = 0; i < dg->buffer_size; i++) {
        for (int j = 0; j < dg->N; j++) {
            for (int k = 0; k < dg->N; k++) {
                send_buffer_left[row_major(i, j, k, dg->N)] = dg->box[row_major_dg2(i + first_x, j, k, dg)];
            }
        }
    }

    /* Create the left buffer of the right neighbour */
    first_x = dg->X0 + dg->NX - dg->buffer_size;
    for (int i = 0; i < dg->buffer_size; i++) {
        for (int j = 0; j < dg->N; j++) {
            for (int k = 0; k < dg->N; k++) {
                send_buffer_right[row_major(i, j, k, dg->N)] = dg->box[row_major_dg2(i + first_x, j, k, dg)];
            }
        }
    }

    /* Send the buffer to the right */
    if (rank > 0) {
        MPI_Recv(dg->buffer_left, size, MPI_DOUBLE, rank_left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(send_buffer_right, size, MPI_DOUBLE, rank_right, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(dg->buffer_left, size, MPI_DOUBLE, rank_left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Send the buffer to the left */
    if (rank > 0) {
        MPI_Recv(dg->buffer_right, size, MPI_DOUBLE, rank_right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(send_buffer_left, size, MPI_DOUBLE, rank_left, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(dg->buffer_right, size, MPI_DOUBLE, rank_right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Free the temporary arrays */
    fftw_free(send_buffer_left);
    fftw_free(send_buffer_right);

    return 0;
}

/* Send the buffers to the neighbour and add the contributions to the main grid.*/
int add_local_buffers(struct distributed_grid *dg) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(dg->comm, &rank);
    MPI_Comm_size(dg->comm, &MPI_Rank_Count);

    /* Is there anything to do? */
    if (MPI_Rank_Count < 2) return 0;

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Sizes of the buffers */
    long long int size = dg->buffer_size * dg->N * dg->N;

    /* Allocate temporary arrays for the buffers */
    double *recv_buffer_left = fftw_alloc_real(size);
    double *recv_buffer_right = fftw_alloc_real(size);


    /* Send the buffer to the right */
    if (rank > 0) {
        MPI_Recv(recv_buffer_left, size, MPI_DOUBLE, rank_left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(dg->buffer_right, size, MPI_DOUBLE, rank_right, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(recv_buffer_left, size, MPI_DOUBLE, rank_left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Send the buffer to the left */
    if (rank > 0) {
        MPI_Recv(recv_buffer_right, size, MPI_DOUBLE, rank_right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(dg->buffer_left, size, MPI_DOUBLE, rank_left, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(recv_buffer_right, size, MPI_DOUBLE, rank_right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    /* Add the left buffer to the main grid */
    int first_x = dg->X0;
    for (int i = 0; i < dg->buffer_size; i++) {
        for (int j = 0; j < dg->N; j++) {
            for (int k = 0; k < dg->N; k++) {
                dg->box[row_major_dg2(i + first_x, j, k, dg)] += recv_buffer_left[row_major(i, j, k, dg->N)];
            }
        }
    }

    /* Add the right buffer to the main grid */
    first_x = dg->X0 + dg->NX - dg->buffer_size;
    for (int i = 0; i < dg->buffer_size; i++) {
        for (int j = 0; j < dg->N; j++) {
            for (int k = 0; k < dg->N; k++) {
                dg->box[row_major_dg2(i + first_x, j, k, dg)] += recv_buffer_right[row_major(i, j, k, dg->N)];
            }
        }
    }

    /* Free the temporary arrays */
    fftw_free(recv_buffer_left);
    fftw_free(recv_buffer_right);

    return 0;
}
