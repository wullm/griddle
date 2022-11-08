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
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

#include <assert.h>
#include <sys/time.h>

#include "../include/particle_exchange.h"
#include "../include/message.h"

#define DIV_CEIL(a, b) (((a) / (b)) + (((a) % (b)) > 0 ? 1 : 0))

/* Exchange particles between MPI ranks (Ng = N_grid != N_particle) */
int exchange_particles(struct particle *parts, double boxlen, long long int Ng,
                       long long int *num_localpart, long long int max_partnum,
                       int iteration, long long int received_left,
                       long long int received_right) {

    /* Timer */
    struct timeval time_sort_0;
    gettimeofday(&time_sort_0, NULL);

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count, MPI_Rank_Half;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    MPI_Rank_Half = MPI_Rank_Count / 2;

    /* Is there anything to do? */
    if (MPI_Rank_Count <= 1) {
        return 0;
    }

    /* Data type for MPI communication of particles */
    MPI_Datatype particle_type = mpi_particle_type();

    /* Determine the maximum width of any local slice of the grid (of size Ng^3) */
    long long int max_block_width = Ng / MPI_Rank_Count + ((Ng % MPI_Rank_Count) ? 1 : 0); //rounded up

    /* Position factors */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_block_width = max_block_width * (boxlen / Ng * pos_to_int_fac);
    const double int_to_rank_fac = 1.0 / int_block_width;

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Count the number of local particles that belong on each MPI rank */
    long long int num_send_left = 0;
    long long int num_send_right = 0;

    /* In the first iteration, particles need to be sorted */
    if (iteration == 0) {
        for (long long i = 0; i < *num_localpart; i++) {
            struct particle *p = &parts[i];

            int home_rank = p->x[0] * int_to_rank_fac;

            /* Decide whether the particle should be sent left or right */
            int dist = home_rank - rank;
            if (dist == 0) {
                p->exchange_dir = 0;
            } else if ((dist < 0 && abs(dist) <= MPI_Rank_Half) || (dist > 0 && abs(dist) >= MPI_Rank_Half)) {
                num_send_left++;
                p->exchange_dir = -1;
            } else {
                num_send_right++;
                p->exchange_dir = +1;
            }
        }

        /* Allocate temporary arrays for particle sort */
        struct particle *temp_left = malloc(num_send_left * sizeof(struct particle));
        struct particle *temp_right = malloc(num_send_right * sizeof(struct particle));
        struct particle *temp_rest = malloc((*num_localpart - num_send_right - num_send_left) * sizeof(struct particle));

        /* Sort the particles into buckets: left, centre, right */
        long int i_left = 0;
        long int i_right = 0;
        long int i_rest = 0;

        for (long long i = 0; i < *num_localpart; i++) {
            struct particle *p = &parts[i];
            if (p->exchange_dir == -1) {
                memcpy(temp_left + i_left, parts + i, sizeof(struct particle));
                i_left++;
            } else if (p->exchange_dir == 0) {
                memcpy(temp_rest + i_rest, parts + i, sizeof(struct particle));
                i_rest++;
            } else if (p->exchange_dir == 1) {
                memcpy(temp_right + i_right, parts + i, sizeof(struct particle));
                i_right++;
            }
        }

        /* Move the particles where they need to be in the main array */
        memcpy(parts, temp_left, num_send_left * sizeof(struct particle));
        memcpy(parts + num_send_left, temp_rest, (*num_localpart - num_send_right - num_send_left) * sizeof(struct particle));
        memcpy(parts + (*num_localpart - num_send_right), temp_right, num_send_right * sizeof(struct particle));

        /* Free the temporary arrays */
        free(temp_left);
        free(temp_rest);
        free(temp_right);

        // if (num_send_left + num_send_right > 0) {
        //     qsort(parts, *num_localpart, sizeof(struct particle), particleSort);
        // }
    } else {
        /* We only need to search the particles that we received from the right ... */
        for (long long i = 0; i < received_right; i++) {
            struct particle *p = &parts[i];

            int home_rank = p->x[0] * int_to_rank_fac;
            if (home_rank != rank) {
                num_send_left++;
            }
        }

        /* ... and those that we received from the left */
        for (long long i = *num_localpart - received_left; i < *num_localpart; i++) {
            struct particle *p = &parts[i];

            int home_rank = p->x[0] * int_to_rank_fac;
            if (home_rank != rank) {
                num_send_right++;
            }
        }

        /* All particles are already sorted, so we are done */
    }

    /* The index of the first particle to be sent left, respectively right */
    long long int first_send_left = 0;
    long long int first_send_right = *num_localpart - num_send_right;

    // printf("Sending %lld right, starting from %lld (%d) and %lld left, starting from %lld, and we have %lld foreigns\n", num_send_right, first_send_right, rank, num_send_left, first_send_left, foreign_particles);

    /** **/
    /* First, send particles to the right */
    /** **/

    /* Communicate the number of particles to be received */
    long long int receive_from_left;
    if (rank > 0) {
        MPI_Recv(&receive_from_left, 1, MPI_LONG_LONG, rank_left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&num_send_right, 1, MPI_LONG_LONG, rank_right,
             0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(&receive_from_left, 1, MPI_LONG_LONG, rank_left,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Allocate memory for particles to be received */
    struct particle *receive_parts_left = malloc(receive_from_left * sizeof(struct particle));

    if (rank > 0) {
        MPI_Recv(receive_parts_left, receive_from_left, particle_type,
                 rank_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&parts[first_send_right], num_send_right, particle_type,
             rank_right, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(receive_parts_left, receive_from_left, particle_type,
                 rank_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /** **/
    /* Next, send particles to the left */
    /** **/

    /* Communicate the number of particles to be received */
    long long int receive_from_right;
    if (rank > 0) {
        MPI_Recv(&receive_from_right, 1, MPI_LONG_LONG, rank_right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&num_send_left, 1, MPI_LONG_LONG, rank_left,
             0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(&receive_from_right, 1, MPI_LONG_LONG, rank_right,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Allocate memory for particles to be received */
    struct particle *receive_parts_right = malloc(receive_from_right * sizeof(struct particle));

    if (rank < MPI_Rank_Count - 1) {
        MPI_Recv(receive_parts_right, receive_from_right, particle_type,
                 rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&parts[first_send_left], num_send_left, particle_type,
             rank_left, 0, MPI_COMM_WORLD);
    if (rank == MPI_Rank_Count - 1) {
        MPI_Recv(receive_parts_right, receive_from_right, particle_type,
                 rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* For the received particles, determine whether they should be moved on */
    long long int remaining_foreign_parts = 0;
    for (long long i = 0; i < receive_from_right; i++) {
        struct particle *p = &receive_parts_right[i];

        int home_rank = p->x[0] * int_to_rank_fac;
        if (home_rank != rank) {
            remaining_foreign_parts++;
        } else {
            p->exchange_dir = 0;
        }
    }

    /* For the received particles, determine whether they should be moved on */
    for (long long i = 0; i < receive_from_left; i++) {
        struct particle *p = &receive_parts_left[i];

        int home_rank = p->x[0] * int_to_rank_fac;
        if (home_rank != rank) {
            remaining_foreign_parts++;
        } else {
            p->exchange_dir = 0;
        }
    }

    /* Just sort the received particles */
    qsort(receive_parts_left, receive_from_left, sizeof(struct particle), particleSort);
    qsort(receive_parts_right, receive_from_right, sizeof(struct particle), particleSort);

    /* Make sure that we can accommodate the received particles */
    if (*num_localpart + receive_from_left + receive_from_right - num_send_right - num_send_left > max_partnum ) {
        printf("Not enough memory to exchange particles on rank %d (%lld < %lld).\n", rank, max_partnum,
               *num_localpart + receive_from_left + receive_from_right - num_send_right - num_send_left);
        exit(1);
    }

    /* Make space for particles on the left (use memmove because of overlap) */
    memmove(parts + receive_from_right, parts + num_send_left,
            (*num_localpart - num_send_left) * sizeof(struct particle));
    /* Insert the particles received from the right on the left (to pass on) */
    memcpy(parts, receive_parts_right, receive_from_right * sizeof(struct particle));
    /* Insert the particles received from the left on the right (to pass on) */
    memcpy(parts + *num_localpart - num_send_left - num_send_right + receive_from_right,
           receive_parts_left, receive_from_left * sizeof(struct particle));

    /* Update the particle numbers */
    *num_localpart = *num_localpart + receive_from_left + receive_from_right - num_send_right - num_send_left;

    // /* It should all be sorted */
    // for (long long i = 1; i < *num_localpart; i++) {
    //     assert (parts[i].exchange_dir >= parts[i-1].exchange_dir);
    // }

    /* Free memory used for receiving particle data */
    free(receive_parts_left);
    free(receive_parts_right);

    /* Communicate the remaining numbers of foreign particles */
    long long int total_foreign_parts;
    MPI_Allreduce(&remaining_foreign_parts, &total_foreign_parts, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    /* Timer */
    struct timeval time_sort_1;
    gettimeofday(&time_sort_1, NULL);
    double seconds = ((time_sort_1.tv_sec - time_sort_0.tv_sec) * 1000000 + time_sort_1.tv_usec - time_sort_0.tv_usec)/1e6;

    if (total_foreign_parts > 0) {
        iteration++;

        message(rank, "Exchanging: %lld foreign particles remain after %d iterations (%.5f s).\n", total_foreign_parts, iteration, seconds);

        /* This should always happen within MPI_Rank_Count / 2 iterations */
        assert(iteration < MPI_Rank_Count + 1);

        exchange_particles(parts, boxlen, Ng, num_localpart, max_partnum, iteration, receive_from_left, receive_from_right);
    }

    return 0;
}
