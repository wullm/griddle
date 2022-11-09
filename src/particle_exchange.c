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

// #define DEBUG_CHECKS

/* Determine the direction in which a particle should be exchanged with
 * neighbouring ranks (left, no move, right) < == > (-1, 0, 1) */
static inline int exchange_dir(IntPosType x0, double int_to_rank_fac, int rank,
                               int MPI_Rank_Half) {
    int home_rank = x0 * int_to_rank_fac;
    int dist = home_rank - rank;
    if (dist == 0) {
        return 0;
    } else if (dist >= MPI_Rank_Half || (dist < 0 && dist >= -MPI_Rank_Half)) {
        return -1;
    } else {
        return 1;
    }
}

/* Exchange particles between MPI ranks (Ng = N_grid != N_particle) */
int exchange_particles(struct particle *parts, double boxlen, long long int Ng,
                       long long int *num_localpart, long long int max_partnum,
                       int exchange_iteration, long long int received_left,
                       long long int received_right, long long int num_send_left,
                       long long int num_send_right) {

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

    /* In the first exchange iteration, particles need to be sorted */
    if (exchange_iteration == 0) {

        /* Reset the counters */
        num_send_left = 0;
        num_send_right = 0;

        for (long long i = 0; i < *num_localpart; i++) {
            struct particle *p = &parts[i];

            /* Decide whether the particle should be sent left or right */
            int exdir = exchange_dir(p->x[0], int_to_rank_fac, rank, MPI_Rank_Half);
            if (exdir == -1) {
                num_send_left++;
            } else if (exdir == 1) {
                num_send_right++;
            }
        }

        if (num_send_left > 0 || num_send_right > 0) {
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
                int exdir = exchange_dir(p->x[0], int_to_rank_fac, rank, MPI_Rank_Half);
                if (exdir == -1) {
                    memcpy(temp_left + i_left, parts + i, sizeof(struct particle));
                    i_left++;
                } else if (exdir == 0) {
                    memcpy(temp_rest + i_rest, parts + i, sizeof(struct particle));
                    i_rest++;
                } else if (exdir == 1) {
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
        }

#ifdef DEBUG_CHECKS
        /* It should all be sorted */
        for (long long i = 1; i < *num_localpart; i++) {
            // assert (parts[i].exchange_dir >= parts[i-1].exchange_dir);
            int exdir_a = exchange_dir(parts[i-1].x[0], int_to_rank_fac, rank, MPI_Rank_Half);
            int exdir_b = exchange_dir(parts[i].x[0], int_to_rank_fac, rank, MPI_Rank_Half);
            assert(exdir_b >= exdir_a);
        }
#endif

        // if (num_send_left + num_send_right > 0) {
        //     qsort(parts, *num_localpart, sizeof(struct particle), particleSort);
        // }
    }

    /* The index of the first particle to be sent left, respectively right */
    long long int first_send_left = 0;
    long long int first_send_right = *num_localpart - num_send_right;

    // printf("Sending %lld right, starting from %lld (%d) and %lld left, starting from %lld, and we have %lld foreigns\n", num_send_right, first_send_right, rank, num_send_left, first_send_left, foreign_particles);

    /* Arrays and counts of received particles */
    struct particle *receive_parts_right = NULL;
    struct particle *receive_parts_left = NULL;
    int receive_from_left = 0;
    int receive_from_right = 0;

    /* Send particles left and right, using non-blocking calls */
    MPI_Request delivery_left;
    MPI_Request delivery_right;
    MPI_Isend(&parts[first_send_left], num_send_left, particle_type,
              rank_left, 0, MPI_COMM_WORLD, &delivery_left);
    MPI_Isend(&parts[first_send_right], num_send_right, particle_type,
              rank_right, 0, MPI_COMM_WORLD, &delivery_right);

    /* Probe and receive particles from the left and right when ready */
    int finished_left = 0, finished_right = 0;
    while (!finished_left || !finished_right) {
        /* Probe and receive left, blocking only when ready */
        if (!finished_left) {
            int ready_left = 0;
            MPI_Status status_left;
            MPI_Iprobe(rank_left, 0, MPI_COMM_WORLD, &ready_left, &status_left);
            if (ready_left) {
                MPI_Get_count(&status_left, particle_type, &receive_from_left);
                receive_parts_left = malloc(receive_from_left * sizeof(struct particle));
                MPI_Recv(receive_parts_left, receive_from_left, particle_type,
                         rank_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished_left = 1;
            }
        }

        /* Probe and receive right, blocking only when ready */
        if (!finished_right) {
            int ready_right = 0;
            MPI_Status status_right;
            MPI_Iprobe(rank_right, 0, MPI_COMM_WORLD, &ready_right, &status_right);
            if (ready_right) {
                MPI_Get_count(&status_right, particle_type, &receive_from_right);
                receive_parts_right = malloc(receive_from_right * sizeof(struct particle));
                MPI_Recv(receive_parts_right, receive_from_right, particle_type,
                         rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished_right = 1;
            }
        }
    }

    /* All particles must be received at this point, though not all may be sent. */

    /* For the received particles, determine whether they should be moved on */
    long long int remaining_foreign_parts_right = 0;
    for (long long i = 0; i < receive_from_right; i++) {
        struct particle *p = &receive_parts_right[i];

        int home_rank = p->x[0] * int_to_rank_fac;
        if (home_rank != rank) {
            remaining_foreign_parts_right++;
        }
    }

    /* For the received particles, determine whether they should be moved on */
    long long int remaining_foreign_parts_left = 0;
    for (long long i = 0; i < receive_from_left; i++) {
        struct particle *p = &receive_parts_left[i];

        int home_rank = p->x[0] * int_to_rank_fac;
        if (home_rank != rank) {
            remaining_foreign_parts_left++;
        }
    }

    long long int remaining_foreign_parts = remaining_foreign_parts_left + remaining_foreign_parts_right;

    /* Just sort the received particles received from the left */
    if (receive_from_left > 0 && remaining_foreign_parts_left > 0) {
        /* Allocate temporary arrays */
        struct particle *temp_right = malloc(remaining_foreign_parts_left * sizeof(struct particle));
        struct particle *temp_rest = malloc((receive_from_left - remaining_foreign_parts_left) * sizeof(struct particle));

        /* Sort the particles into buckets: centre, right (left not possible) */
        long int i_right = 0;
        long int i_rest = 0;

        for (long long i = 0; i < receive_from_left; i++) {
            struct particle *p = &receive_parts_left[i];

            int home_rank = p->x[0] * int_to_rank_fac;
            if (home_rank == rank) {
                memcpy(temp_rest + i_rest, receive_parts_left + i, sizeof(struct particle));
                i_rest++;
            } else {
                memcpy(temp_right + i_right, receive_parts_left + i, sizeof(struct particle));
                i_right++;
            }
        }

        /* Move the particles where they need to be in the main array */
        memcpy(receive_parts_left, temp_rest, (receive_from_left - remaining_foreign_parts_left) * sizeof(struct particle));
        memcpy(receive_parts_left + (receive_from_left - remaining_foreign_parts_left), temp_right, remaining_foreign_parts_left * sizeof(struct particle));

        /* Free the temporary arrays */
        free(temp_rest);
        free(temp_right);

        // qsort(receive_parts_left, receive_from_left, sizeof(struct particle), particleSort);
    }

    /* Just sort the received particles received from the right */
    if (receive_from_right > 0) {
        /* Allocate temporary arrays */
        struct particle *temp_left = malloc(remaining_foreign_parts_right * sizeof(struct particle));
        struct particle *temp_rest = malloc((receive_from_right - remaining_foreign_parts_right) * sizeof(struct particle));

        /* Sort the particles into buckets: left, centre (right not possible) */
        long int i_left = 0;
        long int i_rest = 0;

        for (long long i = 0; i < receive_from_right; i++) {
            struct particle *p = &receive_parts_right[i];

            int home_rank = p->x[0] * int_to_rank_fac;
            if (home_rank == rank) {
                memcpy(temp_rest + i_rest, receive_parts_right + i, sizeof(struct particle));
                i_rest++;
            } else {
                memcpy(temp_left + i_left, receive_parts_right + i, sizeof(struct particle));
                i_left++;
            }
        }

        /* Move the particles where they need to be in the main array */
        memcpy(receive_parts_right, temp_left, remaining_foreign_parts_right * sizeof(struct particle));
        memcpy(receive_parts_right + remaining_foreign_parts_right, temp_rest, (receive_from_right - remaining_foreign_parts_right) * sizeof(struct particle));

        /* Free the temporary arrays */
        free(temp_left);
        free(temp_rest);

        // qsort(receive_parts_right, receive_from_right, sizeof(struct particle), particleSort);
    }

    /* Make sure that we can accommodate the received particles */
    if (*num_localpart + receive_from_left + receive_from_right - num_send_right - num_send_left > max_partnum ) {
        printf("Not enough memory to exchange particles on rank %d (%lld < %lld).\n", rank, max_partnum,
               *num_localpart + receive_from_left + receive_from_right - num_send_right - num_send_left);
        exit(1);
    }

    /* We now want to operate on the particles array, so delivery must be completed */
    MPI_Wait(&delivery_left, MPI_STATUS_IGNORE);
    MPI_Wait(&delivery_right, MPI_STATUS_IGNORE);

    /* Make space for particles on the left (use memmove because of overlap) */
    if (receive_from_right > 0 || num_send_left > 0)
        memmove(parts + receive_from_right, parts + num_send_left,
                (*num_localpart - num_send_left) * sizeof(struct particle));
    /* Insert the particles received from the right on the left (to pass on) */
    if (receive_from_right > 0)
        memcpy(parts, receive_parts_right, receive_from_right * sizeof(struct particle));
    /* Insert the particles received from the left on the right (to pass on) */
    if (receive_from_left > 0)
        memcpy(parts + *num_localpart - num_send_left - num_send_right + receive_from_right,
               receive_parts_left, receive_from_left * sizeof(struct particle));

    /* Update the particle numbers */
    *num_localpart = *num_localpart + receive_from_left + receive_from_right - num_send_right - num_send_left;

#ifdef DEBUG_CHECKS
    /* It should all be sorted */
    for (long long i = 1; i < *num_localpart; i++) {
        // assert (parts[i].exchange_dir >= parts[i-1].exchange_dir);
        int exdir_a = exchange_dir(parts[i-1].x[0], int_to_rank_fac, rank, MPI_Rank_Half);
        int exdir_b = exchange_dir(parts[i].x[0], int_to_rank_fac, rank, MPI_Rank_Half);
        assert(exdir_b >= exdir_a);
    }
#endif

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
        exchange_iteration++;

        message(rank, "Exchanging: %lld foreign particles remain after %d iterations (%.5f s).\n", total_foreign_parts, exchange_iteration, seconds);

        /* This should always happen within MPI_Rank_Count / 2 iterations */
        assert(exchange_iteration < MPI_Rank_Count + 1);

        exchange_particles(parts, boxlen, Ng, num_localpart, max_partnum, exchange_iteration, receive_from_left, receive_from_right, remaining_foreign_parts_right, remaining_foreign_parts_left);
    }

    return 0;
}
