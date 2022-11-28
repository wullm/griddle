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

#include "../include/analysis_fof.h"
#include "../include/analysis_so.h"
#include "../include/message.h"

#define DEBUG_CHECKS

/* Communicate copies of local FOFs centres that overlap with ranks at a
 * distance n = (exchange_iteration + 1) from the home rank. Iterates to
 * cover all distances */
int exchange_fof(struct fof_halo *fofs, double boxlen, long long int Ng,
                 long int num_local_fofs, long int *num_foreign_fofs,
                 long int num_max_fofs, double search_radius,
                 int exchange_iteration) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count, MPI_Rank_Half;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    MPI_Rank_Half = MPI_Rank_Count / 2;

    /* The MPI ranks are placed along a periodic ring */

    /* This is the nth iteration, we are communicating (n+1) ranks away */
    int n = exchange_iteration + 1;
    int rank_left = (rank < n) ? MPI_Rank_Count + rank - n : rank - n;
    int rank_right = (rank + n) % MPI_Rank_Count;

    // message(rank, "Starting FOF exchange iteration n = %d (local = %ld, foreign = %ld, max = %ld)\n", n, num_local_fofs, *num_foreign_fofs, num_max_fofs);

    /* Data type for MPI communication of FOF halos */
    MPI_Datatype fof_type = mpi_fof_halo_type();

    /* The conversion factor from integers to physical lengths */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;

    /* The conversion factor from integers to MPI rank number */
    const long long int max_block_width = Ng / MPI_Rank_Count + ((Ng % MPI_Rank_Count) ? 1 : 0); //rounded up
    const double int_block_width = max_block_width * (boxlen / Ng * pos_to_int_fac);
    const double int_to_rank_fac = 1.0 / int_block_width;

    /* Find local FOFs whose search radius overlaps with a rank at distance n */
    int count_overlap_left = 0;
    int count_overlap_right = 0;
    for (long int i = 0; i < num_local_fofs; i++) {

        /* Compute the integer x-position of the halo COM */
        IntPosType com_x = fofs[i].x_com[0] * pos_to_int_fac;

        /* Determine all ranks that overlap with the search radius */
        IntPosType min_x = com_x - search_radius * pos_to_int_fac;
        IntPosType max_x = com_x + search_radius * pos_to_int_fac;

        int min_rank = min_x * int_to_rank_fac;
        int max_rank = max_x * int_to_rank_fac;

        /* Account for wrapping */
        int dist_min = rank - min_rank;
        if (dist_min < -MPI_Rank_Half) dist_min += MPI_Rank_Count;
        else if (dist_min > MPI_Rank_Half) dist_min -= MPI_Rank_Count;
        int dist_max = max_rank - rank;
        if (dist_max < -MPI_Rank_Half) dist_max += MPI_Rank_Count;
        else if (dist_max > MPI_Rank_Half) dist_max -= MPI_Rank_Count;

        if (dist_min < 0) dist_min = -dist_min;
        if (dist_max < 0) dist_max = -dist_max;

        if (n <= dist_min) {
            count_overlap_left++;
        }

        if (n <= dist_max) {
            count_overlap_right++;
        }
    }

    /* Allocate memory for halos that should be sent */
    struct fof_halo *send_left = malloc(count_overlap_left * sizeof(struct fof_halo));
    struct fof_halo *send_right = malloc(count_overlap_right * sizeof(struct fof_halo));

    int copy_left_counter = 0;
    int copy_right_counter = 0;
    for (long int i = 0; i < num_local_fofs; i++) {

        /* Compute the integer x-position of the halo COM */
        IntPosType com_x = fofs[i].x_com[0] * pos_to_int_fac;

        /* Determine all ranks that overlap with the search radius */
        IntPosType min_x = com_x - search_radius * pos_to_int_fac;
        IntPosType max_x = com_x + search_radius * pos_to_int_fac;

        int min_rank = min_x * int_to_rank_fac;
        int max_rank = max_x * int_to_rank_fac;

        /* Account for wrapping */
        int dist_min = rank - min_rank;
        if (dist_min < -MPI_Rank_Half) dist_min += MPI_Rank_Count;
        else if (dist_min > MPI_Rank_Half) dist_min -= MPI_Rank_Count;
        int dist_max = max_rank - rank;
        if (dist_max < -MPI_Rank_Half) dist_max += MPI_Rank_Count;
        else if (dist_max > MPI_Rank_Half) dist_max -= MPI_Rank_Count;

        if (dist_min < 0) dist_min = -dist_min;
        if (dist_max < 0) dist_max = -dist_max;

        if (n <= dist_min) {
            memcpy(send_left + copy_left_counter, fofs + i, sizeof(struct fof_halo));
            copy_left_counter++;
        }

        if (n <= dist_max) {
            memcpy(send_right + copy_right_counter, fofs + i, sizeof(struct fof_halo));
            copy_right_counter++;
        }
    }

#ifdef DEBUG_CHECKS
    assert(copy_left_counter == count_overlap_left);
    assert(copy_right_counter == count_overlap_right);
#endif

    /* Arrays and counts of received FOFs */
    struct fof_halo *receive_fofs_right = NULL;
    struct fof_halo *receive_fofs_left = NULL;
    int num_receive_from_left = 0;
    int num_receive_from_right = 0;

    /* Send FOFs left and right, using non-blocking calls */
    MPI_Request delivery_left;
    MPI_Request delivery_right;
    MPI_Isend(send_left, count_overlap_left, fof_type,
              rank_left, 0, MPI_COMM_WORLD, &delivery_left);
    MPI_Isend(send_right, count_overlap_right, fof_type,
              rank_right, 0, MPI_COMM_WORLD, &delivery_right);

    /* Probe and receive FOFs from the left and right when ready */
    int finished_left = 0, finished_right = 0;
    while (!finished_left || !finished_right) {
        /* Probe and receive left, blocking only when ready */
        if (!finished_left) {
            int ready_left = 0;
            MPI_Status status_left;
            MPI_Iprobe(rank_left, 0, MPI_COMM_WORLD, &ready_left, &status_left);
            if (ready_left) {
                MPI_Get_count(&status_left, fof_type, &num_receive_from_left);
                receive_fofs_left = malloc(num_receive_from_left * sizeof(struct fof_halo));
                MPI_Recv(receive_fofs_left, num_receive_from_left, fof_type,
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
                MPI_Get_count(&status_right, fof_type, &num_receive_from_right);
                receive_fofs_right = malloc(num_receive_from_right * sizeof(struct fof_halo));
                MPI_Recv(receive_fofs_right, num_receive_from_right, fof_type,
                         rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished_right = 1;
            }
        }
    }

    /* We now want to operate on the FOF array, so delivery must be completed */
    MPI_Wait(&delivery_left, MPI_STATUS_IGNORE);
    MPI_Wait(&delivery_right, MPI_STATUS_IGNORE);

    /* Check that we have enough memory */
    if (num_local_fofs + *num_foreign_fofs + num_receive_from_left + num_receive_from_right > num_max_fofs) {
        printf("Not enough memory to exchange FOF halos on rank %d (%ld < %ld).\n", rank, num_max_fofs,
               num_local_fofs + *num_foreign_fofs + num_receive_from_left + num_receive_from_right);
        exit(1);
    }

    /* Insert the received FOFs into the main array */
    if (num_receive_from_left > 0) {
        memcpy(fofs + num_local_fofs + *num_foreign_fofs, receive_fofs_left,
               num_receive_from_left * sizeof(struct fof_halo));
        *num_foreign_fofs += num_receive_from_left;
    }
    if (num_receive_from_right > 0) {
        memcpy(fofs + num_local_fofs + *num_foreign_fofs, receive_fofs_right,
               num_receive_from_right * sizeof(struct fof_halo));
        *num_foreign_fofs += num_receive_from_right;
    }

    /* Free the delivered and received particle data */
    free(send_left);
    free(send_right);
    free(receive_fofs_left);
    free(receive_fofs_right);

    /* Communicate the remaining numbers of foreign particles */
    long long int local_sent = count_overlap_left + count_overlap_right;
    long long int total_sent;
    MPI_Allreduce(&local_sent, &total_sent, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    /* Iterate? */
    if (total_sent > 0) {

        /* This should always happen within MPI_Rank_Count / 2 iterations */
        assert(exchange_iteration < MPI_Rank_Count + 1);

        exchange_iteration = exchange_fof(fofs, boxlen, Ng, num_local_fofs, num_foreign_fofs, num_max_fofs, search_radius, exchange_iteration + 1);
    }


    return exchange_iteration;
}


/* Communicate copies of local particles that overlap with foreign FOF centres
 * with home rank a distance n = (exchange_iteration + 1) from this rank,
 * Iterates to cover all distances */
int exchange_so_parts(struct particle *parts, struct fof_halo *foreign_fofs,
                      struct so_cell_list *cell_list, long int *cell_counts,
                      long int *cell_offsets, double boxlen, long long int Ng,
                      long long int num_localpart, long long int *num_foreignpart,
                      long long int max_partnum, long int num_foreign_fofs,
                      int N_cells, double search_radius,
                      int exchange_iteration, int max_iterations) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count, MPI_Rank_Half;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    MPI_Rank_Half = MPI_Rank_Count / 2;

    /* The MPI ranks are placed along a periodic ring */

    /* This is the nth iteration, we are communicating (n+1) ranks away */
    int n = exchange_iteration + 1;
    int rank_left = (rank < n) ? MPI_Rank_Count + rank - n : rank - n;
    int rank_right = (rank + n) % MPI_Rank_Count;

    // message(rank, "Starting SO particle exchange iteration n = %d\n", n);

    /* Data type for MPI communication of particles */
    MPI_Datatype particle_type = mpi_particle_type();

    /* The conversion factor from integers to physical lengths */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_to_pos_fac = 1.0 / pos_to_int_fac;
    const double pos_to_cell_fac = N_cells / boxlen;

    /* The search radius squared */
    const double max_radius_2 = search_radius * search_radius;

    /* Find local particles that overlap with foreign FOFs at distance n */
    long int send_left_counter = 0;
    long int send_right_counter = 0;
    for (long int i = 0; i < num_foreign_fofs; i++) {
        /* Skip halos that are not at distance n */
        if (foreign_fofs[i].rank != rank_left && foreign_fofs[i].rank != rank_right) continue;

        /* Compute the integer position of the FOF COM */
        IntPosType com[3] = {foreign_fofs[i].x_com[0] * pos_to_int_fac,
                             foreign_fofs[i].x_com[1] * pos_to_int_fac,
                             foreign_fofs[i].x_com[2] * pos_to_int_fac};

        /* Determine all cells that overlap with the search radius */
        int min_x[3] = {(foreign_fofs[i].x_com[0] - search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[1] - search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[2] - search_radius) * pos_to_cell_fac};
        int max_x[3] = {(foreign_fofs[i].x_com[0] + search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[1] + search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[2] + search_radius) * pos_to_cell_fac};

        /* Loop over cells */
        for (int x = min_x[0]; x <= max_x[0]; x++) {
            for (int y = min_x[1]; y <= max_x[1]; y++) {
                for (int z = min_x[2]; z <= max_x[2]; z++) {

                    /* Handle wrapping */
                    int cx = (x < 0) ? x + N_cells : (x > N_cells - 1) ? x - N_cells : x;
                    int cy = (y < 0) ? y + N_cells : (y > N_cells - 1) ? y - N_cells : y;
                    int cz = (z < 0) ? z + N_cells : (z > N_cells - 1) ? z - N_cells : z;

                    /* Find the particle count and offset of the cell */
                    int cell = row_major_cell(cx, cy, cz, N_cells);
                    long int local_count = cell_counts[cell];
                    long int local_offset = cell_offsets[cell];

                    /* Loop over particles in cells */
                    for (int a = 0; a < local_count; a++) {
                        const int index_a = cell_list[local_offset + a].offset;

                        const IntPosType *xa = parts[index_a].x;
                        const double r2 = int_to_phys_dist2(xa, com, int_to_pos_fac);

                        if (r2 < max_radius_2) {
                            if (foreign_fofs[i].rank == rank_left) {
                                send_left_counter++;
                            } else {
                                send_right_counter++;
                            }
                        }
                    } /* End particle loop */
                }
            }
        } /* End cell loop */
    } /* End halo loop */

    // printf("%d: %ld left %ld right\n", rank, send_left_counter, send_right_counter);

    /* Allocate memory for particles that should be sent */
    struct particle *send_left = malloc(send_left_counter * sizeof(struct particle));
    struct particle *send_right = malloc(send_right_counter * sizeof(struct particle));

    /* Fish out local particles that overlap with foreign FOFs at distance n */
    long int copy_left_counter = 0;
    long int copy_right_counter = 0;
    for (long int i = 0; i < num_foreign_fofs; i++) {
        /* Skip halos that are not at distance n */
        if (foreign_fofs[i].rank != rank_left && foreign_fofs[i].rank != rank_right) continue;

        /* Compute the integer position of the FOF COM */
        IntPosType com[3] = {foreign_fofs[i].x_com[0] * pos_to_int_fac,
                             foreign_fofs[i].x_com[1] * pos_to_int_fac,
                             foreign_fofs[i].x_com[2] * pos_to_int_fac};

        /* Determine all cells that overlap with the search radius */
        int min_x[3] = {(foreign_fofs[i].x_com[0] - search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[1] - search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[2] - search_radius) * pos_to_cell_fac};
        int max_x[3] = {(foreign_fofs[i].x_com[0] + search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[1] + search_radius) * pos_to_cell_fac,
                        (foreign_fofs[i].x_com[2] + search_radius) * pos_to_cell_fac};

        /* Loop over cells */
        for (int x = min_x[0]; x <= max_x[0]; x++) {
            for (int y = min_x[1]; y <= max_x[1]; y++) {
                for (int z = min_x[2]; z <= max_x[2]; z++) {

                    /* Handle wrapping */
                    int cx = (x < 0) ? x + N_cells : (x > N_cells - 1) ? x - N_cells : x;
                    int cy = (y < 0) ? y + N_cells : (y > N_cells - 1) ? y - N_cells : y;
                    int cz = (z < 0) ? z + N_cells : (z > N_cells - 1) ? z - N_cells : z;

                    /* Find the particle count and offset of the cell */
                    int cell = row_major_cell(cx, cy, cz, N_cells);
                    long int local_count = cell_counts[cell];
                    long int local_offset = cell_offsets[cell];

                    /* Loop over particles in cells */
                    for (int a = 0; a < local_count; a++) {
                        const int index_a = cell_list[local_offset + a].offset;

                        const IntPosType *xa = parts[index_a].x;
                        const double r2 = int_to_phys_dist2(xa, com, int_to_pos_fac);

                        if (r2 < max_radius_2) {
                            if (foreign_fofs[i].rank == rank_left) {
                                memcpy(send_left + copy_left_counter, parts + index_a, sizeof(struct particle));
                                copy_left_counter++;
                            } else {
                                memcpy(send_right + copy_right_counter, parts + index_a, sizeof(struct particle));
                                copy_right_counter++;
                            }
                        }
                    } /* End particle loop */
                }
            }
        } /* End cell loop */
    } /* End halo loop */

#ifdef DEBUG_CHECKS
    assert(copy_left_counter == send_left_counter);
    assert(copy_right_counter == send_right_counter);
#endif

    /* Arrays and counts of received particles */
    struct particle *receive_parts_right = NULL;
    struct particle *receive_parts_left = NULL;
    int num_receive_from_left = 0;
    int num_receive_from_right = 0;

    /* Send particles left and right, using non-blocking calls */
    MPI_Request delivery_left;
    MPI_Request delivery_right;
    MPI_Isend(send_left, send_left_counter, particle_type,
              rank_left, 0, MPI_COMM_WORLD, &delivery_left);
    MPI_Isend(send_right, send_right_counter, particle_type,
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
                MPI_Get_count(&status_left, particle_type, &num_receive_from_left);
                receive_parts_left = malloc(num_receive_from_left * sizeof(struct particle));
                MPI_Recv(receive_parts_left, num_receive_from_left, particle_type,
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
                MPI_Get_count(&status_right, particle_type, &num_receive_from_right);
                receive_parts_right = malloc(num_receive_from_right * sizeof(struct particle));
                MPI_Recv(receive_parts_right, num_receive_from_right, particle_type,
                         rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished_right = 1;
            }
        }
    }

    /* We now want to operate on the particle array, so delivery must be completed */
    MPI_Wait(&delivery_left, MPI_STATUS_IGNORE);
    MPI_Wait(&delivery_right, MPI_STATUS_IGNORE);

    /* Check that we have enough memory */
    if (num_localpart + *num_foreignpart + num_receive_from_left + num_receive_from_right > max_partnum) {
        printf("Not enough memory to exchange SO particles on rank %d (%lld < %lld).\n", rank, max_partnum,
               num_localpart + *num_foreignpart + num_receive_from_left + num_receive_from_right);
        exit(1);
    }

    /* Insert the received FOFs into the main array */
    if (num_receive_from_left > 0) {
        memcpy(parts + num_localpart + *num_foreignpart, receive_parts_left,
               num_receive_from_left * sizeof(struct particle));
        *num_foreignpart += num_receive_from_left;
    }
    if (num_receive_from_right > 0) {
        memcpy(parts + num_localpart + *num_foreignpart, receive_parts_right,
               num_receive_from_right * sizeof(struct particle));
        *num_foreignpart += num_receive_from_right;
    }

    /* Free the delivered and received particle data */
    free(send_left);
    free(send_right);
    free(receive_parts_left);
    free(receive_parts_right);

    /* Iterate? */
    if (exchange_iteration <= max_iterations) {

        /* This should always happen within MPI_Rank_Count / 2 iterations */
        assert(exchange_iteration < MPI_Rank_Half + 1);

        exchange_iteration = exchange_so_parts(parts, foreign_fofs, cell_list,
                                               cell_counts, cell_offsets,
                                               boxlen, Ng, num_localpart,
                                               num_foreignpart, max_partnum,
                                               num_foreign_fofs, N_cells, search_radius,
                                               exchange_iteration + 1, max_iterations);
    }

    return exchange_iteration;
}



int analysis_so(struct particle *parts, struct fof_halo *fofs, double boxlen,
                long int Np, long long int Ng, long long int num_localpart,
                long long int max_partnum, long int num_local_fofs,
                int output_num, double a_scale_factor, const struct units *us,
                const struct physical_consts *pcs,
                const struct cosmology *cosmo) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Timer */
    struct timepair so_timer;
    timer_start(rank, &so_timer);

    /* Spherical overdensity search radius */
    const double max_radius = 10.0 * MPC_METRES / us->UnitLengthMetres;
    const double max_radius_2 = max_radius * max_radius;

    /* Compute the critical density */
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
#ifndef WITH_MASSES
    const double Omega_m = cosmo->Omega_cdm + cosmo->Omega_b;
    const double part_mass = rho_crit * Omega_m * pow(boxlen / Np, 3);
#endif

    /* Densitty threshold w.r.t. the critical density (TODO: make parameter) */
    const double threshold = 200.0;

    /* We start holding no foreign FOFs */
    long int num_foreign_fofs = 0;

    /* Allocate additional memory for holding some foreign FOFs */
    long int fof_buffer = 10000; //TODO: make parameter
    long int num_max_fofs = num_local_fofs + fof_buffer;
    fofs = realloc(fofs, num_max_fofs * sizeof(struct fof_halo));

    /* Exchange copies of FOF centres */
    int exchange_iterations = exchange_fof(fofs, boxlen, Ng, num_local_fofs, &num_foreign_fofs, num_max_fofs, max_radius, /* iteration = */ 0);

    timer_stop(rank, &so_timer, "Exchanging FOFs took ");
    message(rank, "It took %d iterations to exchange all FOFs\n", exchange_iterations);

    /* The initial domain decomposition into spatial cells */
    const int N_cells = boxlen / (0.25 * max_radius);
    const double int_to_cell_fac = N_cells / pow(2.0, POSITION_BITS);
    const double pos_to_cell_fac = N_cells / boxlen;

    if (N_cells > 1250) {
        printf("The number of cells is large. We should switch to larger ints (TODO).\n");
        exit(1);
    }

    /* The conversion factor from integers to physical lengths */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_to_pos_fac = 1.0 / pos_to_int_fac;

    /* Cell domain decomposition */
    const int num_cells = N_cells * N_cells * N_cells;
    long int *cell_counts = calloc(num_cells, sizeof(long int));
    long int *cell_offsets = calloc(num_cells, sizeof(long int));

    /* Now create a new particle-cell correspondence for sorting */
    struct so_cell_list *cell_list = malloc(num_localpart * sizeof(struct so_cell_list));
    for (long long i = 0; i < num_localpart; i++) {
        cell_list[i].cell = which_cell(parts[i].x, int_to_cell_fac, N_cells);
        cell_list[i].offset = i;
    }

    /* Sort particles by cell */
    qsort(cell_list, num_localpart, sizeof(struct so_cell_list), cellListSort);

#ifdef DEBUG_CHECKS
    /* Check the sort */
    for (long long i = 1; i < num_localpart; i++) {
        assert(cell_list[i].cell >= cell_list[i - 1].cell);
    }
#endif

    /* Timer */
    timer_stop(rank, &so_timer, "Sorting particles took ");

    /* Reset the cell particle counts and offsets */
    for (int i = 0; i < num_cells; i++) {
        cell_counts[i] = 0;
        cell_offsets[i] = 0;
    }

    /* Count particles in cells */
    for (long long i = 0; i < num_localpart; i++) {
        int c = cell_list[i].cell;
#ifdef DEBUG_CHECKS
        assert((c >= 0) && (c < num_cells));
#endif
        cell_counts[c]++;
    }

    /* Determine the offsets, using the fact that the particles are sorted */
    cell_offsets[0] = 0;
    for (int i = 1; i < num_cells; i++) {
        cell_offsets[i] = cell_offsets[i-1] + cell_counts[i-1];
    }

    /* We start holding no foreign particles */
    long long int num_foreign_parts = 0;

    /* Next, exchange copies of particles */
    exchange_so_parts(parts, fofs + num_local_fofs, cell_list, cell_counts,
                      cell_offsets, boxlen, Ng, num_localpart,
                      &num_foreign_parts, max_partnum, num_foreign_fofs,
                      N_cells, max_radius, /* iter = */ 0,
                      /* max_iter = */ exchange_iterations);

    /* Timer */
    timer_stop(rank, &so_timer, "Exchanging particles took ");
    // printf("%d holds %lld foreign parts for local FOFs\n", rank, num_foreign_parts);

    /* Append the foreign particles to the particle-cell correspondence */
    cell_list = realloc(cell_list, (num_localpart + num_foreign_parts) * sizeof(struct so_cell_list));
    for (long long i = num_localpart; i < num_localpart + num_foreign_parts; i++) {
        cell_list[i].cell = which_cell(parts[i].x, int_to_cell_fac, N_cells);
        cell_list[i].offset = i;
    }

    /* Sort particles by cell */
    qsort(cell_list, num_localpart + num_foreign_parts, sizeof(struct so_cell_list), cellListSort);

#ifdef DEBUG_CHECKS
    /* Check the sort */
    for (long long i = 1; i < num_localpart + num_foreign_parts; i++) {
        assert(cell_list[i].cell >= cell_list[i - 1].cell);
    }
#endif

    timer_stop(rank, &so_timer, "Sorting particles took ");

    /* Reset the cell particle counts and offsets */
    for (int i = 0; i < num_cells; i++) {
        cell_counts[i] = 0;
        cell_offsets[i] = 0;
    }

    /* Count particles in cells */
    for (long long i = 0; i < num_localpart + num_foreign_parts; i++) {
        int c = cell_list[i].cell;
#ifdef DEBUG_CHECKS
        assert((c >= 0) && (c < num_cells));
#endif
        cell_counts[c]++;
    }

    /* Determine the offsets, using the fact that the particles are sorted */
    cell_offsets[0] = 0;
    for (int i = 1; i < num_cells; i++) {
        cell_offsets[i] = cell_offsets[i-1] + cell_counts[i-1];
    }


    /* Allocate memory for spherical overdensity halo properties */
    struct so_halo *halos = malloc(num_local_fofs * sizeof(struct so_halo));
    bzero(halos, num_local_fofs * sizeof(struct so_halo));

    /* The FOF and SO halos are in one-to-one correspondence */
    for (long int i = 0; i < num_local_fofs; i++) {
        halos[i].global_id = fofs[i].global_id;
        halos[i].rank = fofs[i].rank;
    }

    /* Allocate working memory for computing the SO radius */
    /* Start with a reasonable length, reallocate if more is needed */
    long int working_space = 10000;
    struct so_part_data *so_parts = malloc(working_space * sizeof(struct so_part_data));

    /* Loop over local halos */
    for (long int i = 0; i < num_local_fofs; i++) {

        /* Compute the integer position of the FOF COM */
        IntPosType com[3] = {fofs[i].x_com[0] * pos_to_int_fac,
                             fofs[i].x_com[1] * pos_to_int_fac,
                             fofs[i].x_com[2] * pos_to_int_fac};

        /* Determine all cells that overlap with the search radius */
        int min_x[3] = {(fofs[i].x_com[0] - max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[1] - max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[2] - max_radius) * pos_to_cell_fac};
        int max_x[3] = {(fofs[i].x_com[0] + max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[1] + max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[2] + max_radius) * pos_to_cell_fac};

        /* Count the number of particles within the search radius */
        long int nearby_partnum = 0;

        /* Loop over cells to count particles */
        for (int x = min_x[0]; x <= max_x[0]; x++) {
            for (int y = min_x[1]; y <= max_x[1]; y++) {
                for (int z = min_x[2]; z <= max_x[2]; z++) {

                    /* Handle wrapping */
                    int cx = (x < 0) ? x + N_cells : (x > N_cells - 1) ? x - N_cells : x;
                    int cy = (y < 0) ? y + N_cells : (y > N_cells - 1) ? y - N_cells : y;
                    int cz = (z < 0) ? z + N_cells : (z > N_cells - 1) ? z - N_cells : z;

                    /* Find the particle count and offset of the cell */
                    int cell = row_major_cell(cx, cy, cz, N_cells);
                    long int local_count = cell_counts[cell];
                    long int local_offset = cell_offsets[cell];

                    /* Loop over particles in cells */
                    for (int a = 0; a < local_count; a++) {
                        const int index_a = cell_list[local_offset + a].offset;

                        const IntPosType *xa = parts[index_a].x;
                        const double r2 = int_to_phys_dist2(xa, com, int_to_pos_fac);

                        if (r2 < max_radius_2) {
                            nearby_partnum++;
                        }
                    } /* End particle loop */
                }
            }
        } /* End cell loop */

        /* Allocate more memory if needed */
        if (nearby_partnum > working_space) {
            so_parts = realloc(so_parts, nearby_partnum * sizeof(struct so_part_data));
        }

        /* Erase the working memory */
        bzero(so_parts, nearby_partnum * sizeof(struct so_part_data));

        /* Loop over cells to create an array of distances */
        long int part_counter = 0;
        for (int x = min_x[0]; x <= max_x[0]; x++) {
            for (int y = min_x[1]; y <= max_x[1]; y++) {
                for (int z = min_x[2]; z <= max_x[2]; z++) {

                    /* Handle wrapping */
                    int cx = (x < 0) ? x + N_cells : (x > N_cells - 1) ? x - N_cells : x;
                    int cy = (y < 0) ? y + N_cells : (y > N_cells - 1) ? y - N_cells : y;
                    int cz = (z < 0) ? z + N_cells : (z > N_cells - 1) ? z - N_cells : z;

                    /* Find the particle count and offset of the cell */
                    int cell = row_major_cell(cx, cy, cz, N_cells);
                    long int local_count = cell_counts[cell];
                    long int local_offset = cell_offsets[cell];

                    /* Loop over particles in cells */
                    for (int a = 0; a < local_count; a++) {
                        const int index_a = cell_list[local_offset + a].offset;

                        const IntPosType *xa = parts[index_a].x;
                        const double r2 = int_to_phys_dist2(xa, com, int_to_pos_fac);

                        if (r2 < max_radius_2) {
                            so_parts[part_counter].m = parts[index_a].m;
                            so_parts[part_counter].r = sqrtf(r2);
                            part_counter++;
                        }
                    } /* End particle loop */
                }
            }
        } /* End cell loop */

        /* Sort particles by radial distance */
        qsort(so_parts, nearby_partnum, sizeof(struct so_part_data), soPartSort);

        /* Compute the enclosed mass and density ratio at each particle */
        const double dens_fac = (4.0 / 3.0) * M_PI * rho_crit;
        const double inv_fac = 1.0 / dens_fac;
        for (long int j = 1; j < nearby_partnum; j++) {
            so_parts[j].m += so_parts[j - 1].m;
            if (so_parts[j].r > 0) {
                double r3 = so_parts[j].r * so_parts[j].r * so_parts[j].r;
                so_parts[j].Delta = so_parts[j].m * inv_fac / r3;
            }
        }

        /* Find the first particle that exceeds the threshold */
        long int first_above = -1;
        for (long int j = 0; j < nearby_partnum; j++) {
            /* Skip particles at zero radial distance */
            if (so_parts[j].r == 0) continue;

            if (so_parts[j].Delta >= threshold) {
                first_above = j;
                break;
            }
        }

        /* If no particle exceeds the threshold, interpolate up to particle 1 */
        if (first_above == -1) {
            halos[i].R_SO = sqrt(so_parts[0].m * inv_fac / (threshold * so_parts[0].r));
            halos[i].M_SO = halos[i].R_SO * halos[i].R_SO * halos[i].R_SO * dens_fac * threshold;
        } else {
            /* Find the first particle after this that drops below the threshold */
            long int first_below = -1;
            for (long int j = first_above; j < nearby_partnum; j++) {
                /* Skip particles at zero radial distance */
                if (so_parts[j].r == 0) continue;

                if (so_parts[j].Delta < threshold) {
                    first_below = j;
                    break;
                }
            }

            /* No particle below the threshold, we need to expand the search radius */
            if (first_below == -1) {
               printf("Error: No particle below the SO density threshold. Search radius too small.\n");
               exit(1);
            }

            /* We have found an interval where the density drops below the threshold */
            double delta_Delta = so_parts[first_below].Delta - so_parts[first_below - 1].Delta;
            double delta_r = so_parts[first_below].r - so_parts[first_below - 1].r;

            /* If there is no gradient, then use midpoint (TODO: not ideal) */
            if (delta_Delta == 0) {
                halos[i].R_SO = so_parts[first_below - 1].r + 0.5 * delta_r;
                halos[i].M_SO = halos[i].R_SO * halos[i].R_SO * halos[i].R_SO * dens_fac * threshold;
                printf("Warning: have two particles with no density gradient r = (%g %g) Delta = (%g %g)\n", so_parts[first_below - 1].r, so_parts[first_below].r, so_parts[first_below - 1].Delta, so_parts[first_below].Delta);
            } else {
                /* Linearly interpolate to find the SO radius and mass */
                halos[i].R_SO = so_parts[first_below - 1].r + (threshold - so_parts[first_below - 1].Delta) * delta_r / delta_Delta;
                halos[i].M_SO = halos[i].R_SO * halos[i].R_SO * halos[i].R_SO * dens_fac * threshold;
            }


        }

    } /* End halo loop */

    /* Free memory for SO part data */
    free(so_parts);

    /* Timer */
    timer_stop(rank, &so_timer, "Computing spherical overdensity radii took ");

    /* Now compute other SO properties */
    for (long int i = 0; i < num_local_fofs; i++) {

        /* Compute the integer position of the FOF COM */
        IntPosType com[3] = {fofs[i].x_com[0] * pos_to_int_fac,
                             fofs[i].x_com[1] * pos_to_int_fac,
                             fofs[i].x_com[2] * pos_to_int_fac};

        /* Determine all cells that overlap with the search radius */
        int min_x[3] = {(fofs[i].x_com[0] - max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[1] - max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[2] - max_radius) * pos_to_cell_fac};
        int max_x[3] = {(fofs[i].x_com[0] + max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[1] + max_radius) * pos_to_cell_fac,
                        (fofs[i].x_com[2] + max_radius) * pos_to_cell_fac};

        /* The square of the SO radius */
        double R_SO_2 = halos[i].R_SO * halos[i].R_SO;

        /* Loop over cells */
        for (int x = min_x[0]; x <= max_x[0]; x++) {
            for (int y = min_x[1]; y <= max_x[1]; y++) {
                for (int z = min_x[2]; z <= max_x[2]; z++) {

                    /* Handle wrapping */
                    int cx = (x < 0) ? x + N_cells : (x > N_cells - 1) ? x - N_cells : x;
                    int cy = (y < 0) ? y + N_cells : (y > N_cells - 1) ? y - N_cells : y;
                    int cz = (z < 0) ? z + N_cells : (z > N_cells - 1) ? z - N_cells : z;

                    /* Find the particle count and offset of the cell */
                    int cell = row_major_cell(cx, cy, cz, N_cells);
                    long int local_count = cell_counts[cell];
                    long int local_offset = cell_offsets[cell];

                    /* Loop over particles in cells */
                    for (int a = 0; a < local_count; a++) {
                        const int index_a = cell_list[local_offset + a].offset;

                        const IntPosType *xa = parts[index_a].x;
                        const double r2 = int_to_phys_dist2(xa, com, int_to_pos_fac);

                        if (r2 < R_SO_2) {
#ifdef WITH_MASSES
                            double mass = parts[index_a].m;
#else
                            double mass = part_mass;
#endif
                            halos[i].x_com[0] += (int_to_pos_fac * parts[index_a].x[0]) * mass;
                            halos[i].x_com[1] += (int_to_pos_fac * parts[index_a].x[1]) * mass;
                            halos[i].x_com[2] += (int_to_pos_fac * parts[index_a].x[2]) * mass;
                            halos[i].v_com[0] += parts[index_a].v[0] * mass;
                            halos[i].v_com[1] += parts[index_a].v[1] * mass;
                            halos[i].v_com[2] += parts[index_a].v[2] * mass;
                            halos[i].v_com[2] += parts[index_a].v[2] * mass;
                            halos[i].mass_tot += mass;
                            halos[i].npart_tot++;
                        }
                    } /* End particle loop */
                }
            }
        } /* End cell loop */
    } /* End halo loop */

    /* Divide by the mass for the centre of mass properties */
    for (long int i = 0; i < num_local_fofs; i++) {
        double halo_mass = halos[i].mass_tot;
        if (halo_mass > 0) {
            double inv_halo_mass = 1.0 / halo_mass;
            halos[i].x_com[0] *= inv_halo_mass;
            halos[i].x_com[1] *= inv_halo_mass;
            halos[i].x_com[2] *= inv_halo_mass;
            halos[i].v_com[0] *= inv_halo_mass;
            halos[i].v_com[1] *= inv_halo_mass;
            halos[i].v_com[2] *= inv_halo_mass;
        }
    }

    /* Print the halo properties to a file */
    /* TODO: replace by HDF5 output */
    char fname[50];
    sprintf(fname, "halos_SO_%04d_%03d.txt", output_num, rank);
    FILE *f = fopen(fname, "w");

    fprintf(f, "# i M_FOF npart_FOF M_tot M_SO R_SO npart_SO x[0] x[1] x[2] v[0] v[1] v[2] x_fof[0] x_fof[1] x_fof[2] \n");
    for (long int i = 0; i < num_local_fofs; i++) {
        fprintf(f, "%ld %g %d %g %g %g %d %g %g %g %g %g %g %g %g %g\n", fofs[i].global_id, fofs[i].mass_fof, fofs[i].npart, halos[i].mass_tot, halos[i].M_SO, halos[i].R_SO, halos[i].npart_tot, halos[i].x_com[0], halos[i].x_com[1], halos[i].x_com[2], halos[i].v_com[0], halos[i].v_com[1], halos[i].v_com[2], fofs[i].x_com[0], fofs[i].x_com[1], fofs[i].x_com[2]);
    }

    /* Close the file */
    fclose(f);

    /* Timer */
    timer_stop(rank, &so_timer, "Writing SO halo properties took ");

    /* Free all memory */
    free(cell_list);
    free(cell_counts);
    free(cell_offsets);
    free(halos);

    return 0;
}
