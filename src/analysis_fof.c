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

#include <fftw3.h>
#include <assert.h>
#include <sys/time.h>

#include "../include/fft_types.h"
#include "../include/analysis_fof.h"
#include "../include/analysis_so.h"
#include "../include/catalogue_io.h"
#include "../include/message.h"

#define DEBUG_CHECKS

MPI_Datatype fof_first_type;
MPI_Datatype fof_second_type;

/* Should two particles be linked? */
static inline int should_link(const IntPosType ax[3], const IntPosType bx[3],
                              double int_to_pos_fac, double linking_length_2) {

    const double r_2 = int_to_phys_dist2(ax, bx, int_to_pos_fac);
    return r_2 <= linking_length_2;
}

/* Does this global offset correspond to a local particle? */
static inline int is_local(long int global_offset, long int rank_offset,
                           long int num_localpart) {

    return (global_offset >= rank_offset && global_offset < rank_offset + num_localpart);
}

/* Find the root of the set of a given particle */
long int find_root(struct fof_part_data *fof_parts, long int local_offset) {

    struct fof_part_data *part = &fof_parts[local_offset];

#ifdef DEBUG_CHECKS
    assert(part->root >= 0);
#endif

    if (local_offset == part->root)
        return local_offset;
    part->root = find_root(fof_parts, part->root);
    return part->root;
}

/* Perform the union operation on the sets containing two given particles */
void union_roots(struct fof_part_data *fof_parts, long int local_offset_a, long int local_offset_b) {

    struct fof_part_data *a = &fof_parts[local_offset_a];
    struct fof_part_data *b = &fof_parts[local_offset_b];

#ifdef DEBUG_CHECKS
    assert(a->root >= 0);
    assert(b->root >= 0);
#endif

    long int root_a = find_root(fof_parts, local_offset_a);
    long int root_b = find_root(fof_parts, local_offset_b);

    if (root_a != root_b) {
        if (fof_parts[root_a].global_offset <= fof_parts[root_b].global_offset) {
            fof_parts[root_a].root = root_b;
        } else {
            fof_parts[root_b].root = root_a;
        }
    }
}

/* A safe global version of find_root that terminates when the root is nonlocal.
 * This should be run only when the roots are global. By contrast, the regular
 * find_root and union_roots should be run when the roots are local. */
long int find_root_global(struct fof_part_data *fof_parts, struct fof_part_data *part,
                          long int rank_offset, long int num_localpart) {

    if (!is_local(part->root, rank_offset, num_localpart))
        return part->root;
    else if (fof_parts[part->root - rank_offset].root == part->root)
        return part->root;
    part->root = find_root_global(fof_parts, &fof_parts[part->root - rank_offset], rank_offset, num_localpart);
    return part->root;
}

/* Link particles within two cells */
long int link_cells(struct fof_part_data *fof_parts, struct particle *parts,
                    CellOffsetIntType *cl, long int local_offset1,
                    long int local_offset2, long int local_count1,
                    long int local_count2, double int_to_pos_fac,
                    double linking_length_2) {

    if (local_count1 < 1 || local_count2 < 1 || (local_count1 + local_count2 < 2)) return 0;

    long int links = 0;

    for (long int a = 0; a < local_count1; a++) {
        const CellOffsetIntType index_a = cl[local_offset1 + a];
        const IntPosType *xa = parts[index_a].x;

        /* Don't link neutrinos */
        if (compare_particle_type(&parts[index_a], neutrino_type, 0)) continue;

        /* If we are linking within the same cell, only check all pairs once */
        long int max_check = (local_offset1 == local_offset2) ? a : local_count2;

        for (long int b = 0; b < max_check; b++) {
            const CellOffsetIntType index_b = cl[local_offset2 + b];
            const IntPosType *xb = parts[index_b].x;

            /* Don't link neutrinos */
            if (compare_particle_type(&parts[index_b], neutrino_type, 0)) continue;

            if (should_link(xa, xb, int_to_pos_fac, linking_length_2)) {
                links++;
                union_roots(fof_parts, index_a, index_b);
            }
        }
    }

    return links;
}

/* Receive FOF particle data in the first exchange */
void first_receive_fof_parts(struct fof_part_data *dest, struct particle *parts_dest,
                             int *num_received, int from_rank, long int from_rank_offset,
                             long long int current_partnum, long long int max_partnum) {

    /* Prepare to receive particles */
    MPI_Status status;
    MPI_Probe(from_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, fof_first_type, num_received);

    /* Check that we have enough memory */
    if (current_partnum + *num_received > max_partnum) {
        printf("Not enough memory to exchange FOF data (%lld < %lld).\n", max_partnum, current_partnum + *num_received);
        exit(1);
    }

    /* Allocate memory for receiving particles */
    struct fof_part_first_exchange_data *recv_parts = malloc(*num_received * sizeof(struct fof_part_first_exchange_data));

    /* Receive the particle data */
    MPI_Recv(recv_parts, *num_received, fof_first_type, from_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Copy over the data */
    for (long int i = 0; i < *num_received; i++) {
        parts_dest[i].x[0] = recv_parts[i].x[0];
        parts_dest[i].x[1] = recv_parts[i].x[1];
        parts_dest[i].x[2] = recv_parts[i].x[2];
        dest[i].global_offset = recv_parts[i].local_offset + from_rank_offset;
        // roots have not been set yet and are not exchanged
    }

    /* Free the received data */
    free(recv_parts);
}


/* Receive FOF particle data in the second exchange (more data is exchanged here) */
void second_receive_fof_parts(struct fof_part_data *dest, struct particle *parts_dest,
                              int *num_received, int from_rank, long long int current_partnum,
                              long long int max_partnum) {

    /* Prepare to receive particles */
    MPI_Status status;
    MPI_Probe(from_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, fof_second_type, num_received);

    /* Check that we have enough memory */
    if (current_partnum + *num_received > max_partnum) {
        printf("Not enough memory to exchange FOF data (%lld < %lld).\n", max_partnum, current_partnum + *num_received);
        exit(1);
    }

    /* Allocate memory for receiving particles */
    struct fof_part_second_exchange_data *recv_parts = malloc(*num_received * sizeof(struct fof_part_second_exchange_data));

    /* Receive the particle data */
    MPI_Recv(recv_parts, *num_received, fof_second_type, from_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Copy over the data */
    for (long int i = 0; i < *num_received; i++) {
        parts_dest[i].x[0] = recv_parts[i].x[0];
        parts_dest[i].x[1] = recv_parts[i].x[1];
        parts_dest[i].x[2] = recv_parts[i].x[2];
        dest[i].root = recv_parts[i].root;
        dest[i].global_offset = recv_parts[i].global_offset;
    }

    /* Free the received data */
    free(recv_parts);
}

/* Compute the distance from a particle to a given edge:
* type = 0: left edge of the rank
* type = 1: right edge of the domain */
double edge_distance(IntPosType x, double boxlen, double int_to_rank_fac,
                     double int_to_pos_fac, int type) {

    if (type == 0) {
        /* Distance from left edge of rank */
        double rank_to_pos_fac = int_to_pos_fac / int_to_rank_fac;
        double rank_float = x * int_to_rank_fac;
        double dx = rank_float - ((int) rank_float);
        return dx * rank_to_pos_fac;
    } else {
        /* Distance from right edge of domain */
        double pos = x * int_to_pos_fac;
        return boxlen - pos;
    }
}

/* Copy particles within a linking length from an edge. There are two types:
 * type = 0: left edge of the rank
 * type = 1: right edge of the domain */
void copy_edge_parts(struct fof_part_first_exchange_data **dest,
                     struct fof_part_data *fof_parts,
                     struct particle *parts, int *num_copied,
                     long long int num_localpart, int type,
                     double int_to_rank_fac, double int_to_pos_fac,
                     double boxlen, double linking_length) {

    /* Count the number of particles within one linking length from the left edge */
    int count_near_edge = 0;
    for (long int i = 0; i < num_localpart; i++) {
        double dx = edge_distance(parts[i].x[0], boxlen, int_to_rank_fac, int_to_pos_fac, type);
        if (dx < linking_length) {
            count_near_edge++;
        }
    }

    /* Allocate memory for the edge particles */
    *dest = malloc(count_near_edge * sizeof(struct fof_part_first_exchange_data));
    *num_copied = count_near_edge;

    /* Fish out particles within one linking length from the left edge */
    int copy_counter = 0;
    for (long int i = 0; i < num_localpart; i++) {
        double dx = edge_distance(parts[i].x[0], boxlen, int_to_rank_fac, int_to_pos_fac, type);
        if (dx < linking_length) {
            (*dest)[copy_counter].x[0] = parts[i].x[0];
            (*dest)[copy_counter].x[1] = parts[i].x[1];
            (*dest)[copy_counter].x[2] = parts[i].x[2];
            (*dest)[copy_counter].local_offset = i;
            copy_counter++;
        }
    }
}

void print_memory_message(int rank, int MPI_Rank_Count, long int Ng,
                          int num_cells, long int max_partnum_global,
                          long int N_cb) {
    /* Compare memory use of FOF structures with that of the PM grid. */
    const double mem_grid = (Ng * Ng * Ng * sizeof(GridFloatType)) / (1.0e9);
    const double mem_cell_structures = (2 * num_cells * MPI_Rank_Count * sizeof(long int)) / (1.0e9);
    const double mem_cell_list = (max_partnum_global * sizeof(struct fof_cell_list)) / (1.0e9);
    const double mem_particle_list = (max_partnum_global * sizeof(CellOffsetIntType)) / (1.0e9);
    const double mem_fof_parts = (max_partnum_global * sizeof(struct fof_part_data)) / (1.0e9);
    const double mem_roots = (max_partnum_global * sizeof(long int)) / (1.0e9);
    const double mem_sizes_ids = (max_partnum_global * sizeof(long int)) / (1.0e9);
    const double net_presort = (mem_fof_parts + mem_cell_list) - mem_grid;
    const double net_postsort = (mem_fof_parts + mem_cell_structures + mem_particle_list + mem_roots) - mem_grid;
    /* Estimate the number of halos */
    const double halo_num_estimate = 0.005 * N_cb * N_cb * N_cb;
    const double mem_central_parts = (halo_num_estimate * (sizeof(double) + sizeof(long int))) / (1.0e9);
    const double mem_halo_struct = (halo_num_estimate * sizeof(struct fof_halo)) / (1.0e9);
    const double net_final = (mem_fof_parts + mem_sizes_ids + mem_central_parts + mem_halo_struct) - mem_grid;
    message(rank, "\n");
    message(rank, "Estimated memory use of FOF structures.\n");
    message(rank, "FOF particle data (always): %g GB\n", mem_fof_parts);
    message(rank, "Cell list (pre-sort): %g GB\n", mem_cell_list);
    message(rank, "Cell structures (post-sort): %g GB\n", mem_cell_structures);
    message(rank, "Offset list (post-sort): %g GB\n", mem_particle_list);
    message(rank, "Root list  (post-sort): %g GB\n", mem_roots);
    message(rank, "Group sizes & ids  (final): %g GB\n", mem_sizes_ids);
    message(rank, "Central particles (final): %g GB\n", mem_central_parts);
    message(rank, "Halo data (final): %g GB\n", mem_halo_struct);
    message(rank, "Available from PM grid: %g GB\n", mem_grid);
    message(rank, "(Pre-sort) Net use: %g\n", net_presort);
    message(rank, "(Post-sort) Net use: %g\n", net_postsort);
    message(rank, "(Final) Net use: %g\n", net_final);
    message(rank, "\n");
}

int analysis_fof(struct particle *parts, double boxlen, long int N_cb,
                 long int N_nu, long long int Ng, long long int num_localpart,
                 long long int max_partnum, double linking_length,
                 int halo_min_npart, int output_num, double a_scale_factor,
                 const struct units *us, const struct physical_consts *pcs,
                 const struct cosmology *cosmo, struct params *pars,
                 const struct cosmology_tables *ctabs,
                 double dtau_kick, double dtau_drift) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Data type for MPI communication of particles */
    fof_first_type = mpi_fof_part_first_type();
    fof_second_type = mpi_fof_part_second_type();

    /* Communicate the particle counts across all ranks */
    long long int *parts_per_rank = malloc(MPI_Rank_Count * sizeof(long int));
    long long int *rank_offsets = malloc(MPI_Rank_Count * sizeof(long int));

    MPI_Allgather(&num_localpart, 1, MPI_LONG_LONG, parts_per_rank,
                  1, MPI_LONG_LONG, MPI_COMM_WORLD);

    rank_offsets[0] = 0;
    for (int i = 1; i < MPI_Rank_Count; i++) {
        rank_offsets[i] = rank_offsets[i - 1] + parts_per_rank[i - 1];
    }

    /* The local rank offset */
    const long int rank_offset = rank_offsets[rank];

    /* Timer */
    struct timepair fof_timer;
    timer_start(rank, &fof_timer);

    /* The initial domain decomposition into spatial cells */
    const CellIntType N_cells = pars->HaloFindCellNumber;
    const CellIntType num_cells = N_cells * N_cells;
    const double int_to_cell_fac = N_cells / pow(2.0, POSITION_BITS);

    /* The conversion factor from integers to physical lengths */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_to_pos_fac = 1.0 / pos_to_int_fac;
    const double linking_length_2 = linking_length * linking_length;

    /* The conversion factor from integers to MPI rank number */
    const long long int max_block_width = Ng / MPI_Rank_Count + ((Ng % MPI_Rank_Count) ? 1 : 0); //rounded up
    const double int_block_width = max_block_width * (boxlen / Ng * pos_to_int_fac);
    const double int_to_rank_fac = 1.0 / int_block_width;

#ifndef WITH_MASSES
    /* Compute the critical density */
    const double H_0 = cosmo->h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit_0 = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double Omega_cb = cosmo->Omega_cdm + cosmo->Omega_b;
    const double part_mass_cb = rho_crit_0 * Omega_cb * pow(boxlen / N_cb, 3);
#endif

    /* Find the maximum number of particles across all ranks */
    long int max_partnum_global;
    MPI_Allreduce(&max_partnum, &max_partnum_global, 1, MPI_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    print_memory_message(rank, MPI_Rank_Count, Ng, num_cells, max_partnum_global, N_cb);

    if ((double) N_cells * N_cells >= pow(2, CELL_INT_BYTES)) {
        printf("The number of cells is large. We should switch to larger ints (TODO).\n");
        exit(1);
    }

    /* The cells must be larger than the linking length */
    if (boxlen / N_cells < 1.01 * linking_length) {
        printf("The cells are smaller than the linking length. Decrease HaloFinding:CellNumber.\n");
        exit(1);
    }

    /* The ranks must cover a physical length larger than the linking length */
    if (boxlen / MPI_Rank_Count < 1.01 * linking_length) {
        printf("The number of ranks is so high that they do not span at least one linking length!\n");
        exit(1);
    }

    if (max_partnum > pow(2, OFFSET_INT_BYTES)) {
        printf("The number of particles is too large for 32-bit integer offsets in the cell list structure.\n");
        exit(1);
    }

    /* Allocate memory for linking FOF particles */
    struct fof_part_data *fof_parts = malloc(max_partnum * sizeof(struct fof_part_data));

    for (long int i = 0; i < num_localpart; i++) {
        fof_parts[i].global_offset = i + rank_offset;
        // root set after exchanging
    }

    timer_stop(rank, &fof_timer, "Copying particle data took ");

    /* The total number of particles to be received */
    int receive_foreign_count = 0;

    /* When running with multiple ranks, we need to communicate edge particles */
    if (MPI_Rank_Count > 1) {

        /* We aim to communicate towards ranks with the lower number, which is
         * usually to the left, except for the last rank. Hence, three cases:
         *
         * 1) Rank 0             receives particles from both left and right
         * 2) Ranks 1 ... N-2    receive particles from right and send to left
         * 3) Rank N-1           sends particles to left and right
         *
         */
        if (rank == 0) {
            int receive_from_right, receive_from_left;

            /* Receive particle data from the right */
            first_receive_fof_parts(fof_parts + num_localpart, parts + num_localpart,
                                    &receive_from_right, rank_right, rank_offsets[rank_right],
                                    num_localpart, max_partnum);

            /* Receive particle data from the left */
            first_receive_fof_parts(fof_parts + num_localpart + receive_from_right,
                                    parts + num_localpart + receive_from_right,
                                    &receive_from_left, rank_left, rank_offsets[rank_left],
                                    num_localpart + receive_from_right, max_partnum);

            receive_foreign_count += receive_from_right;
            receive_foreign_count += receive_from_left;
        } else if (rank < MPI_Rank_Count - 1) {
            struct fof_part_first_exchange_data *edge_parts;
            int count_near_edge;

            /* Fish out particles within one linking length from the left edge */
            copy_edge_parts(&edge_parts, fof_parts, parts, &count_near_edge,
                            num_localpart, /* left */ 0, int_to_rank_fac,
                            int_to_pos_fac, boxlen, linking_length);

            /* Receive particles from the right */
            first_receive_fof_parts(fof_parts + num_localpart, parts + num_localpart,
                                    &receive_foreign_count, rank_right, rank_offsets[rank_right],
                                    num_localpart, max_partnum);

            /* Communicate the edge particles to the left neighbour */
            MPI_Send(edge_parts, count_near_edge, fof_first_type, rank_left, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);
        } else {
            struct fof_part_first_exchange_data *left_edge_parts;
            struct fof_part_first_exchange_data *right_edge_parts;
            int count_near_left_edge;
            int count_near_right_edge;

            /* Fish out particles within one linking length from the left edge */
            copy_edge_parts(&left_edge_parts, fof_parts, parts,
                            &count_near_left_edge, num_localpart, /* left */ 0,
                            int_to_rank_fac, int_to_pos_fac, boxlen,
                            linking_length);

            /* Fish out particles within one linking length from the right edge */
            copy_edge_parts(&right_edge_parts, fof_parts, parts,
                            &count_near_right_edge, num_localpart, /* right */ 1,
                            int_to_rank_fac, int_to_pos_fac, boxlen,
                            linking_length);

            /* Communicate the edge particles to the left neighbour */
            MPI_Send(left_edge_parts, count_near_left_edge, fof_first_type,
                     rank_left, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(left_edge_parts);

            /* Communicate these edge particles to the right neighbour */
            MPI_Send(right_edge_parts, count_near_right_edge, fof_first_type,
                     rank_right, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(right_edge_parts);
        }

        /* Place a barrier to make sure that the memory used for the exchange
         * is freed on all ranks, before allocating the cell structures below */
        MPI_Barrier(MPI_COMM_WORLD);
        timer_stop(rank, &fof_timer, "The first communication took ");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Now create a particle-cell correspondence for sorting */
    struct fof_cell_list *cell_list = malloc((num_localpart + receive_foreign_count) * sizeof(struct fof_cell_list));
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        cell_list[i].cell = which_cell(parts[i].x, int_to_cell_fac, N_cells);
        cell_list[i].offset = i;
    }

    /* Sort particles by cell */
    qsort(cell_list, num_localpart + receive_foreign_count, sizeof(struct fof_cell_list), cellListSort);

#ifdef DEBUG_CHECKS
    /* Check the sort */
    for (long long i = 1; i < num_localpart + receive_foreign_count; i++) {
        assert(cell_list[i].cell >= cell_list[i - 1].cell);
    }
#endif

    timer_stop(rank, &fof_timer, "Sorting particles took ");

    /* Temporarily use the roots as scratch space to copy over the cell list */
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = cell_list[i].offset;
    }

    /* Free the sorted list and create a new list with offsets only */
    free(cell_list);
    CellOffsetIntType *particle_list = malloc((num_localpart + receive_foreign_count) * sizeof(CellOffsetIntType));
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        particle_list[i] = fof_parts[i].root;
    }

    /* Make each particle its own root using the current local offset */
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = i;
    }

    timer_stop(rank, &fof_timer, "Shrinking cell list took ");

    /* Determine the counts and offsets of particles in each cell */
    long int *cell_counts = calloc(num_cells, sizeof(long int));
    long int *cell_offsets = calloc(num_cells, sizeof(long int));

    /* Count particles in cells */
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        CellIntType c = which_cell(parts[i].x, int_to_cell_fac, N_cells);
#ifdef DEBUG_CHECKS
        assert((c >= 0) && (c < num_cells));
#endif
        cell_counts[c]++;
    }

    /* Determine the offsets, using the fact that the particles are sorted */
    cell_offsets[0] = 0;
    for (CellIntType i = 1; i < num_cells; i++) {
        cell_offsets[i] = cell_offsets[i-1] + cell_counts[i-1];
    }

    message(rank, "Linking particles in cells.\n");

    long int total_links = 0;

    /* Now link particles within and between neighbouring cells */
    for (long int j = 0; j < N_cells; j++) {
        for (long int k = 0; k < N_cells; k++) {

            long int offset = cell_offsets[row_major_cell(j, k, N_cells)];
            long int count = cell_counts[row_major_cell(j, k, N_cells)];

            /* Loop over the 9 neighbour cells (including itself) */
            for (int v = -1; v <= 1; v++) {
                for (int w = -1; w <= 1; w++) {
                    long int j1 = j + v;
                    long int k1 = k + w;

                    /* Account for periodic boundary conditions */
                    if (j1 >= N_cells) j1 -= N_cells;
                    if (k1 >= N_cells) k1 -= N_cells;
                    if (j1 < 0) j1 += N_cells;
                    if (k1 < 0) k1 += N_cells;

                    long int offset1 = cell_offsets[row_major_cell(j1, k1, N_cells)];
                    long int count1 = cell_counts[row_major_cell(j1, k1, N_cells)];

                    /* Link cells */
                    total_links += link_cells(fof_parts, parts, particle_list, offset, offset1, count, count1, int_to_pos_fac, linking_length_2);
                }
            }
        }
    }

    timer_stop(rank, &fof_timer, "Linking particles in cells took ");
    message(rank, "Found %ld links within and across cells on rank %d.\n", total_links, rank);

    /* We are done with the cell structures */
    free(cell_counts);
    free(cell_offsets);
    free(particle_list);

    /* Loop over the roots again to collapse the tree */
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = find_root(fof_parts, i);
    }

    /* Compute the global offsets of the roots */
    long int *global_roots = malloc((num_localpart + receive_foreign_count) * sizeof(long int));
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        global_roots[i] = fof_parts[fof_parts[i].root].global_offset;
    }

    /* Now we can replace the roots by the global roots */
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = global_roots[i];
    }

    /* Free the global root array */
    free(global_roots);

    /* The total number of particles to be received in the second exchange */
    int second_receive_count = 0;

    message(rank, "Communicating and linking particles across ranks.\n");

    /* When running with multiple ranks, we need to communicate edge particles */
    if (MPI_Rank_Count > 1) {

        /* Now all ranks (except the last rank) may have particles with a root
         * particle that belongs to a rank with a higher number. Three cases:
         *
         * Rank 0          sends particles to the left and right
         * Rank 1...N-1    receives from the left, then sends to the right
         * Rank N-1        receives from the left and right
         *
         * This will automatically account for structures that link across
         * multiple ranks by dragging particles along when they are linked.
         *
         */
        if (rank == 0) {
            /* Count particles with roots on the left/right ranks */
            int num_left_rooted = 0;
            int num_right_rooted = 0;
            for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
                /* Is the root local to the left rank or right rank? */
                if (is_local(fof_parts[i].root, rank_offsets[rank_left], parts_per_rank[rank_left])) {
                    num_left_rooted++;
                } else if (is_local(fof_parts[i].root, rank_offsets[rank_right], parts_per_rank[rank_right])) {
                    num_right_rooted++;
                }
            }

            struct fof_part_second_exchange_data *left_root_parts = malloc(num_left_rooted * sizeof(struct fof_part_second_exchange_data));
            struct fof_part_second_exchange_data *right_root_parts = malloc(num_right_rooted * sizeof(struct fof_part_second_exchange_data));

            /* Now, fish out all particles with foreign roots */
            long int copy_counter_left = 0;
            long int copy_counter_right = 0;
            for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
                /* Is the root local to the left rank or right rank? */
                if (is_local(fof_parts[i].root, rank_offsets[rank_left], parts_per_rank[rank_left])) {
                    left_root_parts[copy_counter_left].x[0] = parts[i].x[0];
                    left_root_parts[copy_counter_left].x[1] = parts[i].x[1];
                    left_root_parts[copy_counter_left].x[2] = parts[i].x[2];
                    left_root_parts[copy_counter_left].root = fof_parts[i].root;
                    left_root_parts[copy_counter_left].global_offset = fof_parts[i].global_offset;
                    copy_counter_left++;
                } else if (is_local(fof_parts[i].root, rank_offsets[rank_right], parts_per_rank[rank_right])) {
                    right_root_parts[copy_counter_right].x[0] = parts[i].x[0];
                    right_root_parts[copy_counter_right].x[1] = parts[i].x[1];
                    right_root_parts[copy_counter_right].x[2] = parts[i].x[2];
                    right_root_parts[copy_counter_right].root = fof_parts[i].root;
                    right_root_parts[copy_counter_right].global_offset = fof_parts[i].global_offset;
                    copy_counter_right++;
                }
            }

            /* Disable the remaining copies of the foreign rooted particles on
             * this rank that now live on another rank */
            for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
                if (!is_local(fof_parts[i].root, rank_offset, num_localpart)) {
                    fof_parts[i].root = -1;
                }
            }

            /* Communicate the particles to the left neighbour */
            MPI_Send(left_root_parts, num_left_rooted, fof_second_type, rank_left, 0, MPI_COMM_WORLD);

            /* Communicate the particles to the right neighbour */
            MPI_Send(right_root_parts, num_right_rooted, fof_second_type, rank_right, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(left_root_parts);
            free(right_root_parts);
        } else if (rank < MPI_Rank_Count - 1) {
            /* Receive particles from the left */
            second_receive_fof_parts(fof_parts + num_localpart + receive_foreign_count,
                                     parts + num_localpart + receive_foreign_count,
                                     &second_receive_count, rank_left,
                                     num_localpart + receive_foreign_count, max_partnum);

#ifdef DEBUG_CHECKS
            /* All received particles should have a local root on this rank */
            for (long int i = num_localpart + receive_foreign_count; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                assert(is_local(fof_parts[i].root, rank_offset, num_localpart));
            }
#endif

            /* Among received particles, check for copies of local particles */
            for (long int i = num_localpart + receive_foreign_count; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                /* Is this a copy of a local particle? */
                if (is_local(fof_parts[i].global_offset, rank_offset, num_localpart)) {
                    /* Attach the trees */
                    long int local_copy = fof_parts[i].global_offset - rank_offset;
                    long int global_root_a = fof_parts[i].root;
                    long int global_root_b = fof_parts[local_copy].root;

#ifdef DEBUG_CHECKS
                    assert(is_local(global_root_a, rank_offset, num_localpart));
                    assert((global_root_a < global_root_b) || is_local(global_root_b, rank_offset, num_localpart));
#endif

                    if (global_root_a < global_root_b) {
                        fof_parts[global_root_a - rank_offset].root = global_root_b;
                    } else if (global_root_b < global_root_a) {
                        fof_parts[global_root_b - rank_offset].root = global_root_a;
                    }

                    /* Now disable the particle */
                    fof_parts[i].root = -1;
                }
            }

            /* Collapse the tree using the global roots */
            for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                /* Skip disabled particles */
                if (fof_parts[i].root == -1) continue;

                fof_parts[i].root = find_root_global(fof_parts, &fof_parts[i], rank_offset, num_localpart);
            }

            /* After linking the received particles, determine the number of
             * particles with foreign roots */
            int num_foreign_rooted = 0;
            for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                /* Skip disabled particles */
                if (fof_parts[i].root == -1) continue;

                if (!is_local(fof_parts[i].root, rank_offset, num_localpart)) {
                    num_foreign_rooted++;
                }
            }

            /* Now, fish out particles with foreign roots */
            long int copy_counter = 0;
            struct fof_part_second_exchange_data *foreign_root_parts = malloc(num_foreign_rooted * sizeof(struct fof_part_second_exchange_data));
            for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                /* Skip disabled particles */
                if (fof_parts[i].root == -1) continue;

                if (!is_local(fof_parts[i].root, rank_offset, num_localpart)) {
                    foreign_root_parts[copy_counter].x[0] = parts[i].x[0];
                    foreign_root_parts[copy_counter].x[1] = parts[i].x[1];
                    foreign_root_parts[copy_counter].x[2] = parts[i].x[2];
                    foreign_root_parts[copy_counter].root = fof_parts[i].root;
                    foreign_root_parts[copy_counter].global_offset = fof_parts[i].global_offset;
                    copy_counter++;
                }
            }

            /* Disable the remaining copies of the foreign rooted particles on
             * this rank that now live on another rank */
            for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                if (!is_local(fof_parts[i].root, rank_offset, num_localpart)) {
                    fof_parts[i].root = -1;
                }
            }

            /* Communicate the foreign particles to the right neighbour */
            if (rank < MPI_Rank_Count - 1) {
                MPI_Send(foreign_root_parts, num_foreign_rooted, fof_second_type, rank_right, 0, MPI_COMM_WORLD);
            }

            /* Release the delivered particles */
            free(foreign_root_parts);
        } else {
            int receive_from_right, receive_from_left;

            /* Receive particles from the right */
            second_receive_fof_parts(fof_parts + num_localpart + receive_foreign_count,
                                     parts + num_localpart + receive_foreign_count,
                                     &receive_from_right, rank_right,
                                     num_localpart + receive_foreign_count, max_partnum);

            /* Receive particles from the left */
            second_receive_fof_parts(fof_parts + num_localpart + receive_foreign_count + receive_from_right,
                                     parts + num_localpart + receive_foreign_count + receive_from_right,
                                     &receive_from_left, rank_left,
                                     num_localpart + receive_foreign_count + receive_from_right, max_partnum);

            second_receive_count = receive_from_left + receive_from_right;

#ifdef DEBUG_CHECKS
            /* All received particles should have a local root on this rank */
            for (long int i = num_localpart + receive_foreign_count; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                assert(is_local(fof_parts[i].root, rank_offset, num_localpart));
            }
#endif

            /* Among received particles, check for copies of local particles */
            for (long int i = num_localpart + receive_foreign_count; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                /* Is this a copy of a local particle? */
                if (is_local(fof_parts[i].global_offset, rank_offset, num_localpart)) {
                    /* Attach the trees */
                    long int local_copy = fof_parts[i].global_offset - rank_offset;
                    long int global_root_a = fof_parts[i].root;
                    long int global_root_b = fof_parts[local_copy].root;

#ifdef DEBUG_CHECKS
                    assert(is_local(global_root_a, rank_offset, num_localpart));
                    assert((global_root_a < global_root_b) || is_local(global_root_b, rank_offset, num_localpart));
#endif

                    if (global_root_a < global_root_b) {
                        fof_parts[global_root_a - rank_offset].root = global_root_b;
                    } else if (global_root_b < global_root_a) {
                        fof_parts[global_root_b - rank_offset].root = global_root_a;
                    }

                    /* Now disable the particle */
                    fof_parts[i].root = -1;
                }
            }

            /* Collapse the tree using the global roots */
            for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
                /* Skip disabled particles */
                if (fof_parts[i].root == -1) continue;

                fof_parts[i].root = find_root_global(fof_parts, &fof_parts[i], rank_offset, num_localpart);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        timer_stop(rank, &fof_timer, "The second communication took ");
        MPI_Barrier(MPI_COMM_WORLD);
    }

#ifdef DEBUG_CHECKS
    for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        /* Check that we have no foreign rooted particles */
        assert(is_local(fof_parts[i].root, rank_offset, num_localpart));

        /* Check that we have no non-disabled copies of local particles */
        assert((fof_parts[i].global_offset == (i + rank_offset)) || !is_local(fof_parts[i].global_offset, rank_offset, num_localpart));
    }
#endif

    /* Now we have no foreign rooted particles. This means that all particles
       that are linked to something are located on the "home rank" of the root
       particle of that group. Now we can attach the trees. */

    /* Turn the global_roots back into local roots. This is safe because no
     *  particle has a foreign root. Additionally, update the local offsets.  */
    for (long long i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue; // don't remove

        fof_parts[i].root -= rank_offset;
    }

    /* Attach received particles to the local tree using the proper local offsets */
    for (long long i = num_localpart; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        union_roots(fof_parts, i, fof_parts[i].root);
    }

    /* Linking is now complete. Proceed with finding group sizes and computing
     * FOF halo properties of sufficiently large groups */

    /* Allocate memory for the group sizes (only local parts can be roots) */
    int *group_sizes = calloc(num_localpart, sizeof(int));

    /* Determine group sizes by counting particles with the same root */
    for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        fof_parts[i].root = find_root(fof_parts, i);
        group_sizes[fof_parts[i].root]++;
    }

    /* Count the number of structures (with sufficient size) via the roots */
    long int num_structures = 0;
    for (long int i = 0; i < num_localpart; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        if (i == fof_parts[i].root && group_sizes[i] >= halo_min_npart) {
            num_structures++;
        }
    }

    /* Communicate the halo counts across all ranks */
    long long int *halos_per_rank = malloc(MPI_Rank_Count * sizeof(long int));
    long long int *halo_rank_offsets = malloc(MPI_Rank_Count * sizeof(long int));

    MPI_Allgather(&num_structures, 1, MPI_LONG_LONG, halos_per_rank, 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);

    halo_rank_offsets[0] = 0;
    for (int i = 1; i < MPI_Rank_Count; i++) {
        halo_rank_offsets[i] = halo_rank_offsets[i - 1] + halos_per_rank[i - 1];
    }

    const long long int total_halo_num = halo_rank_offsets[MPI_Rank_Count - 1] + halos_per_rank[MPI_Rank_Count - 1];

    message(rank, "Found %ld structures in total.\n", total_halo_num);

    /* Allocate memory for the local halo catalogue */
    struct fof_halo *fofs = malloc(num_structures * sizeof(struct fof_halo));
    bzero(fofs, num_structures * sizeof(struct fof_halo));

    /* Assign root particles to halos */
    long int halo_count = 0;
    int *halo_ids = malloc(num_localpart * sizeof(int));
    for (long int i = 0; i < num_localpart; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        if (i == fof_parts[i].root && group_sizes[i] >= halo_min_npart) {
            halo_ids[i] = halo_count + halo_rank_offsets[rank];
            fofs[halo_count].global_id = halo_count + halo_rank_offsets[rank];
            halo_count++;
        } else {
            halo_ids[i] = -1;
        }
    }

    /* We are done with the group sizes */
    free(group_sizes);

    /* For each group, find the particle closest to the centre of the box. */
    long int *centre_particles = calloc(num_structures, sizeof(long int));
    double *centre_distances_2 = calloc(num_structures, sizeof(double));
    for (long int i = 0; i < num_structures; i++) {
        centre_distances_2[i] = 0.75 * boxlen * boxlen; // maximum value
    }

    const IntPosType int_half_boxlen = pow(2.0, POSITION_BITS - 1);
    for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        long int root = fof_parts[i].root;
        long int h = halo_ids[root] - halo_rank_offsets[rank];

        if (h >= 0) {
            /* Compute the distance to the centre */
            const IntPosType dx = parts[i].x[0] - int_half_boxlen;
            const IntPosType dy = parts[i].x[1] - int_half_boxlen;
            const IntPosType dz = parts[i].x[2] - int_half_boxlen;

            /* Enforce boundary conditions and convert to physical lengths */
            const double fx = (dx < -dx) ? dx * int_to_pos_fac : -((-dx) * int_to_pos_fac);
            const double fy = (dy < -dy) ? dy * int_to_pos_fac : -((-dy) * int_to_pos_fac);
            const double fz = (dz < -dz) ? dz * int_to_pos_fac : -((-dz) * int_to_pos_fac);

            const double f2 = fx * fx + fy * fy + fz * fz;

            if (f2 < centre_distances_2[h]) {
                centre_particles[h] = i;
                centre_distances_2[h] = f2;
            }
        }
    }

    free(centre_distances_2);

    /* Accumulate the halo properties */
    for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        long int root = fof_parts[i].root;
        long int h = halo_ids[root] - halo_rank_offsets[rank];

        if (h >= 0) {
#ifdef WITH_MASSES
            double mass = parts[i].m;
#else
            double mass = part_mass_cb;
#endif

            /* Friends-of-friends mass */
            fofs[h].mass_fof += mass;

            /* Compute the offset from the most central particle */
            long int centre_particle = centre_particles[h];
            const IntPosType dx = parts[i].x[0] - parts[centre_particle].x[0];
            const IntPosType dy = parts[i].x[1] - parts[centre_particle].x[1];
            const IntPosType dz = parts[i].x[2] - parts[centre_particle].x[2];

            /* Enforce boundary conditions and convert to physical lengths */
            const double fx = (dx < -dx) ? dx * int_to_pos_fac : -((-dx) * int_to_pos_fac);
            const double fy = (dy < -dy) ? dy * int_to_pos_fac : -((-dy) * int_to_pos_fac);
            const double fz = (dz < -dz) ? dz * int_to_pos_fac : -((-dz) * int_to_pos_fac);

            /* Centre of mass (use relative position for periodic boundary conditions) */
            fofs[h].x_com[0] += fx * mass;
            fofs[h].x_com[1] += fy * mass;
            fofs[h].x_com[2] += fz * mass;
            /* Total particle number */
            fofs[h].npart++;
            /* The home rank of the halo */
            fofs[h].rank = rank;
        }
    }

    /* Divide by the mass for the centre of mass properties */
    for (long int i = 0; i < num_structures; i++) {
        double halo_mass = fofs[i].mass_fof;
        if (halo_mass > 0) {
            fofs[i].x_com[0] /= halo_mass;
            fofs[i].x_com[1] /= halo_mass;
            fofs[i].x_com[2] /= halo_mass;
        }

        /* Add the position of the most central particle to get the absolute CoM */
        long int centre_particle = centre_particles[i];
        fofs[i].x_com[0] += int_to_pos_fac * parts[centre_particle].x[0];
        fofs[i].x_com[1] += int_to_pos_fac * parts[centre_particle].x[1];
        fofs[i].x_com[2] += int_to_pos_fac * parts[centre_particle].x[2];
    }

    /* Determine the maximum distance of FOF particles to the CoM */
    for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        long int root = fof_parts[i].root;
        long int h = halo_ids[root] - halo_rank_offsets[rank];

        if (h >= 0) {
            /* Compute the offset from the centre of mass */
            const IntPosType dx = parts[i].x[0] - fofs[h].x_com[0] * pos_to_int_fac;
            const IntPosType dy = parts[i].x[1] - fofs[h].x_com[1] * pos_to_int_fac;
            const IntPosType dz = parts[i].x[2] - fofs[h].x_com[2] * pos_to_int_fac;

            /* Enforce boundary conditions and convert to physical lengths */
            const double fx = (dx < -dx) ? dx * int_to_pos_fac : -((-dx) * int_to_pos_fac);
            const double fy = (dy < -dy) ? dy * int_to_pos_fac : -((-dy) * int_to_pos_fac);
            const double fz = (dz < -dz) ? dz * int_to_pos_fac : -((-dz) * int_to_pos_fac);

            const double f2 = fx * fx + fy * fy + fz * fz;
            const double R2 = fofs[h].radius_fof * fofs[h].radius_fof;

            if (f2 > R2) {
                fofs[h].radius_fof = sqrt(f2);
            }
        }
    }

    timer_stop(rank, &fof_timer, "Computing halo properties took ");

    /* Run a shrinking sphere algorithm to determine a better halo centre */
    const double rfac = pars->ShrinkingSphereRadiusFactorCoarse;
    const double mfac = pars->ShrinkingSphereMassFraction;
    const int minpart = pars->ShrinkingSphereMinParticleNum;

    /* Allocate memory for the shrinking sphere algorithm */
    double *shrink_masses = calloc(num_structures, sizeof(double));
    double *shrink_radii = calloc(num_structures, sizeof(double));
    int *shrink_npart =  calloc(num_structures, sizeof(int));
    char *shrink_finished =  calloc(num_structures, sizeof(char));
    IntPosType *shrink_com =  calloc(num_structures * 3, sizeof(IntPosType));

    for (long int i = 0; i < num_structures; i++) {
        shrink_masses[i] = fofs[i].mass_fof;
        shrink_radii[i] = fofs[i].radius_fof;
        shrink_npart[i] = fofs[i].npart;
        shrink_finished[i] = 0;
        shrink_com[i * 3 + 0] = fofs[i].x_com[0] * pos_to_int_fac;
        shrink_com[i * 3 + 1] = fofs[i].x_com[1] * pos_to_int_fac;
        shrink_com[i * 3 + 2] = fofs[i].x_com[2] * pos_to_int_fac;
    }

    /* Run the shrinking sphere algorithm */
    long int halos_finished = 0;
    while (halos_finished < num_structures) {
        /* Loop over halos */
        for (long int i = 0; i < num_structures; i++) {
            if (shrink_finished[i]) continue;

            fofs[i].x_com_inner[0] = 0.;
            fofs[i].x_com_inner[1] = 0.;
            fofs[i].x_com_inner[2] = 0.;
            shrink_masses[i] = 0.;
            shrink_npart[i] = 0;
        }

        /* Loop over particles */
        for (long int i = 0; i < num_localpart + receive_foreign_count + second_receive_count; i++) {
            /* Skip disabled particles */
            if (fof_parts[i].root == -1) continue;

            long int root = fof_parts[i].root;
            long int h = halo_ids[root] - halo_rank_offsets[rank];

            if (h >= 0) {
                if (shrink_finished[h]) continue;

#ifdef WITH_MASSES
                double mass = parts[i].m;
#else
                double mass = part_mass_cb;
#endif

                /* Compute the offset from the shrinking sphere centre of mass */
                const IntPosType dx = parts[i].x[0] - shrink_com[h * 3 + 0];
                const IntPosType dy = parts[i].x[1] - shrink_com[h * 3 + 1];
                const IntPosType dz = parts[i].x[2] - shrink_com[h * 3 + 2];

                /* Enforce boundary conditions and convert to physical lengths */
                const double fx = (dx < -dx) ? dx * int_to_pos_fac : -((-dx) * int_to_pos_fac);
                const double fy = (dy < -dy) ? dy * int_to_pos_fac : -((-dy) * int_to_pos_fac);
                const double fz = (dz < -dz) ? dz * int_to_pos_fac : -((-dz) * int_to_pos_fac);

                const double f2 = fx * fx + fy * fy + fz * fz;
                const double R2 = shrink_radii[h] * shrink_radii[h];

                if (f2 <= R2) {
                    fofs[h].x_com_inner[0] += fx * mass;
                    fofs[h].x_com_inner[1] += fy * mass;
                    fofs[h].x_com_inner[2] += fz * mass;
                    shrink_masses[h] += mass;
                    shrink_npart[h]++;
                }
            }
        }

        /* Loop over halos */
        for (long int i = 0; i < num_structures; i++) {
            if (shrink_finished[i]) continue;

            /* Stop if we drop below the particle number or mass thresholds */
            if (shrink_npart[i] < minpart || shrink_masses[i] < fofs[i].mass_fof * mfac) {
                halos_finished++;
                shrink_finished[i] = 1;
            } else {
                /* Otherwise, shrink the radius further */
                shrink_radii[i] *= rfac;
            }

            if (shrink_masses[i] > 0) {
                fofs[i].x_com_inner[0] /= shrink_masses[i];
                fofs[i].x_com_inner[1] /= shrink_masses[i];
                fofs[i].x_com_inner[2] /= shrink_masses[i];
            }

            /* Update the CoM */
            shrink_com[i * 3 + 0] += pos_to_int_fac * fofs[i].x_com_inner[0];
            shrink_com[i * 3 + 1] += pos_to_int_fac * fofs[i].x_com_inner[1];
            shrink_com[i * 3 + 2] += pos_to_int_fac * fofs[i].x_com_inner[2];
            fofs[i].x_com_inner[0] = int_to_pos_fac * shrink_com[i * 3 + 0];
            fofs[i].x_com_inner[1] = int_to_pos_fac * shrink_com[i * 3 + 1];
            fofs[i].x_com_inner[2] = int_to_pos_fac * shrink_com[i * 3 + 2];
        }
    }

    /* Free shrinking sphere data */
    free(shrink_masses);
    free(shrink_com);
    free(shrink_npart);
    free(shrink_radii);
    free(shrink_finished);

    timer_stop(rank, &fof_timer, "Finding shrinking sphere centres took ");

    /* We are done with the FOF particle data and cell structures */
    free(halo_ids);
    free(centre_particles);
    free(fof_parts);

    /* We are done with halo rank offsets and sizes */
    free(halos_per_rank);
    free(halo_rank_offsets);

    /* Export the FOF properties to an HDF5 file */
    exportCatalogue(pars, us, pcs, output_num, a_scale_factor, total_halo_num, num_structures, fofs);

    /* Timer */
    timer_stop(rank, &fof_timer, "Writing FOF halo properties took ");
    message(rank, "\n");

    if (pars->DoSphericalOverdensities) {
        message(rank, "Proceeding with spherical overdensity calculations.\n");

        analysis_so(parts, &fofs, boxlen, N_cb, N_nu, Ng, num_localpart,
                    max_partnum, total_halo_num, num_structures, output_num,
                    a_scale_factor, us, pcs, cosmo, pars, ctabs, dtau_kick,
                    dtau_drift);
    }

    /* Free the remaining memory */
    free(parts_per_rank);
    free(rank_offsets);
    free(fofs);

    return 0;
}
