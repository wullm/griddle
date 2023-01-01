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

/* On-the-fly friends-of-friends halo finder */

#ifndef ANALYSIS_FOF_H
#define ANALYSIS_FOF_H

#include "particle.h"
#include "units.h"
#include "params.h"
#include "cosmology.h"

struct fof_cell_list {
    long int offset;
    long int cell;
};

static inline long int row_major_cell(long int i, long int j, long int k, long int N_cells) {
    return i * N_cells * N_cells + j * N_cells + k;
}

/* Determine the cell containing a given particle */
static inline long int which_cell(IntPosType x[3], double int_to_cell_fac, long int N_cells) {
    return row_major_cell((long int) (int_to_cell_fac * x[0]),
                          (long int) (int_to_cell_fac * x[1]),
                          (long int) (int_to_cell_fac * x[2]), N_cells);
}

/* Order particles by their spatial cell index */
static inline int cellListSort(const void *a, const void *b) {
    struct fof_cell_list *ca = (struct fof_cell_list*) a;
    struct fof_cell_list *cb = (struct fof_cell_list*) b;

    return ca->cell >= cb->cell;
}

/* Compute the squared physical distance between two integer positions */
static inline double int_to_phys_dist2(const IntPosType ax[3],
                                       const IntPosType bx[3],
                                       double int_to_pos_fac) {

    /* Vector distance */
    const IntPosType dx = bx[0] - ax[0];
    const IntPosType dy = bx[1] - ax[1];
    const IntPosType dz = bx[2] - ax[2];

    /* Enforce boundary conditions */
    const IntPosType tx = (dx < -dx) ? dx : -dx;
    const IntPosType ty = (dy < -dy) ? dy : -dy;
    const IntPosType tz = (dz < -dz) ? dz : -dz;

    /* Convert to physical lengths */
    const double fx = tx * int_to_pos_fac;
    const double fy = ty * int_to_pos_fac;
    const double fz = tz * int_to_pos_fac;

    return fx * fx + fy * fy + fz * fz;
}

/* If you add/change/remove fields to fof_part_data or fof_halo,
 * update the corresponding MPI data types below. */

struct fof_part_data {
    /* Integer positons of the particle */
    IntPosType x[3];
    /* The offset (local or global) of the root of the linked particle tree */
    long int root;
    /* The global offset of the corresponding particle in parts */
    long int global_offset;
    /* The local offset of the fof_part, which is not necessarily related to the global_offset */
    long int local_offset;
};

struct fof_halo {
    /* Global ID of the halo */
    long int global_id;
    /* Centre of mass of the FOF particles */
    double x_com[3];
    /* Total mass of the FOF particles */
    double mass_fof;
    /* Maximum distance between FOF particles and the CoM */
    double radius_fof;
    /* Number of linked FOF particles */
    int npart;
    /* Home rank of the halo */
    int rank;
};

int analysis_fof(struct particle *parts, double boxlen, long int Np,
                 long long int Ng, long long int num_localpart,
                 long long int max_partnum, double linking_length,
                 int halo_min_npart, int output_num, double a_scale_factor,
                 const struct units *us, const struct physical_consts *pcs,
                 const struct cosmology *cosmo, struct params *pars,
                 const struct cosmology_tables *ctabs);

/* The MPI data type of the FOF particle data */
static inline MPI_Datatype mpi_fof_part_type() {

    /* Construct an MPI data type from the constituent fields */
    MPI_Datatype particle_type;
    MPI_Datatype types[5] = {MPI_INTPOS_TYPE, MPI_LONG, MPI_LONG,
                             MPI_LONG};
    int lengths[5];
    MPI_Aint displacements[5];
    MPI_Aint base_address;
    struct fof_part_data temp;
    MPI_Get_address(&temp, &base_address);

    int counter = 0;

    /* Position */
    lengths[counter] = 3;
    MPI_Get_address(&temp.x[0], &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Root */
    lengths[counter] = 1;
    MPI_Get_address(&temp.root, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Global offset */
    lengths[counter] = 1;
    MPI_Get_address(&temp.global_offset, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Local offset */
    lengths[counter] = 1;
    MPI_Get_address(&temp.local_offset, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Create the datatype */
    MPI_Type_create_struct(counter, lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    return particle_type;
}

/* The MPI data type of the FOF halo data */
static inline MPI_Datatype mpi_fof_halo_type() {

    /* Construct an MPI data type from the constituent fields */
    MPI_Datatype particle_type;
    MPI_Datatype types[6] = {MPI_LONG, MPI_DOUBLE, MPI_DOUBLE,
                             MPI_DOUBLE, MPI_INT, MPI_INT};
    int lengths[6];
    MPI_Aint displacements[6];
    MPI_Aint base_address;
    struct fof_halo temp;
    MPI_Get_address(&temp, &base_address);

    int counter = 0;

    /* ID */
    lengths[counter] = 1;
    MPI_Get_address(&temp.global_id, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Position */
    lengths[counter] = 3;
    MPI_Get_address(&temp.x_com, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Mass */
    lengths[counter] = 1;
    MPI_Get_address(&temp.mass_fof, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Maximum distance to CoM */
    lengths[counter] = 1;
    MPI_Get_address(&temp.radius_fof, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Particle number */
    lengths[counter] = 1;
    MPI_Get_address(&temp.npart, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* MPI Home Rank */
    lengths[counter] = 1;
    MPI_Get_address(&temp.rank, &displacements[counter]);
    displacements[counter] = MPI_Aint_diff(displacements[counter], base_address);
    counter++;

    /* Create the datatype */
    MPI_Type_create_struct(counter, lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    return particle_type;
}

#endif
