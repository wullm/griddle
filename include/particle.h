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

#ifndef PARTICLE_H
#define PARTICLE_H

#include <stdint.h>
#include <mpi.h>

#define SINGLE_PRECISION_IDS
#define SINGLE_PRECISION_POSITIONS
#define SINGLE_PRECISION_VELOCITIES
#define WITH_ACCELERATIONS
#define WITH_MASSES
#define WITH_PARTICLE_IDS
#define WITH_PARTICLE_SEEDS
#define WITH_PARTTYPE

#define WITH_NEUTRINOS

#ifdef SINGLE_PRECISION_IDS
#define PID_BITS 32
#define MPI_PID_TYPE MPI_UINT32_T
typedef uint32_t IntIDType;
#else
#define PID_BITS 64
#define MPI_PID_TYPE MPI_UINT64_T
typedef uint64_t IntIDType;
#endif

#ifdef SINGLE_PRECISION_POSITIONS
#define POSITION_BITS 32
#define MPI_INTPOS_TYPE MPI_UINT32_T
typedef uint32_t IntPosType;
#else
#define POSITION_BITS 64
#define MPI_INTPOS_TYPE MPI_UINT64_T
typedef uint64_t IntPosType;
#endif

#ifdef SINGLE_PRECISION_VELOCITIES
#define VELOCITY_BITS 32
#define MPI_FLOATVEL_TYPE MPI_FLOAT
typedef float FloatVelType;
#else
#define VELOCITY_BITS 64
#define MPI_FLOATVEL_TYPE MPI_DOUBLE
typedef double FloatVelType;
#endif

struct particle {

#ifdef WITH_PARTICLE_IDS
    /* Particle ID */
    IntIDType id;
#endif

#ifdef WITH_PARTICLE_SEEDS
    /* Particle seed */
    uint32_t seed;
#endif

    /* Position, velocity */
    IntPosType x[3];
    FloatVelType v[3];

#ifdef WITH_ACCELERATIONS
    /* Accelerations */
    float a[3];
#endif

#ifdef WITH_MASSES
    /* Particle mass */
    float m;
#endif

#ifdef WITH_PARTTYPE
    /* The particle type */
    uint16_t type;
#endif
};

enum particle_type {
    cdm_type,
    neutrino_type
};

static inline int match_particle_type(const struct particle *p,
                                      enum particle_type type,
                                      int default_if_undefined) {
#if defined(WITH_PARTTYPE)
    if (p->type == type) {
        return 1;
    } else {
        return 0;
    }
#elif defined(WITH_PARTICLE_SEEDS)
    if (p->seed == 0 && type == cdm_type) {
        return 1;
    } else if (p->seed > 0 && type == neutrino_type) {
        return 1;
    } else {
        return 0;
    }
#else
    return default_if_undefined;
#endif
}

static inline MPI_Datatype mpi_particle_type() {

    /* Construct an MPI data type from the constituent fields */
    MPI_Datatype particle_type;
    MPI_Datatype types[7] = {MPI_PID_TYPE, MPI_UINT32_T, MPI_INTPOS_TYPE,
                             MPI_FLOATVEL_TYPE, MPI_FLOAT, MPI_FLOAT,
                             MPI_UINT16_T};
    int lengths[7];
    MPI_Aint displacements[7];
    MPI_Aint base_address;
    struct particle temp;
    MPI_Get_address(&temp, &base_address);

#ifdef WITH_PARTICLE_IDS
    /* ID */
    lengths[0] = 1;
    MPI_Get_address(&temp.id, &displacements[0]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
#else
    lengths[0] = 0;
    displacements[0] = 0;
#endif

#ifdef WITH_PARTICLE_SEEDS
    /* Seed */
    lengths[0] = 1;
    MPI_Get_address(&temp.seed, &displacements[0]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
#else
    lengths[0] = 0;
    displacements[0] = 0;
#endif

    /* Position */
    lengths[1] = 3;
    MPI_Get_address(&temp.x[0], &displacements[1]);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);

    /* Velocity */
    lengths[2] = 3;
    MPI_Get_address(&temp.v[0], &displacements[2]);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);

#ifdef WITH_ACCELERATIONS
    /* Acceleration */
    lengths[3] = 3;
    MPI_Get_address(&temp.a[0], &displacements[3]);
    displacements[3] = MPI_Aint_diff(displacements[3], base_address);
#else
    lengths[3] = 0;
    displacements[3] = 0;
#endif

#ifdef WITH_MASSES
    /* Mass */
    lengths[4] = 1;
    MPI_Get_address(&temp.m, &displacements[4]);
    displacements[4] = MPI_Aint_diff(displacements[4], base_address);
#else
    lengths[4] = 0;
    displacements[4] = 0;
#endif

#ifdef WITH_PARTTYPE
    /* Particle type */
    lengths[5] = 1;
    MPI_Get_address(&temp.type, &displacements[6]);
    displacements[5] = MPI_Aint_diff(displacements[6], base_address);
#else
    lengths[5] = 0;
    displacements[5] = 0;
#endif

    /* Create the datatype */
    MPI_Type_create_struct(7, lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    return particle_type;
}

// static inline int particleSort(const void *a, const void *b) {
//     struct particle *pa = (struct particle*) a;
//     struct particle *pb = (struct particle*) b;
//     return pa->exchange_dir >= pb->exchange_dir;
// }

#if defined(WITH_PARTTYPE)
static inline int particleTypeSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->type >= pb->type;
}
#elif defined(WITH_PARTICLE_SEEDS)
static inline int particleTypeSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->seed >= pb->seed;
}
#endif

#endif
