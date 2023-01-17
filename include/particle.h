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
// #define WITH_ACCELERATIONS
// #define WITH_MASSES
// #define WITH_PARTICLE_IDS
// #define WITH_PARTTYPE
// #define WITH_SEEDS

#ifdef SINGLE_PRECISION_IDS
#define PID_BITS 32
#define MPI_PID_TYPE MPI_UINT32_T
typedef uint32_t IntIDType;
#else
#define PID_BITS 64
#define MPI_PID_TYPE MPI_UINT64_T
typedef uint64_t IntIDType;
#endif

#if defined(HALF_PRECISION_POSITIONS)
#define POSITION_BITS 16
#define MPI_INTPOS_TYPE MPI_UINT16_T
typedef uint16_t IntPosType;
#elif defined(SINGLE_PRECISION_POSITIONS)
#define POSITION_BITS 32
#define MPI_INTPOS_TYPE MPI_UINT32_T
typedef uint32_t IntPosType;
#else
#define POSITION_BITS 64
#define MPI_INTPOS_TYPE MPI_UINT64_T
typedef uint64_t IntPosType;
#endif

#if defined(HALF_PRECISION_VELOCITIES)
#define VELOCITY_BITS 16
#define MPI_FLOATVEL_TYPE MPI_FLOAT
#define MPI_INTVEL_TYPE MPI_UINT16_T
typedef float FloatVelType;
typedef uint16_t IntVelType;
#elif defined(SINGLE_PRECISION_VELOCITIES)
#define VELOCITY_BITS 32
#define MPI_FLOATVEL_TYPE MPI_FLOAT
#define MPI_INTVEL_TYPE MPI_UINT32_T
typedef float FloatVelType;
typedef uint32_t IntVelType;
#else
#define VELOCITY_BITS 64
#define MPI_FLOATVEL_TYPE MPI_DOUBLE
#define MPI_INTVEL_TYPE MPI_UINT64_T
typedef double FloatVelType;
typedef uint64_t IntVelType;
#endif

struct particle {

#ifdef WITH_PARTICLE_IDS
    /* Particle ID */
    IntIDType id;
#endif

    /* Position, velocity */
    IntPosType x[3];
    IntVelType v[3];

#ifdef WITH_ACCELERATIONS
    /* Accelerations */
    float a[3];
#endif

#ifdef WITH_MASSES
    /* Particle mass */
    float m;
#endif

#ifdef WITH_PARTTYPE
    /* Neutrino delta-f weight */
    float w;

    /* The particle type */
    uint16_t type;
#endif
};

/* Use memory-efficient packing of particle data */
// #define WITH_PARTICLE_PACKING
#ifdef WITH_PARTICLE_PACKING
#include "particle_packing.h"
#else

typedef struct particle particle_data;
// typedef struct particle particle;

static inline IntPosType unpack_particle_xpos(particle_data *p) {
    return p->x[0];
}

static inline IntPosType unpack_particle_ypos(particle_data *p) {
    return p->x[1];
}

static inline IntPosType unpack_particle_zpos(particle_data *p) {
    return p->x[2];
}

static inline void unpack_particle_position(particle_data *p, IntPosType x[3]) {
    x[0] = p->x[0];
    x[1] = p->x[1];
    x[2] = p->x[2];
}

static inline void pack_particle_position(particle_data *p, IntPosType x[3]) {
    p->x[0] = x[0];
    p->x[1] = x[1];
    p->x[2] = x[2];
}

static inline void unpack_particle_velocity(particle_data *p, IntVelType v[3]) {
    v[0] = p->x[0];
    v[1] = p->x[1];
    v[2] = p->x[2];
}

static inline void pack_particle_velocity(particle_data *p, IntVelType v[3]) {
    p->v[0] = v[0];
    p->v[1] = v[1];
    p->v[2] = v[2];
}

static inline MPI_Datatype mpi_particle_type() {
 
    /* Construct an MPI data type from the constituent fields */
    MPI_Datatype particle_type;
    MPI_Datatype types[7] = {MPI_PID_TYPE, MPI_INTPOS_TYPE, MPI_INTVEL_TYPE,
                             MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_UINT16_T};
    int lengths[7];
    MPI_Aint displacements[7];
    MPI_Aint base_address;
    particle_data temp;
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
    /* Neutrino delta-f weight */
    lengths[5] = 1;
    MPI_Get_address(&temp.w, &displacements[5]);
    displacements[5] = MPI_Aint_diff(displacements[5], base_address);

    /* Particle type */
    lengths[6] = 1;
    MPI_Get_address(&temp.type, &displacements[6]);
    displacements[6] = MPI_Aint_diff(displacements[6], base_address);
#else
    lengths[5] = 0;
    displacements[5] = 0;
    lengths[6] = 0;
    displacements[6] = 0;
#endif

    /* Create the datatype */
    MPI_Type_create_struct(7, lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    return particle_type;
}

#endif

// static inline int particleSort(const void *a, const void *b) {
//     struct particle *pa = (struct particle*) a;
//     struct particle *pb = (struct particle*) b;
//     return pa->exchange_dir >= pb->exchange_dir;
// }

#ifdef WITH_PARTTYPE
static inline int particleTypeSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->type >= pb->type;
}
#endif

#endif
