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

#ifndef PARTICLE_PACKING_H
#define PARTICLE_PACKING_H

#ifdef WITH_PARTICLE_PACKING

#include <stdint.h>
#include <mpi.h>

#define PACK_POSITION_BITS 18
#define PACK_VELOCITY_BITS 24

struct particle_data {
#ifdef WITH_SEEDS    
    uint32_t seed;
#endif    
    uint64_t register_a;
    uint64_t register_b;
};

typedef struct particle_data particle_data;
typedef struct particle particle;

static inline IntPosType unpack_particle_xpos(particle_data *p) {
    const uint64_t mask_x = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS + 2 * PACK_POSITION_BITS - 64);
    IntPosType x = (p->register_a & mask_x) >> (3 * PACK_VELOCITY_BITS + 2 * PACK_POSITION_BITS - 64);
    x <<= (POSITION_BITS - PACK_POSITION_BITS); 
    return x;
}

static inline IntPosType unpack_particle_ypos(particle_data *p) {
    const uint64_t mask_y = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS + PACK_POSITION_BITS - 64);
    IntPosType y = (p->register_a & mask_y) >> (3 * PACK_VELOCITY_BITS + PACK_POSITION_BITS - 64);
    y <<= (POSITION_BITS - PACK_POSITION_BITS); 
    return y;
}

static inline IntPosType unpack_particle_zpos(particle_data *p) {
    const uint64_t mask_z = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS - 64);
    IntPosType z = (p->register_a & mask_z) >> (3 * PACK_VELOCITY_BITS - 64);
    z <<= (POSITION_BITS - PACK_POSITION_BITS); 
    return z;
}

static inline void unpack_particle_position(particle_data *p, IntPosType x[3]) {
    const uint64_t mask_x = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS + 2 * PACK_POSITION_BITS - 64);
    const uint64_t mask_y = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS + PACK_POSITION_BITS - 64);
    const uint64_t mask_z = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS - 64);
    
    x[0] = (p->register_a & mask_x) >> (3 * PACK_VELOCITY_BITS + 2 * PACK_POSITION_BITS - 64);
    x[1] = (p->register_a & mask_y) >> (3 * PACK_VELOCITY_BITS + PACK_POSITION_BITS - 64);
    x[2] = (p->register_a & mask_z) >> (3 * PACK_VELOCITY_BITS - 64);
    x[0] <<= (POSITION_BITS - PACK_POSITION_BITS); 
    x[1] <<= (POSITION_BITS - PACK_POSITION_BITS);
    x[2] <<= (POSITION_BITS - PACK_POSITION_BITS);
}

static inline void pack_particle_position(particle_data *p, IntPosType x[3]) {
    const uint64_t mask_x = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS + 2 * PACK_POSITION_BITS - 64);
    const uint64_t mask_y = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS + PACK_POSITION_BITS - 64);
    const uint64_t mask_z = ((1ull << (PACK_POSITION_BITS)) - 1ull) << (3 * PACK_VELOCITY_BITS - 64);

    p->register_a &= ~mask_x;
    p->register_a &= ~mask_y;
    p->register_a &= ~mask_z;
    p->register_a += (((uint64_t)x[0] >> (POSITION_BITS - PACK_POSITION_BITS)) << (3 * PACK_VELOCITY_BITS + 2 * PACK_POSITION_BITS - 64));
    p->register_a += (((uint64_t)x[1] >> (POSITION_BITS - PACK_POSITION_BITS)) << (3 * PACK_VELOCITY_BITS + PACK_POSITION_BITS - 64));
    p->register_a += (((uint64_t)x[2] >> (POSITION_BITS - PACK_POSITION_BITS)) << (3 * PACK_VELOCITY_BITS - 64));
}

static inline void unpack_particle_velocity(particle_data *p, IntVelType v[3]) {
    const uint64_t mask_vx_a = ((1ull << (64 - 3 * PACK_POSITION_BITS)) - 1ull);
    const uint64_t mask_vx_b = ((1ull << (PACK_VELOCITY_BITS)) - 1ull) << (2 * PACK_VELOCITY_BITS);
    const uint64_t mask_vy = ((1ull << (PACK_VELOCITY_BITS)) - 1ull) << PACK_VELOCITY_BITS;
    const uint64_t mask_vz = ((1ull << (PACK_VELOCITY_BITS)) - 1ull);    
    
    v[0] = ((p->register_a & mask_vx_a) << (PACK_VELOCITY_BITS - (64 - 3 * PACK_POSITION_BITS)))
         + ((p->register_b & mask_vx_b) >> (2 * PACK_VELOCITY_BITS));
    v[1] = (p->register_b & mask_vy) >> PACK_VELOCITY_BITS;
    v[2] = (p->register_b & mask_vz);
    v[0] <<= (VELOCITY_BITS - PACK_VELOCITY_BITS); 
    v[1] <<= (VELOCITY_BITS - PACK_VELOCITY_BITS);
    v[2] <<= (VELOCITY_BITS - PACK_VELOCITY_BITS);
}

static inline void pack_particle_velocity(particle_data *p, IntVelType v[3]) {
    const uint64_t mask_vx_a = ((1ull << (64 - 3 * PACK_POSITION_BITS)) - 1ull);
    const uint64_t mask_vx_b = ((1ull << (PACK_VELOCITY_BITS)) - 1ull) << (2 * PACK_VELOCITY_BITS);
    const uint64_t mask_vy = ((1ull << (PACK_VELOCITY_BITS)) - 1ull) << PACK_VELOCITY_BITS;
    const uint64_t mask_vz = ((1ull << (PACK_VELOCITY_BITS)) - 1ull);    
    
    p->register_a &= ~mask_vx_a;
    p->register_b &= ~mask_vx_b;
    p->register_b &= ~mask_vy;
    p->register_b &= ~mask_vz;
    p->register_a += (((uint64_t)v[0] >> (VELOCITY_BITS - PACK_VELOCITY_BITS)) >> (PACK_VELOCITY_BITS - (64 - 3 * PACK_POSITION_BITS)));
    p->register_b += (((uint64_t)v[0] >> (VELOCITY_BITS - PACK_VELOCITY_BITS)) & (((1ull << (PACK_VELOCITY_BITS - (64 - 3 * PACK_POSITION_BITS)))) - 1ull)) << (2 * PACK_VELOCITY_BITS);
    p->register_b += (((uint64_t)v[1] >> (VELOCITY_BITS - PACK_VELOCITY_BITS)) << PACK_VELOCITY_BITS);
    p->register_b += (((uint64_t)v[2] >> (VELOCITY_BITS - PACK_VELOCITY_BITS)));
}

static inline MPI_Datatype mpi_particle_type() {

    /* Construct an MPI data type from the constituent fields */
    MPI_Datatype particle_type;
    MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT64_T, MPI_UINT64_T};
    int lengths[3];
    MPI_Aint displacements[3];
    MPI_Aint base_address;
    particle_data temp;
    MPI_Get_address(&temp, &base_address);

#ifdef WITH_SEEDS
    /* Seed */
    lengths[0] = 1;
    MPI_Get_address(&temp.seed, &displacements[0]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
#else
    lengths[0] = 0;
    displacements[0] = 0;
#endif

    /* Register A */
    lengths[1] = 1;
    MPI_Get_address(&temp.register_a, &displacements[1]);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);

    /* Register B */
    lengths[2] = 1;
    MPI_Get_address(&temp.register_b, &displacements[2]);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);

    /* Create the datatype */
    MPI_Type_create_struct(3, lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    return particle_type;
}

#endif
#endif
