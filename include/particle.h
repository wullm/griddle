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

#define SINGLE_PRECISION_IDS
#define SINGLE_PRECISION_POSITIONS
#define SINGLE_PRECISION_VELOCITIES
#define WITH_ACCELERATIONS
#define WITH_MASSES

#ifdef SINGLE_PRECISION_IDS
#define PID_BITS 32
typedef uint32_t IntIDType;
#else
#define PID_BITS 64
typedef uint64_t IntIDType;
#endif

#ifdef SINGLE_PRECISION_POSITIONS
#define POSITION_BITS 32
typedef uint32_t IntPosType;
#else
#define POSITION_BITS 64
typedef uint64_t IntPosType;
#endif

#ifdef SINGLE_PRECISION_VELOCITIES
#define VELOCITY_BITS 32
typedef float FloatVelType;
#else
#define VELOCITY_BITS 64
typedef double FloatVelType;
#endif

struct particle {
    /* Basic particle data */
    IntIDType id;

    /* Position, velocity */
    IntPosType x[3];
    FloatVelType v[3];

#ifdef WITH_MASSES
    /* Particle mass */
    float m;
#endif

#ifdef WITH_ACCELERATIONS
    /* Accelerations */
    float a[3];
#endif

#ifdef WITH_PARTTYPE
    /* Neutrino delta-f weight */
    float w;

    /* The particle type */
    uint16_t type;
#endif

    /* Used for communications (only 2 bits used) */
    int16_t exchange_dir;
};

static inline int particleSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->exchange_dir >= pb->exchange_dir;
}

#ifdef WITH_PARTTYPE
static inline int particleTypeSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->type >= pb->type;
}
#endif

#endif
