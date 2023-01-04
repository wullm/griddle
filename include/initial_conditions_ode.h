/*******************************************************************************
 * This file is part of Sedulus.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
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

/* Fluid equations ODE solver, ported over from Zwindstroom (ascl: 2202.003) */

#ifndef FLUID_EQUATIONS_H
#define FLUID_EQUATIONS_H

#include "units.h"
#include "cosmology.h"

struct growth_factors {
    /* Wavenumber (input) */
    double k;
    /* Density transfer functions at a_start (input) */
    double delta_c;
    double delta_b;
    double *delta_n;
    /* Growth rates at a_start: g = f = dlog(D)/dlog(a) (input)*/
    double gc;
    double gb;
    double *gn;
    /* Relative growth factors between a_start and a_final (output) */
    double Dc;
    double Db;
    double *Dn;
};

void integrate_fluid_equations(struct cosmology *c, struct units *us,
                               struct physical_consts *pcs,
                               struct cosmology_tables *tab,
                               struct growth_factors *gfac,
                               double a_start, double a_final);

void prepare_fluid_integrator(struct cosmology *c, struct units *us,
                              struct physical_consts *pcs,
                              struct cosmology_tables *tab, double tol,
                              double hstart);
void free_fluid_integrator();

#endif
