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

#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include "units.h"

struct cosmology {
    double h;
    double n_s;
    double A_s;
    double k_pivot;

    double Omega_b;
    double Omega_cdm;
    double Omega_k;
    double N_ur;
    int N_nu;
    double *M_nu;
    double *deg_nu;
    double *c_s_nu;
    double T_nu_0;
    double T_CMB_0;
    double w0;
    double wa;
};

int readCosmology(struct cosmology *cosmo, const char *fname);
int cleanCosmology(struct cosmology *cosmo);
int print_cosmology_information(int rank, struct cosmology *cosmo);
double primordialPower(const double k, const struct cosmology *cosmo);

struct cosmology_tables {
    double *avec;
    double *Avec;
    double *Bvec;
    double *Hvec;
    double *Omega_nu;
    double *f_nu_nr;
    double *f_nu_nr_tot;
    double *kick_factors;
    double *drift_factors;
    int size;
};

void integrate_cosmology_tables(struct cosmology *c, struct units *us,
                                struct physical_consts *pcs,
                                struct cosmology_tables *tab, double a_start,
                                double a_final, int size);
void free_cosmology_tables(struct cosmology_tables *tab);

double get_H_of_a(struct cosmology_tables *tab, double a);
double get_f_nu_nr_tot_of_a(struct cosmology_tables *tab, double a);


#endif
