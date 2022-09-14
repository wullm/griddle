/*******************************************************************************
 * This file is part of Nyver.
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
#include <string.h>
#include <math.h>

#include "../include/cosmology.h"

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

int readCosmology(struct cosmology *cosmo, const char *fname) {
    /* Read the basic cosmological parameters */
    cosmo->h = ini_getd("Cosmology", "h", 0.70, fname);
    cosmo->Omega_b = ini_getd("Cosmology", "Omega_b", 0.05, fname);
    cosmo->Omega_cdm = ini_getd("Cosmology", "Omega_cdm", 0.25, fname);
    cosmo->Omega_k = ini_getd("Cosmology", "Omega_k", 0.0, fname);

    /* Power spectrum */
    cosmo->n_s = ini_getd("Cosmology", "n_s", 0.97, fname);
    cosmo->A_s = ini_getd("Cosmology", "A_s", 2.215e-9, fname);
    cosmo->k_pivot = ini_getd("Cosmology", "k_pivot", 0.05, fname);

    /* Neutrinos, radiation, and dark energy */
    cosmo->N_ur = ini_getd("Cosmology", "N_ur", 3.046, fname);
    cosmo->N_nu = ini_getl("Cosmology", "N_nu", 0, fname); // beware int
    cosmo->T_nu_0 = ini_getd("Cosmology", "T_nu_0", 1.951757805, fname);
    cosmo->T_CMB_0 = ini_getd("Cosmology", "T_CMB_0", 2.7255, fname);
    cosmo->w0 = ini_getd("Cosmology", "w0", -1.0, fname);
    cosmo->wa = ini_getd("Cosmology", "wa", 0.0, fname);

    /* Handle separate comma-separted lists of neutrino properties */
    if (cosmo->N_nu > 0) {
        /* Allocate arrays for neutrino properties */
        cosmo->M_nu = malloc(cosmo->N_nu * sizeof(double));
        cosmo->deg_nu = malloc(cosmo->N_nu * sizeof(double));

        /* Prepare reading the strings of comma-separated lists */
        int charlen = 100;
        char M_nu_string[charlen]; // masses
        char deg_nu_string[charlen]; // degeneracies
        ini_gets("Cosmology", "M_nu", "", M_nu_string, charlen, fname);
        ini_gets("Cosmology", "deg_nu", "", deg_nu_string, charlen, fname);

        /* Check that the required properties are there */
        if (strlen(M_nu_string) <= 0) {
            printf("Error: specifying the neutrino masses is mandatory if N_nu > 0.\n");
            exit(1);
        } else if (strlen(deg_nu_string) <= 0) {
            printf("Error: specifying the neutrino degeneracies is mandatory if N_nu > 0.\n");
            exit(1);
        }

        /* Permissible delimiters */
        char delimiters[] = " ,\t\n";

        /* Parse masses */
        char *m_token = strtok(M_nu_string, delimiters);
        int m_species = 0;
        while (m_token != NULL) {
            sscanf(m_token, "%lf", &cosmo->M_nu[m_species++]);
            m_token = strtok (NULL, delimiters);
        }

        /* Parse degeneracies */
        char *d_token = strtok(deg_nu_string, delimiters);
        int d_species = 0;
        while (d_token != NULL) {
            sscanf(d_token, "%lf", &cosmo->deg_nu[d_species++]);
            d_token = strtok (NULL, delimiters);
        }

        /* Check that all degeneracies are there */
        if (m_species != cosmo->N_nu || d_species != cosmo->N_nu) {
            printf("Error: number of neutrino masses or degenerarcies != N_nu.\n");
            exit(1);
        }
    }

    return 0;
}

int cleanCosmology(struct cosmology *cosmo) {
    if (cosmo->N_nu > 0) {
        free(cosmo->M_nu);
        free(cosmo->deg_nu);
    }
    return 0;
}

double primordialPower(const double k, const struct cosmology *cosmo) {
    if (k == 0) return 0;

    double A_s = cosmo->A_s;
    double n_s = cosmo->n_s;
    double k_pivot = cosmo->k_pivot;

    return A_s * pow(k/k_pivot, n_s - 1.) * k * (2. * M_PI * M_PI);
}
