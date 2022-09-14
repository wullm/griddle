/*******************************************************************************
 * This file is part of Nyver.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_integration.h>

#include "../include/cosmology_tables.h"
#include "../include/units.h"
#include "../include/strooklat.h"


double F_integrand(double x, void *params) {
    double y = *((double*) params);
    double x2 = x * x;
    double y2 = y * y;
    return x2 * sqrt(x2 + y2) / (1.0 + exp(x));
}

double G_integrand(double x, void *params) {
    double y = *((double*) params);
    double x2 = x * x;
    double y2 = y * y;
    return y * x2 / (sqrt(x2 + y2) * (1.0 + exp(x)));
}

double w_tilde(double a, double w0, double wa) {
    return (a - 1.0) * wa - (1.0 + w0 + wa) * log(a);
}

double E2(double a, double Omega_CMB, double Omega_ur, double Omega_nu,
          double Omega_cdm, double Omega_b, double Omega_lambda, double Omega_k,
          double w0, double wa) {

    const double a_inv = 1.0 / a;
    const double a_inv2 = a_inv * a_inv;
    const double E2 = (Omega_CMB + Omega_ur + Omega_nu) * a_inv2 * a_inv2 +
                      (Omega_cdm + Omega_b) * a_inv2 * a_inv +
                      Omega_k * a_inv2 +
                      Omega_lambda * exp(3. * w_tilde(a, w0, wa));
    return E2;
}

struct int_pars {
    struct strooklat *spline;
    double *Omega_nu_tot;
    double Omega_CMB;
    double Omega_ur;
    double Omega_cdm;
    double Omega_b;
    double Omega_lambda;
    double Omega_k;
    double w0;
    double wa;
    double H_0;
};

/* The kick time step dt / a = dtau */
double kick_integrand(double a, void *params) {
    struct int_pars *pars = (struct int_pars *) params;
    double Omega_nu_a = strooklat_interp(pars->spline, pars->Omega_nu_tot, a);
    double Ea = sqrt(E2(a, pars->Omega_CMB, pars->Omega_ur, Omega_nu_a,
                        pars->Omega_cdm, pars->Omega_b, pars->Omega_lambda,
                        pars->Omega_k, pars->w0, pars->wa));
    return 1.0 / (a * a * Ea * pars->H_0);
}

/* The drift time step dt / (a^2) = dtau / a */
double drift_integrand(double a, void *params) {
    struct int_pars *pars = (struct int_pars *) params;
    double Omega_nu_a = strooklat_interp(pars->spline, pars->Omega_nu_tot, a);
    double Ea = sqrt(E2(a, pars->Omega_CMB, pars->Omega_ur, Omega_nu_a,
                        pars->Omega_cdm, pars->Omega_b, pars->Omega_lambda,
                        pars->Omega_k, pars->w0, pars->wa));
    return 1.0 / (a * a * a * Ea * pars->H_0);
}

void integrate_cosmology_tables(struct cosmology *c, struct units *us,
                                struct physical_consts *pcs,
                                struct cosmology_tables *tab, double a_start,
                                double a_final, int size) {


    /* Prepare interpolation tables of F(y) and G(y) with y > 0 */
    const int table_size = 500;
    const double y_min = 1e-4;
    const double y_max = 1e6;
    const double log_y_min = log(y_min);
    const double log_y_max = log(y_max);
    const double delta_log_y = (log_y_max - log_y_min) / table_size;

    /* Allocate the tables */
    double *y = malloc(table_size * sizeof(double));
    double *Fy = malloc(table_size * sizeof(double));
    double *Gy = malloc(table_size * sizeof(double));

    /* Prepare GSL integration workspace */
    const int gsl_workspace_size = 1000;
    const double abs_tol = 1e-10;
    const double rel_tol = 1e-10;

    /* Allocate the workspace */
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(gsl_workspace_size);
    gsl_function func_F = {F_integrand};
    gsl_function func_G = {G_integrand};

    /* Perform the numerical integration */
    for (int i=0; i<table_size; i++) {
        y[i] = exp(log_y_min + i * delta_log_y);

        /* Integration result and absolute error */
        double res, abs_err;

        /* Evaluate F(y) by integrating on [0, infinity) */
        func_F.params = &y[i];
        gsl_integration_qagiu(&func_F, 0.0, abs_tol, rel_tol, gsl_workspace_size, workspace, &res, &abs_err);
        Fy[i] = res;

        /* Evaluate G(y) by integrating on [0, infinity) and dividing by F(y) */
        func_G.params = &y[i];
        gsl_integration_qagiu(&func_G, 0.0, abs_tol, rel_tol, gsl_workspace_size, workspace, &res, &abs_err);
        Gy[i] = y[i] * res / Fy[i];
    }

    /* Prepare an interpolation spline for the y argument of F and G */
    struct strooklat spline_y = {y, table_size};
    init_strooklat_spline(&spline_y, 100);

    /* We want to interpolate the scale factor */
    const double a_min = a_start;
    const double a_max = fmax(a_final, 1.01);
    const double log_a_min = log(a_min);
    const double log_a_max = log(a_max);
    const double delta_log_a = (log_a_max - log_a_min) / size;

    tab->size = size;
    tab->avec = malloc(size * sizeof(double));
    tab->Hvec = malloc(size * sizeof(double));
    double *Ga = malloc(size * sizeof(double));
    double *E2a = malloc(size * sizeof(double));
    double *dHdloga = malloc(size * sizeof(double));

    for (int i=0; i<size; i++) {
        tab->avec[i] = exp(log_a_min + i * delta_log_a);
        Ga[i] = sqrt(tab->avec[i] + 1);
    }

    /* Prepare a spline for the scale factor */
    struct strooklat spline = {tab->avec, size};
    init_strooklat_spline(&spline, 100);

    /* The critical density */
    const double h = c->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double G_grav = pcs->GravityG;
    const double rho_crit_0 = 3.0 * H_0 * H_0 / (8.0 * M_PI * G_grav);

    /* First, calculate the present-day CMB density from the temperature */
    const double h_bar = pcs->hPlanck / (2.0 * M_PI);
    const double kT = c->T_CMB_0 * pcs->kBoltzmann;
    const double hc = h_bar * pcs->SpeedOfLight;
    const double kT4 = kT * kT * kT * kT;
    const double hc3 = hc * hc * hc;
    const double c2 = pcs->SpeedOfLight * pcs->SpeedOfLight;
    const double Omega_CMB = M_PI * M_PI / 15.0 * (kT4 / hc3) / (rho_crit_0 * c2);

    /* Other density components */
    const double Omega_cdm = c->Omega_cdm;
    const double Omega_b = c->Omega_b;
    const double Omega_k = c->Omega_k;

    /* Next, calculate the ultra-relativistic density */
    const double ratio = 4. / 11.;
    const double ratio4 = ratio * ratio * ratio * ratio;
    const double Omega_ur = c->N_ur * (7. / 8.) * cbrt(ratio4) * Omega_CMB;

    /* Now, we want to evaluate the neutrino density and equation of state */
    const int N_nu = c->N_nu;
    const double kT_nu_eV_0 = c->T_nu_0 * pcs->kBoltzmann / pcs->ElectronVolt;
    const double T_on_pi = c->T_nu_0 / c->T_CMB_0 / M_PI;
    const double pre_factor = Omega_CMB * 15.0 * T_on_pi * T_on_pi * T_on_pi * T_on_pi;
    double *Omega_nu = malloc(N_nu * size * sizeof(double));
    double *w_nu = malloc(N_nu * size * sizeof(double));

    /* For each neutrino species */
    for (int j=0; j<N_nu; j++) {
        const double M_nu = c->M_nu[j];
        const double deg_nu = c->deg_nu[j];

        /* For each time step, interpolate the distribution function */
        for (int i=0; i<size; i++) {
            /* Compute the density */
            const double arg = tab->avec[i] * M_nu / kT_nu_eV_0;
            const double Farg = strooklat_interp(&spline_y, Fy, arg);
            const double Onu_ij = deg_nu * pre_factor * Farg;
            Omega_nu[j * size + i] = Onu_ij;

            /* Also compute the equation of state */
            const double Garg = strooklat_interp(&spline_y, Gy, arg);
            w_nu[j * size + i] = (1.0 - Garg) / 3.0;
        }

    }

    /* Split the neutrino densities into relativistic and non-relativistic parts */
    double *Omega_nu_nr = malloc(size * sizeof(double));
    double *Omega_nu_tot = malloc(size * sizeof(double));
    double *Omega_r = malloc(size * sizeof(double));
    double *Omega_m = malloc(size * sizeof(double));
    tab->f_nu_nr = malloc(size * N_nu * sizeof(double));
    tab->f_nu_nr_tot = malloc(size * sizeof(double));

    for (int i=0; i<size; i++) {

        /* Start with constant contributions to radiation & matter */
        Omega_r[i] = Omega_CMB + Omega_ur;
        Omega_m[i] = Omega_cdm + Omega_b;
        Omega_nu_nr[i] = 0.0;
        Omega_nu_tot[i] = 0.0;

        /* Add the massive neutrino species */
        for (int j=0; j<N_nu; j++) {
            const double O_nu = Omega_nu[j * size + i];
            const double w = w_nu[j * size + i];
            Omega_nu_tot[i] += O_nu;
            Omega_nu_nr[i] += (1.0 - 3.0 * w) * O_nu;
            Omega_r[i] += 3.0 * w * O_nu;
            /* We rescale by 1/a, since this is in fact Omega_m * E^2 * a^3 and
             * Omega_nu is in fact Omega_nu * E^2 * a^4 */
            Omega_m[i] += (1.0 - 3.0 * w) * O_nu / tab->avec[i];
        }

        /* Fraction of non-relativistic neutrinos in matter */
        tab->f_nu_nr_tot[i] = Omega_nu_nr[i] / Omega_m[i] / tab->avec[i];

        /* Fraction per species */
        for (int j=0; j<N_nu; j++) {
            const double O_nu = Omega_nu[j * size + i];
            const double w = w_nu[j * size + i];
            const double O_nu_nr = (1.0 - 3.0 * w) * O_nu;
            tab->f_nu_nr[j * size + i] = O_nu_nr / Omega_m[i] / tab->avec[i];
        }
    }

    /* The total neutrino density at z = 0 */
    const double Omega_nu_tot_0 = strooklat_interp(&spline, Omega_nu_tot, 1.0);

    /* The neutrino density per species at z = 0 */
    double *Omega_nu_0 = malloc(N_nu * sizeof(double));
    for (int i = 0; i < N_nu; i++) {
        Omega_nu_0[i] = strooklat_interp(&spline, Omega_nu + i * size, 1.0);
    }

    /* Close the universe */
    const double Omega_lambda = 1.0 - Omega_nu_tot_0 - Omega_k - Omega_ur - Omega_CMB - Omega_cdm - Omega_b;
    const double w0 = c->w0;
    const double wa = c->wa;

    /* Now, create a table with the Hubble rate */
    for (int i=0; i<size; i++) {
        double Omega_nu_a = strooklat_interp(&spline, Omega_nu_tot, tab->avec[i]);
        E2a[i] = E2(tab->avec[i], Omega_CMB, Omega_ur, Omega_nu_a, Omega_cdm,
                       Omega_b, Omega_lambda, Omega_k, w0, wa);
        tab->Hvec[i] = sqrt(E2a[i]) * H_0;
    }

    /* Now, differentiate the Hubble rate */
    for (int i=0; i<size; i++) {
        /* Forward at the start, five-point in the middle, backward at the end */
        if (i < 2) {
            dHdloga[i] = (tab->Hvec[i+1] - tab->Hvec[i]) / delta_log_a;
        } else if (i < size - 2) {
            dHdloga[i]  = tab->Hvec[i-2];
            dHdloga[i] -= tab->Hvec[i-1] * 8.0;
            dHdloga[i] += tab->Hvec[i+1] * 8.0;
            dHdloga[i] -= tab->Hvec[i+2];
            dHdloga[i] /= 12.0 * delta_log_a;
        } else {
            dHdloga[i] = (tab->Hvec[i] - tab->Hvec[i-1]) / delta_log_a;
        }
    }

    /* Now, allocate space for the kick and drift factors */
    tab->kick_factors = malloc(size * sizeof(double));
    tab->drift_factors = malloc(size * sizeof(double));

    /* Package the parameters needed for the integration */
    struct int_pars ips = {&spline, Omega_nu_tot, Omega_CMB, Omega_ur, Omega_cdm,
                           Omega_b, Omega_lambda, Omega_k, w0, wa, H_0};

   /* Prepare integrating the kick and drift factors with GSL */
   gsl_function func_kick = {kick_integrand};
   gsl_function func_drift = {drift_integrand};

   double kd_abs_tol = 1e-8;
   double kd_rel_tol = 1e-8;

   /* Compute the kick and drift factors */
    for (int i=0; i<size; i++) {
        double a = tab->avec[i];

        /* Integration result and absolute error */
        double res, abs_err;

        /* Evaluate the kick time step */
        func_kick.params = &ips;
        gsl_integration_qag(&func_kick, a_start, a, kd_abs_tol, kd_rel_tol,
                            gsl_workspace_size, GSL_INTEG_GAUSS15, workspace, &res, &abs_err);
        tab->kick_factors[i] = res;

        /* Evaluate the drift time step */
        func_drift.params = &ips;
        gsl_integration_qag(&func_drift, a_start, a, kd_abs_tol, kd_rel_tol,
                            gsl_workspace_size, GSL_INTEG_GAUSS15, workspace, &res, &abs_err);
        tab->drift_factors[i] = res;
    }

    /* Free the GSL workspace */
    gsl_integration_workspace_free(workspace);

    /* Now, create the A and B functions */
    tab->Avec = malloc(size * sizeof(double));
    tab->Bvec = malloc(size * sizeof(double));

    for (int i=0; i<size; i++) {
        double a = tab->avec[i];
        tab->Avec[i] = -(2.0 + dHdloga[i] / tab->Hvec[i]);
        tab->Bvec[i] = -1.5 * Omega_m[i] / (a * a * a) / E2a[i];
    }

    /* Clean up arrays */
    free(Omega_nu_nr);
    free(Omega_r);
    free(Omega_m);
    free(Omega_nu);
    free(Omega_nu_tot);
    free(Omega_nu_0);
    free(w_nu);
    free(dHdloga);
    free(E2a);
    free(Ga);

    /* Free the interpolation tables */
    free(y);
    free(Fy);
    free(Gy);

    /* Free the interpolation splines */
    free_strooklat_spline(&spline);
    free_strooklat_spline(&spline_y);
}

double get_H_of_a(struct cosmology_tables *tab, double a) {
    /* Prepare a spline for the scale factor */
    struct strooklat spline = {tab->avec, tab->size};
    init_strooklat_spline(&spline, 100);

    /* Interpolate */
    double Ha = strooklat_interp(&spline, tab->Hvec, a);

    /* Free the spline */
    free_strooklat_spline(&spline);

    return Ha;
}

double get_f_nu_nr_tot_of_a(struct cosmology_tables *tab, double a) {
    /* Prepare a spline for the scale factor */
    struct strooklat spline = {tab->avec, tab->size};
    init_strooklat_spline(&spline, 100);

    /* Interpolate */
    double f_nu_nr_tot = strooklat_interp(&spline, tab->f_nu_nr_tot, a);

    /* Free the spline */
    free_strooklat_spline(&spline);

    return f_nu_nr_tot;
}

void free_cosmology_tables(struct cosmology_tables *tab) {
    free(tab->avec);
    free(tab->Avec);
    free(tab->Bvec);
    free(tab->Hvec);
    free(tab->f_nu_nr);
    free(tab->f_nu_nr_tot);
    free(tab->kick_factors);
    free(tab->drift_factors);
}
