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

#ifndef FFT_KERNELS_H
#define FFT_KERNELS_H

#include "cosmology.h"
#include "strooklat.h"

typedef void (*kernel_func)(struct kernel *the_kernel);

static inline void kernel_constant(struct kernel *the_kernel) {
    double constant = *((double*) the_kernel->params);
    the_kernel->kern = constant;
}

static inline void kernel_lowpass(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double k_max = *((double*) the_kernel->params);
    the_kernel->kern = (k < k_max) ? 1.0 : 0.0;
}

static inline void kernel_hipass(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double k_min = *((double*) the_kernel->params);
    the_kernel->kern = (k > k_min) ? 1.0 : 0.0;
}

static inline void kernel_tophat(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double *param = (double *) the_kernel->params;
    double k_min = param[0];
    double k_max = param[1];
    the_kernel->kern = (k >= k_min && k <= k_max) ? 1.0 : 0.0;
}


static inline void kernel_tophat_real(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double R = *((double *) the_kernel->params);
    double kR = k * R;
    the_kernel->kern = (k > 0) ? 3. * (sin(kR) - kR * cos(kR))/(kR * kR * kR) : 1.0;
}

static inline void kernel_elliptic_tophat(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    double ky = the_kernel->ky;
    double kz = the_kernel->kz;
    double *param = (double *) the_kernel->params;
    double rx = param[0];
    double ry = param[1];
    double rz = param[2];
    double eta2 = param[3];
    double kr = kx*rx + ky*ry + kz*rz;
    double kk = kx*kx + ky*ky + kz*kz;
    double rr = rx*rx + ry*ry + rz*rz;
    double theta = kk*rr + (eta2 - 1)*kr*kr;
    the_kernel->kern = (theta <= 4*M_PI*M_PI) ? 1.0 : 0.0;
}

static inline void kernel_translate(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    double ky = the_kernel->ky;
    double kz = the_kernel->kz;
    double *param = (double *) the_kernel->params;
    double rx = param[0];
    double ry = param[1];
    double rz = param[2];
    the_kernel->kern = cexp(I*(rx*kx + ry*ky + rz*kz));
}

static inline void kernel_gaussian(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double R = *((double*) the_kernel->params);
    double kR = k * R;
    the_kernel->kern = exp(-kR * kR);
}

static inline void kernel_gaussian_z(struct kernel *the_kernel) {
    double kz = the_kernel->kz;
    double R = *((double*) the_kernel->params);
    double kR = kz * R;
    the_kernel->kern = exp(-kR * kR);
}

static inline void kernel_gaussian_inv(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double R = *((double*) the_kernel->params);
    double kR = k * R;
    the_kernel->kern = exp(kR * kR);
}


static inline void kernel_inv_poisson(struct kernel *the_kernel) {
    double k = the_kernel->k;
    the_kernel->kern = (k > 0) ? -1.0/k/k : 1.0;
}

static inline void kernel_dx(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    the_kernel->kern = I*kx;
}

static inline void kernel_dy(struct kernel *the_kernel) {
    double ky = the_kernel->ky;
    the_kernel->kern = I*ky;
}

static inline void kernel_dz(struct kernel *the_kernel) {
    double kz = the_kernel->kz;
    the_kernel->kern = I*kz;
}

static inline void kernel_power_no_transfer(struct kernel *the_kernel) {
    const struct cosmology *cosmo = (const struct cosmology *) the_kernel->params;
    double k = the_kernel->k;
    double kern = sqrt(primordialPower(k, cosmo));
    the_kernel->kern = kern;
}

struct spline_params {
    struct strooklat *spline_z;
    struct strooklat *spline_k;
    double z; //redshift
    double *tf; //the transfer function
};

/* Multiply by the transfer function */
static inline void kernel_transfer_function(struct kernel *the_kernel) {
    double k = the_kernel->k;

    if (k == 0) {
        /* Ignore the DC mode */
        the_kernel->kern = 0.0;
    } else {
        /* Unpack the spline parameters */
        struct spline_params *sp = (struct spline_params *) the_kernel->params;

        /* Evaluate the transfer function T(z,k) using bilinear interpolation */
        the_kernel->kern = strooklat_interp_2d(sp->spline_z, sp->spline_k, sp->tf, sp->z, k);
    }
}

/* Divide by the transfer function */
static inline void kernel_inv_transfer_function(struct kernel *the_kernel) {
    double k = the_kernel->k;

    if (k == 0) {
        /* Ignore the DC mode */
        the_kernel->kern = 0.f;
    } else {
        /* Unpack the spline parameters */
        struct spline_params *sp = (struct spline_params *) the_kernel->params;

        /* Evaluate the transfer function T(z,k) using bilinear interpolation */
        the_kernel->kern = 1.0 / strooklat_interp_2d(sp->spline_z, sp->spline_k, sp->tf, sp->z, k);
    }
}



/* Sinc function */
inline double sinc(double x) { return x == 0 ? 1. : sin(x) / x; }

struct Hermite_kern_params {
    int order;
    int N;
    double boxlen;
};


/* Support for undoing the CIC (order = 2), TSC (order = 3), and higher Hermite
 * polynonial window functions. The order is passed as paramater to the kernel. */
static inline void kernel_undo_Hermite_window(struct kernel *the_kernel) {

    if (the_kernel->k == 0) {
        the_kernel->kern = 1.0;
    } else {
        /* Unpack the parameters */
        struct Hermite_kern_params *p = (struct Hermite_kern_params *) the_kernel->params;
        int order = p->order;

        double kx = the_kernel->kx;
        double ky = the_kernel->ky;
        double kz = the_kernel->kz;

        /* The Hermite Window function in Fourier space */
        double W_x = sinc(0.5 * kx * p->boxlen / p->N);
        double W_y = sinc(0.5 * ky * p->boxlen / p->N);
        double W_z = sinc(0.5 * kz * p->boxlen / p->N);
        double W = pow(W_x * W_y * W_z, order);

        the_kernel->kern = 1.0 / W;
    }
}

#endif
