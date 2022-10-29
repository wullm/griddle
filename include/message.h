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

#ifndef MESSAGE_H
#define MESSAGE_H

#include <stdlib.h>
#include <stdarg.h>
#include <sys/time.h>

#define TXT_RED "\033[31;1m"
#define TXT_GREEN "\033[32;1m"
#define TXT_BLUE "\033[34;1m"
#define TXT_RESET "\033[0m"

static inline void header(int rank, const char *s) {
    if (rank == 0) {
        printf("%s%s%s", TXT_BLUE, s, TXT_RESET);
    }
}

static inline void message(int rank, const char *format, ...) {
    if (rank == 0) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

static inline void catch_error(int err, const char *format, ...) {
    if (err > 0) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        exit(err);
    }
}

struct timepair {
    struct timeval start, stop;
};

static inline void timer_start(int rank, struct timepair *tp) {
    gettimeofday(&tp->start, NULL);
}

static inline void timer_stop(int rank, struct timepair *tp, const char *msg) {
    gettimeofday(&tp->stop, NULL);
    if (rank == 0) {
        printf("%s%.5f s\n", msg,
                ((tp->stop.tv_sec - tp->start.tv_sec) * 1000000
                + tp->stop.tv_usec - tp->start.tv_usec)/1e6);
    }
    gettimeofday(&tp->start, NULL); // ready for next use
}


#endif
