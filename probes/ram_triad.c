/* INSTRUMENT (not artifact). Exempt from the no-library rule: this only
 * measures the machine; it is not part of the model. Uses libc + OpenMP.
 *
 * Host RAM bandwidth via the STREAM TRIAD kernel:  a[i] = b[i] + s*c[i].
 * Per element: 2 reads (b,c) + 1 write (a) = 3 * sizeof(double) bytes moved.
 * Arrays are sized far beyond LLC so we measure DRAM, not cache. Multi-
 * threaded (OpenMP) to reach aggregate memory-controller bandwidth.
 *
 * Build: gcc -O3 -march=native -fopenmp -o ram_triad ram_triad.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifndef N
#define N (48*1024*1024)   /* 48M doubles per array -> 3 arrays = 1.15 GiB */
#endif
#define NTRIAL 10

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec;
}

int main(void){
    double *a = malloc((size_t)N*sizeof(double));
    double *b = malloc((size_t)N*sizeof(double));
    double *c = malloc((size_t)N*sizeof(double));
    if(!a||!b||!c){ fprintf(stderr,"alloc failed\n"); return 1; }

    #pragma omp parallel for
    for(long i=0;i<N;i++){ a[i]=0.0; b[i]=1.0; c[i]=2.0; }

    const double s = 3.0;
    double best = 1e300;
    for(int t=0;t<NTRIAL;t++){
        double t0 = now_s();
        #pragma omp parallel for
        for(long i=0;i<N;i++) a[i] = b[i] + s*c[i];
        double dt = now_s() - t0;
        if(dt<best) best = dt;
        /* touch a to prevent dead-code elimination */
        if(a[t % N] < 0) printf("x");
    }
    double bytes = 3.0*(double)N*sizeof(double);
    double gbps = bytes/best/1e9;   /* GB/s (decimal, matches vendor convention) */
    printf("{\"kernel\":\"triad\",\"array_doubles\":%d,\"bytes_per_iter\":%.0f,"
           "\"best_time_s\":%.6f,\"gbps\":%.2f}\n", N, bytes, best, gbps);
    free(a);free(b);free(c);
    return 0;
}
