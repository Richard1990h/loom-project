/* INSTRUMENT (not artifact). Exempt from the no-library rule: measures the
 * machine only. Uses libc + OpenMP + x86 AVX2 intrinsics.
 *
 * Sustained FMA throughput (GFLOP/s), fp32 and fp64, using 256-bit AVX2
 * fused multiply-add (_mm256_fmadd) with a bank of independent accumulators
 * to hide FMA latency and saturate both FMA ports on Zen4. A scalar/auto-
 * vectorized loop understates throughput (compiler emitted 128-bit FMA:
 * fp32 ~= fp64 ~760 GF, a tell-tale that width was 2 doubles). Hand-issuing
 * 256-bit FMAs measures the real sustained rate. 1 FMA lane = 2 FLOPs.
 *
 * Build: gcc -O3 -march=native -fopenmp -o cpu_flops cpu_flops.c
 * Reported as MEASURED sustained throughput, not a vendor peak claim.
 */
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define ACC 12               /* independent 256-bit accumulators */
#define ITERS 40000000L      /* FMA-issue iterations per thread */

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec;
}

int main(void){
    int nth=0;
    #pragma omp parallel
    {
        #pragma omp master
        nth = omp_get_num_threads();
    }

    /* ---- fp64: 4 lanes per 256-bit vector ---- */
    double t0 = now_s();
    double red64 = 0.0;
    #pragma omp parallel reduction(+:red64)
    {
        __m256d a = _mm256_set1_pd(1.0000000001);
        __m256d b = _mm256_set1_pd(0.9999999999);
        __m256d x[ACC];
        for(int k=0;k<ACC;k++) x[k]=_mm256_set1_pd((double)(k+1));
        for(long i=0;i<ITERS;i++)
            for(int k=0;k<ACC;k++) x[k]=_mm256_fmadd_pd(x[k],a,b);
        double buf[4]; __m256d s=_mm256_setzero_pd();
        for(int k=0;k<ACC;k++) s=_mm256_add_pd(s,x[k]);
        _mm256_storeu_pd(buf,s);
        red64 += buf[0]+buf[1]+buf[2]+buf[3];
    }
    double dt64 = now_s()-t0;
    double gf64 = (2.0*4.0*(double)ACC*(double)ITERS*(double)nth)/dt64/1e9;

    /* ---- fp32: 8 lanes per 256-bit vector ---- */
    t0 = now_s();
    float red32 = 0.0f;
    #pragma omp parallel reduction(+:red32)
    {
        __m256 a = _mm256_set1_ps(1.0000001f);
        __m256 b = _mm256_set1_ps(0.9999999f);
        __m256 x[ACC];
        for(int k=0;k<ACC;k++) x[k]=_mm256_set1_ps((float)(k+1));
        for(long i=0;i<ITERS;i++)
            for(int k=0;k<ACC;k++) x[k]=_mm256_fmadd_ps(x[k],a,b);
        float buf[8]; __m256 s=_mm256_setzero_ps();
        for(int k=0;k<ACC;k++) s=_mm256_add_ps(s,x[k]);
        _mm256_storeu_ps(buf,s);
        for(int j=0;j<8;j++) red32 += buf[j];
    }
    double dt32 = now_s()-t0;
    double gf32 = (2.0*8.0*(double)ACC*(double)ITERS*(double)nth)/dt32/1e9;

    printf("{\"threads\":%d,\"isa\":\"avx2-fma256\",\"fp64_gflops\":%.1f,"
           "\"fp32_gflops\":%.1f,\"red64\":%.3g,\"red32\":%.3g}\n",
           nth, gf64, gf32, red64, (double)red32);
    return 0;
}
