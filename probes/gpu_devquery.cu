/* INSTRUMENT (not artifact). Hand-written CUDA device query — no GPU
 * libraries (no cuBLAS/cuDNN/thrust), only the CUDA runtime, which the
 * purity decision admits as the GPU's C/driver layer.
 *
 * Prints device name + SM count (must be RTX 4070) plus the numbers that
 * later design math will cite. Build: nvcc -O2 -o gpu_devquery gpu_devquery.cu
 */
#include <cstdio>
#include <cuda_runtime.h>

int main(void){
    int n=0;
    if(cudaGetDeviceCount(&n)!=cudaSuccess || n==0){
        printf("{\"error\":\"no CUDA device visible\"}\n"); return 1;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p,0);
    int rtV=0, drV=0;
    cudaRuntimeGetVersion(&rtV);
    cudaDriverGetVersion(&drV);
    double memGiB = (double)p.totalGlobalMem/ (1024.0*1024.0*1024.0);
    printf("{\"name\":\"%s\",\"sm_count\":%d,\"cc\":\"%d.%d\","
           "\"clock_mhz\":%d,\"mem_clock_mhz\":%d,\"bus_width_bits\":%d,"
           "\"l2_bytes\":%d,\"total_mem_gib\":%.3f,"
           "\"runtime\":%d,\"driver\":%d}\n",
           p.name, p.multiProcessorCount, p.major, p.minor,
           p.clockRate/1000, p.memoryClockRate/1000, p.memoryBusWidth,
           p.l2CacheSize, memGiB, rtV, drV);
    return 0;
}
