/* INSTRUMENT (not artifact). Hand-written CUDA bandwidth probes — no GPU
 * libraries, only the CUDA runtime.
 *
 * Measures:
 *   - VRAM streaming bandwidth: a grid-stride copy kernel c[i]=a[i].
 *     Bytes moved = 2 * N * 4  (one read + one write). Effective GB/s.
 *   - PCIe host<->device: timed cudaMemcpy, pinned (cudaMallocHost) and
 *     pageable (malloc) host buffers, both directions.
 * Timing by cudaEvent (device timeline). Best of several trials.
 *
 * Build: nvcc -O2 -o gpu_bandwidth gpu_bandwidth.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("{\"error\":\"%s at line %d\"}\n",cudaGetErrorString(e),__LINE__); \
    return 1; } }while(0)

__global__ void copyKernel(const float* __restrict__ a, float* __restrict__ c, long n){
    long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x*blockDim.x;
    for(; i<n; i+=stride) c[i]=a[i];
}

static float time_ms(cudaEvent_t s, cudaEvent_t e){ float m=0; cudaEventElapsedTime(&m,s,e); return m; }

int main(void){
    const long N = 256L*1024*1024;      /* 256M floats = 1 GiB per buffer */
    const size_t bytes = (size_t)N*sizeof(float);
    const int TRIALS = 10;

    float *da=nullptr,*dc=nullptr;
    CK(cudaMalloc(&da,bytes));
    CK(cudaMalloc(&dc,bytes));
    CK(cudaMemset(da,1,bytes));

    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));

    /* ---- VRAM streaming copy ---- */
    int block=256, grid=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid,copyKernel,block,0);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    grid = grid>0 ? grid*p.multiProcessorCount : 4096;
    copyKernel<<<grid,block>>>(da,dc,N); CK(cudaDeviceSynchronize());  /* warmup */
    double bestCopy=1e30;
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s));
        copyKernel<<<grid,block>>>(da,dc,N);
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=time_ms(s,e); if(ms<bestCopy) bestCopy=ms;
    }
    double vramGBs = (2.0*(double)bytes)/(bestCopy/1e3)/1e9;

    /* ---- PCIe: pinned ---- */
    float *hp=nullptr; CK(cudaMallocHost(&hp,bytes));
    double bestH2Dp=1e30,bestD2Hp=1e30;
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s)); CK(cudaMemcpy(da,hp,bytes,cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=time_ms(s,e); if(ms<bestH2Dp) bestH2Dp=ms;
    }
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s)); CK(cudaMemcpy(hp,da,bytes,cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=time_ms(s,e); if(ms<bestD2Hp) bestD2Hp=ms;
    }

    /* ---- PCIe: pageable ---- */
    float *hg=(float*)malloc(bytes);
    double bestH2Dg=1e30,bestD2Hg=1e30;
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s)); CK(cudaMemcpy(da,hg,bytes,cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=time_ms(s,e); if(ms<bestH2Dg) bestH2Dg=ms;
    }
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s)); CK(cudaMemcpy(hg,da,bytes,cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=time_ms(s,e); if(ms<bestD2Hg) bestD2Hg=ms;
    }
    double GB = (double)bytes/1e9;
    printf("{\"buffer_bytes\":%zu,"
           "\"vram_copy_gbps\":%.1f,"
           "\"pcie_h2d_pinned_gbps\":%.2f,\"pcie_d2h_pinned_gbps\":%.2f,"
           "\"pcie_h2d_pageable_gbps\":%.2f,\"pcie_d2h_pageable_gbps\":%.2f}\n",
           bytes, vramGBs,
           GB/(bestH2Dp/1e3), GB/(bestD2Hp/1e3),
           GB/(bestH2Dg/1e3), GB/(bestD2Hg/1e3));

    cudaFreeHost(hp); free(hg); cudaFree(da); cudaFree(dc);
    return 0;
}
