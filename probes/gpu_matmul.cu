/* INSTRUMENT (not artifact, though these kernels prototype what the model
 * will later need). Hand-written CUDA fp32 matmul — no cuBLAS, no libraries.
 *
 * Two kernels for C = A*B (row-major, square NxN):
 *   naive : one thread per C element, K-length dot product from global memory.
 *   tiled : shared-memory tiling, TILE x TILE blocks, coalesced loads.
 * GFLOP/s = 2*N^3 / time. Correctness cross-checked naive-vs-tiled.
 *
 * Build: nvcc -O2 -o gpu_matmul gpu_matmul.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t err_=(x); if(err_!=cudaSuccess){ \
    printf("{\"error\":\"%s at line %d\"}\n",cudaGetErrorString(err_),__LINE__); \
    return 1; } }while(0)

#define TILE 16

__global__ void mmNaive(const float* A,const float* B,float* C,int N){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N && col<N){
        float acc=0.f;
        for(int k=0;k<N;k++) acc += A[row*N+k]*B[k*N+col];
        C[row*N+col]=acc;
    }
}

__global__ void mmTiled(const float* A,const float* B,float* C,int N){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y*TILE+threadIdx.y;
    int col = blockIdx.x*TILE+threadIdx.x;
    float acc=0.f;
    for(int t=0;t<N;t+=TILE){
        As[threadIdx.y][threadIdx.x] = (row<N && t+threadIdx.x<N)? A[row*N+(t+threadIdx.x)] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (col<N && t+threadIdx.y<N)? B[(t+threadIdx.y)*N+col] : 0.f;
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++) acc += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(row<N && col<N) C[row*N+col]=acc;
}

static float tms(cudaEvent_t s,cudaEvent_t e){ float m=0; cudaEventElapsedTime(&m,s,e); return m; }

int main(void){
    const int N=2048;
    const size_t bytes=(size_t)N*N*sizeof(float);
    float *hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes);
    for(int i=0;i<N*N;i++){ hA[i]=(float)((i*1103515245u+12345u)&1023)/1024.f;
                            hB[i]=(float)((i*22695477u+1u)&1023)/1024.f; }
    float *dA,*dB,*dC1,*dC2;
    CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dB,bytes));
    CK(cudaMalloc(&dC1,bytes)); CK(cudaMalloc(&dC2,bytes));
    CK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));

    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    dim3 blk(TILE,TILE), grd((N+TILE-1)/TILE,(N+TILE-1)/TILE);
    const int TRIALS=5;
    double flop=2.0*(double)N*N*N;

    mmNaive<<<grd,blk>>>(dA,dB,dC1,N); CK(cudaDeviceSynchronize());
    double bestNaive=1e30;
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s)); mmNaive<<<grd,blk>>>(dA,dB,dC1,N);
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=tms(s,e); if(ms<bestNaive)bestNaive=ms;
    }
    mmTiled<<<grd,blk>>>(dA,dB,dC2,N); CK(cudaDeviceSynchronize());
    double bestTiled=1e30;
    for(int t=0;t<TRIALS;t++){
        CK(cudaEventRecord(s)); mmTiled<<<grd,blk>>>(dA,dB,dC2,N);
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        double ms=tms(s,e); if(ms<bestTiled)bestTiled=ms;
    }

    /* correctness cross-check */
    float *h1=(float*)malloc(bytes),*h2=(float*)malloc(bytes);
    CK(cudaMemcpy(h1,dC1,bytes,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2,dC2,bytes,cudaMemcpyDeviceToHost));
    double maxrel=0;
    for(int i=0;i<N*N;i+=997){
        double d=fabs((double)h1[i]-h2[i]); double r=d/(fabs((double)h1[i])+1e-6);
        if(r>maxrel)maxrel=r;
    }
    printf("{\"N\":%d,\"naive_gflops\":%.1f,\"tiled_gflops\":%.1f,"
           "\"naive_ms\":%.3f,\"tiled_ms\":%.3f,\"naive_vs_tiled_maxrel\":%.2e}\n",
           N, flop/(bestNaive/1e3)/1e9, flop/(bestTiled/1e3)/1e9,
           bestNaive, bestTiled, maxrel);
    free(hA);free(hB);free(h1);free(h2);
    cudaFree(dA);cudaFree(dB);cudaFree(dC1);cudaFree(dC2);
    return 0;
}
