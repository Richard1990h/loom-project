/* INSTRUMENT / proto-kernel. Hand-written fp32 SGEMM — no cuBLAS, no libraries.
 * Register-blocked, shared-memory-tiled, float4-vectorized. Each thread
 * computes an 8x8 micro-tile of C; a 128x128x8 block tile is staged in shared
 * memory with vectorized (float4) global loads. Target: >= 8190 GFLOP/s at
 * N=2048 (3x the measured tiled baseline of 2730). Correctness cross-checked
 * against a naive kernel.
 *
 * Build: nvcc -O2 -arch=sm_89 -o gpu_matmul_fast gpu_matmul_fast.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t err_=(x); if(err_!=cudaSuccess){ \
    printf("{\"error\":\"%s at line %d\"}\n",cudaGetErrorString(err_),__LINE__); return 1; } }while(0)

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
/* threads/block = (BM/TM)*(BN/TN) = 16*16 = 256 */

__global__ void sgemm_rb(int M,int N,int K,const float* A,const float* B,float* C){
    const int blockRow=blockIdx.y, blockCol=blockIdx.x;
    const int tid=threadIdx.x;
    const int threadRow=tid/(BN/TN);     /* 0..15 */
    const int threadCol=tid%(BN/TN);     /* 0..15 */

    __shared__ float As[BK][BM];         /* transposed: As[k][m] */
    __shared__ float Bs[BK][BN];

    A += blockRow*BM*K;
    B += blockCol*BN;
    C += blockRow*BM*N + blockCol*BN;

    /* load-index maps (256 threads, float4 => 1024 floats per tile) */
    const int aRow=tid/(BK/4), aCol=tid%(BK/4);   /* BK/4=2 */
    const int bRow=tid/(BN/4), bCol=tid%(BN/4);   /* BN/4=32 */

    float acc[TM*TN]={0.0f};
    float regM[TM], regN[TN];

    for(int bk=0; bk<K; bk+=BK){
        float4 av=*reinterpret_cast<const float4*>(&A[aRow*K + aCol*4]);
        As[aCol*4+0][aRow]=av.x; As[aCol*4+1][aRow]=av.y;
        As[aCol*4+2][aRow]=av.z; As[aCol*4+3][aRow]=av.w;
        *reinterpret_cast<float4*>(&Bs[bRow][bCol*4]) =
            *reinterpret_cast<const float4*>(&B[bRow*N + bCol*4]);
        __syncthreads();
        A += BK; B += BK*N;
        #pragma unroll
        for(int d=0; d<BK; d++){
            #pragma unroll
            for(int i=0;i<TM;i++) regM[i]=As[d][threadRow*TM+i];
            #pragma unroll
            for(int j=0;j<TN;j++) regN[j]=Bs[d][threadCol*TN+j];
            #pragma unroll
            for(int i=0;i<TM;i++)
                #pragma unroll
                for(int j=0;j<TN;j++) acc[i*TN+j]+=regM[i]*regN[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<TM;i++)
        #pragma unroll
        for(int j=0;j<TN;j++)
            C[(threadRow*TM+i)*N + threadCol*TN+j]=acc[i*TN+j];
}

/* naive reference for correctness */
__global__ void sgemm_naive(int M,int N,int K,const float*A,const float*B,float*C){
    int r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<M&&c<N){ float s=0; for(int k=0;k<K;k++) s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=s; }
}

static float tms(cudaEvent_t s,cudaEvent_t e){ float m=0; cudaEventElapsedTime(&m,s,e); return m; }

static int bench(int N){
    const size_t bytes=(size_t)N*N*sizeof(float);
    float*hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes);
    for(int i=0;i<N*N;i++){ hA[i]=(float)((i*1103515245u+12345u)&1023)/1024.f;
                            hB[i]=(float)((i*22695477u+1u)&1023)/1024.f; }
    float*dA,*dB,*dC,*dR;
    CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dB,bytes));
    CK(cudaMalloc(&dC,bytes)); CK(cudaMalloc(&dR,bytes));
    CK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));
    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));

    dim3 blk(256), grd(N/BN, N/BM);
    sgemm_rb<<<grd,blk>>>(N,N,N,dA,dB,dC); CK(cudaDeviceSynchronize());
    const int T=10; double best=1e30;
    for(int t=0;t<T;t++){ CK(cudaEventRecord(s)); sgemm_rb<<<grd,blk>>>(N,N,N,dA,dB,dC);
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); double ms=tms(s,e); if(ms<best)best=ms; }
    double gflops=2.0*(double)N*N*N/(best/1e3)/1e9;

    dim3 nb(16,16), ng((N+15)/16,(N+15)/16);
    sgemm_naive<<<ng,nb>>>(N,N,N,dA,dB,dR); CK(cudaDeviceSynchronize());
    float*h1=(float*)malloc(bytes),*h2=(float*)malloc(bytes);
    CK(cudaMemcpy(h1,dC,bytes,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2,dR,bytes,cudaMemcpyDeviceToHost));
    double maxrel=0; for(int i=0;i<N*N;i+=911){ double d=fabs((double)h1[i]-h2[i]);
        double r=d/(fabs((double)h2[i])+1e-6); if(r>maxrel)maxrel=r; }
    printf("  N=%d: register-blocked %.1f GFLOP/s (best %.3f ms) vs naive-check maxrel %.2e\n",
           N, gflops, best, maxrel);
    printf("JSON {\"N\":%d,\"rb_gflops\":%.1f,\"best_ms\":%.3f,\"correct_maxrel\":%.2e}\n",
           N, gflops, best, maxrel);
    free(hA);free(hB);free(h1);free(h2);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dR);
    return 0;
}
int main(){ printf("=== register-blocked SGEMM ===\n"); if(bench(2048)) return 1; if(bench(4096)) return 1; return 0; }
