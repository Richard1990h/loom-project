/* INSTRUMENT / proto-kernels. Hand-written fused single-head causal attention,
 * forward + backward, fp32 CUDA — no libraries. Verified against a CPU-fp64
 * reference (fp32 tolerance). Math matches the CPU engine (day3): scaled
 * dot-product scores, causal mask, row-softmax, O=P·V.
 * Build: nvcc -O2 -arch=sm_89 -o gpu_attention gpu_attention.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("{\"error\":\"%s@%d\"}\n",cudaGetErrorString(e_),__LINE__); return 1; } }while(0)

/* forward: one thread per query row i. Fuses score+mask+softmax+PV; writes P. */
__global__ void attn_fwd(int Ls,int d,float scale,const float*Q,const float*K,const float*V,
                         float*O,float*P){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=Ls) return;
    float m=-1e30f;
    for(int j=0;j<=i;j++){ float s=0; for(int k=0;k<d;k++) s+=Q[i*d+k]*K[j*d+k]; s*=scale;
        P[i*Ls+j]=s; if(s>m)m=s; }
    float se=0; for(int j=0;j<=i;j++){ float e=expf(P[i*Ls+j]-m); P[i*Ls+j]=e; se+=e; }
    for(int j=0;j<=i;j++) P[i*Ls+j]/=se;
    for(int j=i+1;j<Ls;j++) P[i*Ls+j]=0.f;
    for(int k=0;k<d;k++){ float o=0; for(int j=0;j<=i;j++) o+=P[i*Ls+j]*V[j*d+k]; O[i*d+k]=o; }
}
/* backward part 1: per query row i -> dS row, dQ row. Writes dS. */
__global__ void attn_bwd_row(int Ls,int d,float scale,const float*Q,const float*K,const float*V,
                             const float*P,const float*dO,float*dS,float*dQ){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=Ls) return;
    /* dP[i,j]=sum_k dO[i,k]V[j,k]; rowdot=sum_j P[i,j]dP[i,j] */
    float rowdot=0;
    for(int j=0;j<=i;j++){ float dp=0; for(int k=0;k<d;k++) dp+=dO[i*d+k]*V[j*d+k];
        dS[i*Ls+j]=dp; rowdot+=P[i*Ls+j]*dp; }
    for(int j=0;j<=i;j++){ float ds=P[i*Ls+j]*(dS[i*Ls+j]-rowdot)*scale; dS[i*Ls+j]=ds; }
    for(int j=i+1;j<Ls;j++) dS[i*Ls+j]=0.f;
    for(int k=0;k<d;k++){ float g=0; for(int j=0;j<=i;j++) g+=dS[i*Ls+j]*K[j*d+k]; dQ[i*d+k]=g; }
}
/* backward part 2: per key row j -> dK[j], dV[j] */
__global__ void attn_bwd_col(int Ls,int d,const float*Q,const float*P,const float*dO,const float*dS,
                             float*dK,float*dV){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=Ls) return;
    for(int k=0;k<d;k++){ float gv=0,gk=0;
        for(int i=j;i<Ls;i++){ gv+=P[i*Ls+j]*dO[i*d+k]; gk+=dS[i*Ls+j]*Q[i*d+k]; }
        dV[j*d+k]=gv; dK[j*d+k]=gk; }
}

static float rr(unsigned* s){ *s=*s*1103515245u+12345u; return (float)((*s>>9)&0x7fff)/32768.f*2-1; }

int main(){
    int L=64,d=32; float scale=1.f/sqrtf((float)d); unsigned s=13;
    size_t szLd=(size_t)L*d*4, szLL=(size_t)L*L*4;
    float*Q=(float*)malloc(szLd),*K=(float*)malloc(szLd),*Vv=(float*)malloc(szLd),*dO=(float*)malloc(szLd);
    for(int i=0;i<L*d;i++){Q[i]=rr(&s);K[i]=rr(&s);Vv[i]=rr(&s);dO[i]=rr(&s);}
    float*dQ,*dK,*dV,*dOd,*dO_,*dP,*dS,*dOut;
    CK(cudaMalloc(&dQ,szLd));CK(cudaMalloc(&dK,szLd));CK(cudaMalloc(&dV,szLd));
    CK(cudaMalloc(&dOd,szLd));CK(cudaMalloc(&dO_,szLd));CK(cudaMalloc(&dP,szLL));CK(cudaMalloc(&dS,szLL));CK(cudaMalloc(&dOut,szLd));
    CK(cudaMemcpy(dQ,Q,szLd,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,K,szLd,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,Vv,szLd,cudaMemcpyHostToDevice));CK(cudaMemcpy(dO_,dO,szLd,cudaMemcpyHostToDevice));
    float *dgQ,*dgK,*dgV; CK(cudaMalloc(&dgQ,szLd));CK(cudaMalloc(&dgK,szLd));CK(cudaMalloc(&dgV,szLd));
    int nb=(L+63)/64;
    attn_fwd<<<nb,64>>>(L,d,scale,dQ,dK,dV,dOut,dP); CK(cudaDeviceSynchronize());
    attn_bwd_row<<<nb,64>>>(L,d,scale,dQ,dK,dV,dP,dO_,dS,dgQ); CK(cudaDeviceSynchronize());
    attn_bwd_col<<<nb,64>>>(L,d,dQ,dP,dO_,dS,dgK,dgV); CK(cudaDeviceSynchronize());
    float*gO=(float*)malloc(szLd),*gQ=(float*)malloc(szLd),*gK=(float*)malloc(szLd),*gV=(float*)malloc(szLd);
    CK(cudaMemcpy(gO,dOut,szLd,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gQ,dgQ,szLd,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gK,dgK,szLd,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gV,dgV,szLd,cudaMemcpyDeviceToHost));

    /* ---- CPU fp64 reference ---- */
    double*P=(double*)calloc(L*L,8),*O=(double*)calloc(L*d,8);
    for(int i=0;i<L;i++){ double m=-1e300; double s2[64];
        for(int j=0;j<=i;j++){ double x=0; for(int k=0;k<d;k++)x+=(double)Q[i*d+k]*K[j*d+k]; x*=scale; s2[j]=x; if(x>m)m=x; }
        double se=0; for(int j=0;j<=i;j++){ double e=exp(s2[j]-m); P[i*L+j]=e; se+=e; }
        for(int j=0;j<=i;j++) P[i*L+j]/=se;
        for(int k=0;k<d;k++){ double o=0; for(int j=0;j<=i;j++)o+=P[i*L+j]*Vv[j*d+k]; O[i*d+k]=o; } }
    double*gQr=(double*)calloc(L*d,8),*gKr=(double*)calloc(L*d,8),*gVr=(double*)calloc(L*d,8);
    double*Sr=(double*)calloc(L*L,8);
    for(int i=0;i<L;i++){ double rowdot=0; double dp[64];
        for(int j=0;j<=i;j++){ double x=0; for(int k=0;k<d;k++)x+=(double)dO[i*d+k]*Vv[j*d+k]; dp[j]=x; rowdot+=P[i*L+j]*x; }
        for(int j=0;j<=i;j++) Sr[i*L+j]=P[i*L+j]*(dp[j]-rowdot)*scale;
        for(int k=0;k<d;k++){ double g=0; for(int j=0;j<=i;j++)g+=Sr[i*L+j]*K[j*d+k]; gQr[i*d+k]=g; } }
    for(int j=0;j<L;j++) for(int k=0;k<d;k++){ double gv=0,gk=0;
        for(int i=j;i<L;i++){ gv+=P[i*L+j]*dO[i*d+k]; gk+=Sr[i*L+j]*Q[i*d+k]; } gVr[j*d+k]=gv; gKr[j*d+k]=gk; }

    double eO=0,eQ=0,eK=0,eV=0;
    for(int i=0;i<L*d;i++){
        eO=fmax(eO,fabs(O[i]-gO[i])/(fabs(O[i])+1e-4));
        eQ=fmax(eQ,fabs(gQr[i]-gQ[i])/(fabs(gQr[i])+1e-4));
        eK=fmax(eK,fabs(gKr[i]-gK[i])/(fabs(gKr[i])+1e-4));
        eV=fmax(eV,fabs(gVr[i]-gV[i])/(fabs(gVr[i])+1e-4)); }
    int pass = (eO<1e-3&&eQ<1e-3&&eK<1e-3&&eV<1e-3);
    printf("fused attention vs fp64: O %.2e dQ %.2e dK %.2e dV %.2e -> %s\n", eO,eQ,eK,eV, pass?"PASS":"FAIL");
    printf("JSON {\"attn_O\":%.2e,\"attn_dQ\":%.2e,\"attn_dK\":%.2e,\"attn_dV\":%.2e,\"pass\":%s}\n",
           eO,eQ,eK,eV,pass?"true":"false");
    return pass?0:1;
}
