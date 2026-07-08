/* INSTRUMENT / proto-kernels. Hand-written fp32 CUDA — no libraries.
 * RMSProp optimizer update, embedding gather (fwd), embedding scatter-add
 * (bwd). Each verified against a CPU-fp64 reference (fp32 tolerance).
 * Build: nvcc -O2 -arch=sm_89 -o gpu_kernels gpu_kernels.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("{\"error\":\"%s@%d\"}\n",cudaGetErrorString(e_),__LINE__); return 1; } }while(0)

/* ---- RMSProp: v=beta v+(1-beta)g^2 ; p -= lr*g/(sqrt(v)+eps) ---- */
__global__ void rmsprop(int n,float* p,const float* g,float* v,float lr,float beta,float eps){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i]; float vi=beta*v[i]+(1.f-beta)*gi*gi; v[i]=vi;
    p[i]-=lr*gi/(sqrtf(vi)+eps);
}
/* ---- embedding gather: O[t,:] = E[idx[t],:] ---- */
__global__ void embed_gather(int T,int d,const float* E,const int* idx,float* O){
    int t=blockIdx.x, k=threadIdx.x; if(t>=T||k>=d) return;
    O[t*d+k]=E[idx[t]*d+k];
}
/* ---- embedding scatter-add: dE[idx[t],:] += dO[t,:] (atomic) ---- */
__global__ void embed_scatter(int T,int d,float* dE,const int* idx,const float* dO){
    int t=blockIdx.x, k=threadIdx.x; if(t>=T||k>=d) return;
    atomicAdd(&dE[idx[t]*d+k], dO[t*d+k]);
}

static float rrand(unsigned* s){ *s=*s*1103515245u+12345u; return (float)((*s>>9)&0x7fff)/32768.f*2-1; }

int main(){
    unsigned s=7; int fails=0;
    /* ---- RMSProp test ---- */
    {
        int n=4096; float lr=0.01f,beta=0.9f,eps=1e-8f;
        float *p=(float*)malloc(n*4),*g=(float*)malloc(n*4),*v=(float*)malloc(n*4);
        for(int i=0;i<n;i++){p[i]=rrand(&s);g[i]=rrand(&s);v[i]=fabsf(rrand(&s));}
        double *pp=(double*)malloc(n*8),*vv=(double*)malloc(n*8);
        for(int i=0;i<n;i++){pp[i]=p[i];vv[i]=v[i];}
        float *dp,*dg,*dv; CK(cudaMalloc(&dp,n*4));CK(cudaMalloc(&dg,n*4));CK(cudaMalloc(&dv,n*4));
        CK(cudaMemcpy(dp,p,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dg,g,n*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv,v,n*4,cudaMemcpyHostToDevice));
        rmsprop<<<(n+255)/256,256>>>(n,dp,dg,dv,lr,beta,eps); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(p,dp,n*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<n;i++){ double vn=beta*vv[i]+(1.0-beta)*(double)g[i]*g[i];
            double pn=pp[i]-(double)lr*g[i]/(sqrt(vn)+eps); }
        double mx=0; for(int i=0;i<n;i++){ double vn=beta*vv[i]+(1.0-beta)*(double)g[i]*g[i];
            double pn=pp[i]-(double)lr*g[i]/(sqrt(vn)+eps);
            double r=fabs(pn-p[i])/(fabs(pn)+1e-6); if(r>mx)mx=r; }
        printf("rmsprop update: maxrel %.2e -> %s\n", mx, mx<1e-3?"PASS":"FAIL"); if(mx>=1e-3)fails++;
        cudaFree(dp);cudaFree(dg);cudaFree(dv); free(p);free(g);free(v);free(pp);free(vv);
    }
    /* ---- embed gather + scatter test ---- */
    {
        int Vn=100,d=32,T=64;
        float*E=(float*)malloc(Vn*d*4); for(int i=0;i<Vn*d;i++)E[i]=rrand(&s);
        int*idx=(int*)malloc(T*4); for(int t=0;t<T;t++)idx[t]=(int)((s=s*1103515245u+12345u)>>16)%Vn;
        float*dO=(float*)malloc(T*d*4); for(int i=0;i<T*d;i++)dO[i]=rrand(&s);
        float*dE,*dEd,*dOO; int*didx; float*dOut;
        CK(cudaMalloc(&dE,Vn*d*4));CK(cudaMalloc(&didx,T*4));CK(cudaMalloc(&dOut,T*d*4));
        CK(cudaMalloc(&dEd,Vn*d*4));CK(cudaMalloc(&dOO,T*d*4));
        CK(cudaMemcpy(dE,E,Vn*d*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(didx,idx,T*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dOO,dO,T*d*4,cudaMemcpyHostToDevice));CK(cudaMemset(dEd,0,Vn*d*4));
        embed_gather<<<T,d>>>(T,d,dE,didx,dOut); CK(cudaDeviceSynchronize());
        float*outg=(float*)malloc(T*d*4); CK(cudaMemcpy(outg,dOut,T*d*4,cudaMemcpyDeviceToHost));
        double mg=0; for(int t=0;t<T;t++)for(int k=0;k<d;k++){ double ref=E[idx[t]*d+k];
            double r=fabs(ref-outg[t*d+k]); if(r>mg)mg=r; }
        printf("embed gather: maxabs %.2e -> %s\n", mg, mg<1e-6?"PASS":"FAIL"); if(mg>=1e-6)fails++;
        embed_scatter<<<T,d>>>(T,d,dEd,didx,dOO); CK(cudaDeviceSynchronize());
        float*eg=(float*)malloc(Vn*d*4); CK(cudaMemcpy(eg,dEd,Vn*d*4,cudaMemcpyDeviceToHost));
        double*ref=(double*)calloc(Vn*d,8); for(int t=0;t<T;t++)for(int k=0;k<d;k++)ref[idx[t]*d+k]+=(double)dO[t*d+k];
        double ms2=0; for(int i=0;i<Vn*d;i++){ double r=fabs(ref[i]-eg[i])/(fabs(ref[i])+1e-6); if(r>ms2)ms2=r; }
        printf("embed scatter-add: maxrel %.2e -> %s\n", ms2, ms2<1e-3?"PASS":"FAIL"); if(ms2>=1e-3)fails++;
        free(E);free(idx);free(dO);free(outg);free(eg);free(ref);
        cudaFree(dE);cudaFree(didx);cudaFree(dOut);cudaFree(dEd);cudaFree(dOO);
    }
    printf("JSON {\"kernels_failed\":%d}\n", fails);
    return fails?1:0;
}
