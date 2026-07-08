/* day4c.cu — CPU-fp64 vs GPU-fp32 TRAINING PARITY proof.
 *
 * Identical tiny model (embed(cur,prev) -> Q,K,V -> causal attention -> O ->
 * Wout -> masked cross-entropy) on the day3 induction task. Same seed, same
 * per-step data. One path trains entirely in CPU double; the other trains
 * entirely in GPU float using the hand-written kernels (no cuBLAS/cuDNN).
 * Both start from identical weights.
 *
 * Pre-registered tolerance (ZERO.md): |L_gpu_fp32 - L_cpu_fp64| <= 2e-2 for the
 * first 200 steps AND final-step losses within 5% relative. Gate before any
 * GPU training. No libraries.
 *
 * Build: nvcc -O2 -arch=sm_89 -o day4c day4c.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("CUDA %s@%d\n",cudaGetErrorString(e_),__LINE__); exit(1);} }while(0)

#define V 8
#define L 24
#define D 32
#define F (2*V)          /* input feature dim (cur one-hot + prev one-hot) */

/* ---- host RNG (deterministic, precision-independent) ---- */
static unsigned long long RNG=88172645463325252ULL;
static void seed(unsigned long long s){ RNG=s; }
static double urand(){ RNG^=RNG<<13; RNG^=RNG>>7; RNG^=RNG<<17; return (double)(RNG>>11)/(double)(1ULL<<53); }
static int irand(int n){ return (int)(urand()*n); }

/* ================= GPU kernels (fp32) ================= */
/* flexible gemm: C[M,N] (+)= opA(A)[M,K] * opB(B)[K,N]; transA/transB pick layout */
__global__ void gemm(int M,int N,int K,const float*A,int tA,const float*B,int tB,float*C,int acc){
    int m=blockIdx.y*blockDim.y+threadIdx.y, n=blockIdx.x*blockDim.x+threadIdx.x;
    if(m>=M||n>=N) return; float s=0;
    for(int k=0;k<K;k++){ float a= tA? A[k*M+m] : A[m*K+k]; float b= tB? B[n*K+k] : B[k*N+n]; s+=a*b; }
    C[m*N+n] = acc? C[m*N+n]+s : s;
}
__global__ void attn_fwd(float scale,const float*Q,const float*K,const float*Vv,float*O,float*P){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return;
    float m=-1e30f;
    for(int j=0;j<=i;j++){ float s=0; for(int k=0;k<D;k++) s+=Q[i*D+k]*K[j*D+k]; s*=scale; P[i*L+j]=s; if(s>m)m=s; }
    float se=0; for(int j=0;j<=i;j++){ float e=expf(P[i*L+j]-m); P[i*L+j]=e; se+=e; }
    for(int j=0;j<=i;j++) P[i*L+j]/=se; for(int j=i+1;j<L;j++) P[i*L+j]=0.f;
    for(int k=0;k<D;k++){ float o=0; for(int j=0;j<=i;j++) o+=P[i*L+j]*Vv[j*D+k]; O[i*D+k]=o; }
}
__global__ void attn_bwd_row(float scale,const float*K,const float*Vv,const float*P,const float*dO,float*dS,float*dQ){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return; float rowdot=0;
    for(int j=0;j<=i;j++){ float dp=0; for(int k=0;k<D;k++) dp+=dO[i*D+k]*Vv[j*D+k]; dS[i*L+j]=dp; rowdot+=P[i*L+j]*dp; }
    for(int j=0;j<=i;j++){ float ds=P[i*L+j]*(dS[i*L+j]-rowdot)*scale; dS[i*L+j]=ds; }
    for(int j=i+1;j<L;j++) dS[i*L+j]=0.f;
    for(int k=0;k<D;k++){ float g=0; for(int j=0;j<=i;j++) g+=dS[i*L+j]*K[j*D+k]; dQ[i*D+k]=g; }
}
__global__ void attn_bwd_col(const float*Q,const float*P,const float*dO,const float*dS,float*dK,float*dV){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=L) return;
    for(int k=0;k<D;k++){ float gv=0,gk=0; for(int i=j;i<L;i++){ gv+=P[i*L+j]*dO[i*D+k]; gk+=dS[i*L+j]*Q[i*D+k]; } dV[j*D+k]=gv; dK[j*D+k]=gk; }
}
/* masked xent: probs, per-row loss, dlogits = mask*(p-onehot)/nact */
__global__ void xent(const float*logit,const int*tg,const int*mask,int nact,float*dlog,float*rowloss){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return;
    float mx=logit[i*V]; for(int v=1;v<V;v++) if(logit[i*V+v]>mx) mx=logit[i*V+v];
    float se=0; for(int v=0;v<V;v++) se+=expf(logit[i*V+v]-mx);
    float inv = nact>0? 1.f/nact : 0.f;
    for(int v=0;v<V;v++){ float p=expf(logit[i*V+v]-mx)/se; dlog[i*V+v]=mask[i]*(p-(v==tg[i]?1.f:0.f))*inv; }
    rowloss[i]= mask[i]? ((mx+logf(se))-logit[i*V+tg[i]]) : 0.f;
}
__global__ void rmsprop(int n,float*p,const float*g,float*v,float lr,float beta,float eps){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i], vi=beta*v[i]+(1.f-beta)*gi*gi; v[i]=vi; p[i]-=lr*gi/(sqrtf(vi)+eps);
}

/* ================= parameters ================= */
struct Params { float *E,*Wq,*Wk,*Wv,*Wo; };            /* device fp32 */
struct ParamsD{ double *E,*Wq,*Wk,*Wv,*Wo; };           /* host double  */
#define NE (F*D)
#define NW (D*D)
#define NO (D*V)

int main(){
    seed(20260707ULL);
    float scale=1.f/sqrtf((float)D), lr=0.02f,beta=0.9f,eps=1e-8f;
    /* master init (fp32 values, stored as double too so both start identical) */
    float hE[NE],hWq[NW],hWk[NW],hWv[NW],hWo[NO];
    #define INIT(arr,n,sc) for(int i=0;i<n;i++) arr[i]=(float)((2.0*urand()-1.0)*sc);
    INIT(hE,NE,0.5/sqrt((double)F)) INIT(hWq,NW,0.5/sqrt((double)D)) INIT(hWk,NW,0.5/sqrt((double)D))
    INIT(hWv,NW,0.5/sqrt((double)D)) INIT(hWo,NO,0.5/sqrt((double)D))

    /* ---- CPU double state ---- */
    ParamsD C; C.E=(double*)malloc(NE*8);C.Wq=(double*)malloc(NW*8);C.Wk=(double*)malloc(NW*8);
    C.Wv=(double*)malloc(NW*8);C.Wo=(double*)malloc(NO*8);
    for(int i=0;i<NE;i++)C.E[i]=hE[i]; for(int i=0;i<NW;i++){C.Wq[i]=hWq[i];C.Wk[i]=hWk[i];C.Wv[i]=hWv[i];}
    for(int i=0;i<NO;i++)C.Wo[i]=hWo[i];
    double vE[NE]={0},vWq[NW]={0},vWk[NW]={0},vWv[NW]={0},vWo[NO]={0};

    /* ---- GPU float state ---- */
    Params G; float*gvE,*gvWq,*gvWk,*gvWv,*gvWo;
    CK(cudaMalloc(&G.E,NE*4));CK(cudaMalloc(&G.Wq,NW*4));CK(cudaMalloc(&G.Wk,NW*4));CK(cudaMalloc(&G.Wv,NW*4));CK(cudaMalloc(&G.Wo,NO*4));
    CK(cudaMalloc(&gvE,NE*4));CK(cudaMalloc(&gvWq,NW*4));CK(cudaMalloc(&gvWk,NW*4));CK(cudaMalloc(&gvWv,NW*4));CK(cudaMalloc(&gvWo,NO*4));
    CK(cudaMemcpy(G.E,hE,NE*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(G.Wq,hWq,NW*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G.Wk,hWk,NW*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(G.Wv,hWv,NW*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G.Wo,hWo,NO*4,cudaMemcpyHostToDevice));
    CK(cudaMemset(gvE,0,NE*4));CK(cudaMemset(gvWq,0,NW*4));CK(cudaMemset(gvWk,0,NW*4));CK(cudaMemset(gvWv,0,NW*4));CK(cudaMemset(gvWo,0,NO*4));
    /* device scratch */
    float *dX,*dXe,*dQ,*dK,*dVv,*dO,*dP,*dLog,*ddLog,*dRL,*dS,*ddO;
    float *dgWo,*dgE,*dgWq,*dgWk,*dgWv,*dgXe;
    CK(cudaMalloc(&dX,L*F*4));CK(cudaMalloc(&dXe,L*D*4));CK(cudaMalloc(&dQ,L*D*4));CK(cudaMalloc(&dK,L*D*4));
    CK(cudaMalloc(&dVv,L*D*4));CK(cudaMalloc(&dO,L*D*4));CK(cudaMalloc(&dP,L*L*4));CK(cudaMalloc(&dLog,L*V*4));
    CK(cudaMalloc(&ddLog,L*V*4));CK(cudaMalloc(&dRL,L*4));CK(cudaMalloc(&dS,L*L*4));CK(cudaMalloc(&ddO,L*D*4));
    CK(cudaMalloc(&dgWo,NO*4));CK(cudaMalloc(&dgE,NE*4));CK(cudaMalloc(&dgWq,NW*4));CK(cudaMalloc(&dgWk,NW*4));
    CK(cudaMalloc(&dgWv,NW*4));CK(cudaMalloc(&dgXe,L*D*4));
    int *dtg,*dmask; CK(cudaMalloc(&dtg,L*4));CK(cudaMalloc(&dmask,L*4));

    int STEPS=400; double worst200=0, cpuL=0, gpuL=0;
    for(int step=1; step<=STEPS; step++){
        /* ---- generate one sequence (shared data) ---- */
        int x[L]; for(int i=0;i<L;i++) x[i]=irand(V);
        float hX[L*F]; memset(hX,0,sizeof(hX));
        int tg[L],mask[L]; int nact=0;
        for(int i=0;i<L;i++){ hX[i*F+x[i]]=1.f; if(i>0) hX[i*F+V+x[i-1]]=1.f;
            tg[i]=0; mask[i]=0; for(int j=i-1;j>=0;j--) if(x[j]==x[i]){ tg[i]=x[j+1]; mask[i]=1; break; }
            if(mask[i]) nact++; }

        /* ================= CPU fp64 ================= */
        double Xe[L*D],Q[L*D],Kk[L*D],Vv[L*D],O[L*D],P[L*L],Lg[L*V];
        for(int i=0;i<L;i++)for(int a=0;a<D;a++){ double e=0; for(int f=0;f<F;f++) if(hX[i*F+f]!=0) e+=C.E[f*D+a]; Xe[i*D+a]=e; }
        for(int i=0;i<L;i++)for(int a=0;a<D;a++){ double q=0,k2=0,v=0; for(int b=0;b<D;b++){ double xe=Xe[i*D+b]; q+=xe*C.Wq[b*D+a]; k2+=xe*C.Wk[b*D+a]; v+=xe*C.Wv[b*D+a]; } Q[i*D+a]=q;Kk[i*D+a]=k2;Vv[i*D+a]=v; }
        for(int i=0;i<L;i++){ double mx=-1e300; for(int j=0;j<=i;j++){ double s=0; for(int a=0;a<D;a++)s+=Q[i*D+a]*Kk[j*D+a]; s*=scale; P[i*L+j]=s; if(s>mx)mx=s; }
            double se=0; for(int j=0;j<=i;j++){ double e=exp(P[i*L+j]-mx); P[i*L+j]=e; se+=e; } for(int j=0;j<=i;j++)P[i*L+j]/=se; for(int j=i+1;j<L;j++)P[i*L+j]=0;
            for(int a=0;a<D;a++){ double o=0; for(int j=0;j<=i;j++)o+=P[i*L+j]*Vv[j*D+a]; O[i*D+a]=o; } }
        for(int i=0;i<L;i++)for(int v=0;v<V;v++){ double s=0; for(int a=0;a<D;a++)s+=O[i*D+a]*C.Wo[a*V+v]; Lg[i*V+v]=s; }
        double closs=0; double dLg[L*V];
        for(int i=0;i<L;i++){ double mx=Lg[i*V]; for(int v=1;v<V;v++) if(Lg[i*V+v]>mx)mx=Lg[i*V+v];
            double se=0; for(int v=0;v<V;v++)se+=exp(Lg[i*V+v]-mx); double inv=nact>0?1.0/nact:0;
            for(int v=0;v<V;v++){ double p=exp(Lg[i*V+v]-mx)/se; dLg[i*V+v]=mask[i]*(p-(v==tg[i]?1.0:0.0))*inv; }
            if(mask[i]) closs+=(mx+log(se))-Lg[i*V+tg[i]]; }
        closs = nact>0? closs/nact:0;
        /* backward (CPU locals prefixed b* to avoid clashing with device ptrs) */
        double bdO[L*D],gWo[NO]={0}; for(int a=0;a<D;a++)for(int v=0;v<V;v++){ double g=0; for(int i=0;i<L;i++)g+=O[i*D+a]*dLg[i*V+v]; gWo[a*V+v]=g; }
        for(int i=0;i<L;i++)for(int a=0;a<D;a++){ double g=0; for(int v=0;v<V;v++)g+=dLg[i*V+v]*C.Wo[a*V+v]; bdO[i*D+a]=g; }
        double bS[L*L],bQ[L*D]={0},bK[L*D]={0},bV[L*D]={0};
        for(int i=0;i<L;i++){ double rowdot=0; for(int j=0;j<=i;j++){ double dp=0; for(int a=0;a<D;a++)dp+=bdO[i*D+a]*Vv[j*D+a]; bS[i*L+j]=dp; rowdot+=P[i*L+j]*dp; }
            for(int j=0;j<=i;j++) bS[i*L+j]=P[i*L+j]*(bS[i*L+j]-rowdot)*scale; for(int j=i+1;j<L;j++)bS[i*L+j]=0;
            for(int a=0;a<D;a++){ double g=0; for(int j=0;j<=i;j++)g+=bS[i*L+j]*Kk[j*D+a]; bQ[i*D+a]=g; } }
        for(int j=0;j<L;j++)for(int a=0;a<D;a++){ double gv=0,gk=0; for(int i=j;i<L;i++){ gv+=P[i*L+j]*bdO[i*D+a]; gk+=bS[i*L+j]*Q[i*D+a]; } bV[j*D+a]=gv; bK[j*D+a]=gk; }
        double bXe[L*D],gWq[NW]={0},gWk[NW]={0},gWv[NW]={0},gE[NE]={0};
        for(int b=0;b<D;b++)for(int a=0;a<D;a++){ double q=0,k2=0,v=0; for(int i=0;i<L;i++){ q+=Xe[i*D+b]*bQ[i*D+a]; k2+=Xe[i*D+b]*bK[i*D+a]; v+=Xe[i*D+b]*bV[i*D+a]; } gWq[b*D+a]=q;gWk[b*D+a]=k2;gWv[b*D+a]=v; }
        for(int i=0;i<L;i++)for(int b=0;b<D;b++){ double g=0; for(int a=0;a<D;a++)g+=bQ[i*D+a]*C.Wq[b*D+a]+bK[i*D+a]*C.Wk[b*D+a]+bV[i*D+a]*C.Wv[b*D+a]; bXe[i*D+b]=g; }
        for(int f=0;f<F;f++)for(int a=0;a<D;a++){ double g=0; for(int i=0;i<L;i++) if(hX[i*F+f]!=0) g+=bXe[i*D+a]; gE[f*D+a]=g; }
        #define UPD(p,g,vv,n) for(int q=0;q<n;q++){ double gg=g[q]; vv[q]=beta*vv[q]+(1.0-beta)*gg*gg; p[q]-=lr*gg/(sqrt(vv[q])+eps);}
        UPD(C.Wo,gWo,vWo,NO) UPD(C.Wq,gWq,vWq,NW) UPD(C.Wk,gWk,vWk,NW) UPD(C.Wv,gWv,vWv,NW) UPD(C.E,gE,vE,NE)

        /* ================= GPU fp32 ================= */
        CK(cudaMemcpy(dX,hX,L*F*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtg,tg,L*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dmask,mask,L*4,cudaMemcpyHostToDevice));
        dim3 b16(16,16);
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,F,dX,0,G.E,0,dXe,0);
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dXe,0,G.Wq,0,dQ,0);
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dXe,0,G.Wk,0,dK,0);
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dXe,0,G.Wv,0,dVv,0);
        attn_fwd<<<1,L>>>(scale,dQ,dK,dVv,dO,dP);
        gemm<<<dim3((V+15)/16,(L+15)/16),b16>>>(L,V,D,dO,0,G.Wo,0,dLog,0);
        xent<<<1,L>>>(dLog,dtg,dmask,nact,ddLog,dRL);
        float rl[L]; CK(cudaMemcpy(rl,dRL,L*4,cudaMemcpyDeviceToHost));
        double gloss=0; for(int i=0;i<L;i++) gloss+=rl[i]; gloss= nact>0? gloss/nact:0;
        /* backward */
        gemm<<<dim3((V+15)/16,(D+15)/16),b16>>>(D,V,L,dO,1,ddLog,0,dgWo,0);   /* dWo = O^T dLog */
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,V,ddLog,0,G.Wo,1,ddO,0);  /* dO = dLog Wo^T */
        attn_bwd_row<<<1,L>>>(scale,dK,dVv,dP,ddO,dS,dQ /*reuse as dQgrad*/);
        /* NOTE: dQ now holds dQ-grad; need Q for bwd_col but Q overwritten -> recompute Q */
        float* dQg=dQ; float* dKg; float* dVg;
        CK(cudaMalloc(&dKg,L*D*4)); CK(cudaMalloc(&dVg,L*D*4));
        /* recompute Q,K into temp for bwd_col (Q consumed). We still have dK(=K),dVv(=V) intact. */
        float* dQrecomp; CK(cudaMalloc(&dQrecomp,L*D*4));
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dXe,0,G.Wq,0,dQrecomp,0);
        attn_bwd_col<<<1,L>>>(dQrecomp,dP,ddO,dS,dKg,dVg);
        /* dXe = dQg Wq^T + dKg Wk^T + dVg Wv^T */
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dQg,0,G.Wq,1,dgXe,0);
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dKg,0,G.Wk,1,dgXe,1);
        gemm<<<dim3((D+15)/16,(L+15)/16),b16>>>(L,D,D,dVg,0,G.Wv,1,dgXe,1);
        /* dWq = Xe^T dQg etc */
        gemm<<<dim3((D+15)/16,(D+15)/16),b16>>>(D,D,L,dXe,1,dQg,0,dgWq,0);
        gemm<<<dim3((D+15)/16,(D+15)/16),b16>>>(D,D,L,dXe,1,dKg,0,dgWk,0);
        gemm<<<dim3((D+15)/16,(D+15)/16),b16>>>(D,D,L,dXe,1,dVg,0,dgWv,0);
        /* dE = X^T dXe */
        gemm<<<dim3((D+15)/16,(F+15)/16),b16>>>(F,D,L,dX,1,dgXe,0,dgE,0);
        /* rmsprop updates */
        rmsprop<<<(NO+255)/256,256>>>(NO,G.Wo,dgWo,gvWo,lr,beta,eps);
        rmsprop<<<(NW+255)/256,256>>>(NW,G.Wq,dgWq,gvWq,lr,beta,eps);
        rmsprop<<<(NW+255)/256,256>>>(NW,G.Wk,dgWk,gvWk,lr,beta,eps);
        rmsprop<<<(NW+255)/256,256>>>(NW,G.Wv,dgWv,gvWv,lr,beta,eps);
        rmsprop<<<(NE+255)/256,256>>>(NE,G.E,dgE,gvE,lr,beta,eps);
        CK(cudaDeviceSynchronize());
        cudaFree(dKg);cudaFree(dVg);cudaFree(dQrecomp);

        double diff=fabs(closs-gloss);
        if(step<=200 && diff>worst200) worst200=diff;
        cpuL=closs; gpuL=gloss;
        if(step<=5 || step%100==0) printf("step %3d: cpu_fp64 %.6f  gpu_fp32 %.6f  |d| %.2e\n", step, closs, gloss, diff);
    }
    double finalrel = fabs(cpuL-gpuL)/(fabs(cpuL)+1e-9);
    int pass = (worst200<=2e-2) && (finalrel<=0.05);
    printf("\nworst |d| over first 200 steps: %.3e (bar<=2e-2)\n", worst200);
    printf("final: cpu %.6f gpu %.6f  rel %.3e (bar<=5%%)\n", cpuL, gpuL, finalrel);
    printf("PARITY -> %s\n", pass?"PASS":"FAIL");
    printf("JSON {\"worst200\":%.3e,\"cpu_final\":%.6f,\"gpu_final\":%.6f,\"final_rel\":%.3e,\"pass\":%s}\n",
           worst200,cpuL,gpuL,finalrel,pass?"true":"false");
    return pass?0:1;
}
