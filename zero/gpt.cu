/* gpt.cu — GPU transformer trainer (full 5e architecture), fp32, hand-written
 * kernels only (no cuBLAS/cuDNN/thrust). ARTIFACT + a self-contained CPU-fp64
 * reference used to GATE it. Pre-norm transformer:
 *   h = embed(tokens);  per layer: h += Wo(attn(RoPE Q,K; causal; softmax; ·V));
 *   h += FFN(GELU) ;  logits = LN(h) @ Eᵀ  (output head tied to token embedding).
 *
 * modes:
 *   check   gate: gradcheck (fp32 analytic vs fp64 finite-diff) + CPU-fp64 vs
 *           GPU-fp32 loss-curve parity, on a tiny config.
 *   train   Day 5g pretrain (added after the gate passes).
 *
 * Build: nvcc -O2 -arch=sm_89 -o gpt gpt.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("CUDA %s@%d\n",cudaGetErrorString(e_),__LINE__); exit(1);} }while(0)

static int V,D,DFF,NL,L; static float SCALE; static double RB=10000.0;
static unsigned long long RNG=88172645463325252ULL;
static void seed(unsigned long long s){ RNG=s; }
static double urand(){ RNG^=RNG<<13; RNG^=RNG>>7; RNG^=RNG<<17; return (double)(RNG>>11)/(double)(1ULL<<53); }
static int irnd(int n){ return (int)(urand()*n); }

/* ================= kernels ================= */
__global__ void kgemm(int M,int N,int Kk,const float*A,int tA,const float*B,int tB,float*C,int acc){
    int m=blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.x*blockDim.x+threadIdx.x; if(m>=M||n>=N)return; float s=0;
    for(int k=0;k<Kk;k++){ float a=tA?A[k*M+m]:A[m*Kk+k]; float b=tB?B[n*Kk+k]:B[k*N+n]; s+=a*b; } C[m*N+n]=acc?C[m*N+n]+s:s; }
__global__ void krope(const float*in,float*out,int Ls,int Dd,double base,int inv){
    int i=blockIdx.x,k=threadIdx.x; if(i>=Ls||k>=Dd/2)return; double th=(double)i*pow(base,-2.0*k/Dd); float c=cos(th),s=sin(th);
    float x0=in[i*Dd+2*k],x1=in[i*Dd+2*k+1];
    if(!inv){ out[i*Dd+2*k]=x0*c-x1*s; out[i*Dd+2*k+1]=x0*s+x1*c; } else { out[i*Dd+2*k]=x0*c+x1*s; out[i*Dd+2*k+1]=-x0*s+x1*c; } }
__global__ void kln_fwd(const float*X,const float*g,float*Y,float*mean,float*rstd,int R,int C){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=R)return; float mu=0; for(int j=0;j<C;j++)mu+=X[i*C+j]; mu/=C;
    float var=0; for(int j=0;j<C;j++){float d=X[i*C+j]-mu;var+=d*d;} var/=C; float is=rsqrtf(var+1e-5f); mean[i]=mu; rstd[i]=is;
    for(int j=0;j<C;j++)Y[i*C+j]=((X[i*C+j]-mu)*is)*g[j]; }
__global__ void kln_bwd(const float*X,const float*g,const float*dY,float*dX,float*dg,const float*mean,const float*rstd,int R,int C){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=R)return; float mu=mean[i],is=rstd[i]; float mdx=0,mxx=0;
    for(int j=0;j<C;j++){ float xh=(X[i*C+j]-mu)*is; float dxh=dY[i*C+j]*g[j]; mdx+=dxh; mxx+=dxh*xh; } mdx/=C; mxx/=C;
    for(int j=0;j<C;j++){ float xh=(X[i*C+j]-mu)*is; float dxh=dY[i*C+j]*g[j]; dX[i*C+j]=is*(dxh-mdx-xh*mxx); atomicAdd(&dg[j],dY[i*C+j]*xh); } }
__global__ void kaddb_fwd(const float*X,const float*b,float*Y,int R,int C){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=R*C)return; Y[i]=X[i]+b[i%C]; }
__global__ void kaddb_db(const float*dY,float*db,int R,int C){ int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=C)return; float s=0; for(int i=0;i<R;i++)s+=dY[i*C+j]; atomicAdd(&db[j],s); }
__global__ void kgelu_fwd(const float*X,float*Y,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float x=X[i]; Y[i]=0.5f*x*(1.f+erff(x*0.70710678f)); }
__global__ void kgelu_bwd(const float*X,const float*dY,float*dX,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float x=X[i]; float c=0.5f*(1.f+erff(x*0.70710678f)); float d=c+x*expf(-0.5f*x*x)*0.39894228f; dX[i]=d*dY[i]; }
__global__ void kadd(const float*a,const float*b,float*y,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=a[i]+b[i]; }
__global__ void kcopy(const float*a,float*y,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=a[i]; }
__global__ void kzero(float*p,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)p[i]=0; }
__global__ void kattn_fwd(float sc,const float*Q,const float*K,const float*Vv,float*O,float*P,int Ls,int Dd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=Ls)return; float m=-1e30f;
    for(int j=0;j<=i;j++){ float s=0; for(int k=0;k<Dd;k++)s+=Q[i*Dd+k]*K[j*Dd+k]; s*=sc; P[i*Ls+j]=s; if(s>m)m=s; }
    float se=0; for(int j=0;j<=i;j++){ float e=expf(P[i*Ls+j]-m); P[i*Ls+j]=e; se+=e; }
    for(int j=0;j<=i;j++)P[i*Ls+j]/=se; for(int j=i+1;j<Ls;j++)P[i*Ls+j]=0;
    for(int k=0;k<Dd;k++){ float o=0; for(int j=0;j<=i;j++)o+=P[i*Ls+j]*Vv[j*Dd+k]; O[i*Dd+k]=o; } }
__global__ void kattn_bwd_row(float sc,const float*K,const float*Vv,const float*P,const float*dO,float*dS,float*dQ,int Ls,int Dd){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=Ls)return; float rd=0;
    for(int j=0;j<=i;j++){ float dp=0; for(int k=0;k<Dd;k++)dp+=dO[i*Dd+k]*Vv[j*Dd+k]; dS[i*Ls+j]=dp; rd+=P[i*Ls+j]*dp; }
    for(int j=0;j<=i;j++)dS[i*Ls+j]=P[i*Ls+j]*(dS[i*Ls+j]-rd)*sc; for(int j=i+1;j<Ls;j++)dS[i*Ls+j]=0;
    for(int k=0;k<Dd;k++){ float g=0; for(int j=0;j<=i;j++)g+=dS[i*Ls+j]*K[j*Dd+k]; dQ[i*Dd+k]=g; } }
__global__ void kattn_bwd_col(const float*Q,const float*P,const float*dO,const float*dS,float*dK,float*dV,int Ls,int Dd){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=Ls)return;
    for(int k=0;k<Dd;k++){ float gv=0,gk=0; for(int i=j;i<Ls;i++){ gv+=P[i*Ls+j]*dO[i*Dd+k]; gk+=dS[i*Ls+j]*Q[i*Dd+k]; } dV[j*Dd+k]=gv; dK[j*Dd+k]=gk; } }
__global__ void kxent(const float*lg,const int*tg,const int*mask,int nact,float*dlg,float*rl,int Ls,int Vv){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=Ls)return; float mx=lg[i*Vv]; for(int v=1;v<Vv;v++)if(lg[i*Vv+v]>mx)mx=lg[i*Vv+v];
    float se=0; for(int v=0;v<Vv;v++)se+=expf(lg[i*Vv+v]-mx); float inv=nact>0?1.f/nact:0;
    for(int v=0;v<Vv;v++){ float p=expf(lg[i*Vv+v]-mx)/se; dlg[i*Vv+v]=mask[i]*(p-(v==tg[i]?1.f:0.f))*inv; }
    rl[i]=mask[i]?((mx+logf(se))-lg[i*Vv+tg[i]]):0.f; }
__global__ void krmsprop(int n,float*p,const float*g,float*v,float lr,float beta,float eps){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float gi=g[i],vi=beta*v[i]+(1.f-beta)*gi*gi; v[i]=vi; p[i]-=lr*gi/(sqrtf(vi)+eps); }

static dim3 B2(16,16);
static void GEMM(int M,int N,int Kk,const float*A,int tA,const float*B,int tB,float*C,int acc){ kgemm<<<dim3((N+15)/16,(M+15)/16),B2>>>(M,N,Kk,A,tA,B,tB,C,acc); }
static void Z(float*p,int n){ kzero<<<(n+255)/256,256>>>(p,n); }
static void E1(int n){ } /* placeholder */

/* ================= state ================= */
struct Layer{ float *Wq,*Wk,*Wv,*Wo,*W1,*b1,*W2,*b2,*l1g,*l1b,*l2g,*l2b; };
static Layer P_[16],Gr[16],Vs[16];
static float *E,*gE,*vE,*lnfg,*glnfg,*vlnfg,*lnfb,*glnfb,*vlnfb;
static float *Hs[17],*HR1[16],*n1[16],*m1[16],*r1[16],*m2[16],*r2[16],*qr[16],*kr[16],*vv[16],*Pt[16],*ao[16];
static float *n2[16],*f1[16],*gg[16],*nf,*mf,*rf,*logit;
static float *dS,*dq,*dk,*dv,*dqr,*dkr,*dn,*dn2,*dtmp,*dtmp2,*dh,*dhr,*ddlog,*rowloss,*dnf,*qtmp,*ktmp,*dao,*daout,*dff;
static int *dtg,*dmask;
static float* dm(int n){ float*p; CK(cudaMalloc(&p,(size_t)n*4)); return p; }

/* list of all param tensors for zeroing grads + rmsprop */
struct PT{ float*p,*g,*v; int n; };
static PT PL[512]; static int NPT=0;
static void reg(float*p,float*g,float*v,int n){ PL[NPT++] =(PT){p,g,v,n}; }

static void alloc_model(){
    NPT=0;
    E=dm(V*D);gE=dm(V*D);vE=dm(V*D); reg(E,gE,vE,V*D);
    lnfg=dm(D);glnfg=dm(D);vlnfg=dm(D); reg(lnfg,glnfg,vlnfg,D);
    lnfb=dm(D);glnfb=dm(D);vlnfb=dm(D); reg(lnfb,glnfb,vlnfb,D);
    for(int l=0;l<NL;l++){ Layer*P=&P_[l],*G=&Gr[l],*Vv=&Vs[l]; int dd=D*D,df=D*DFF;
        #define A3(f,n) P->f=dm(n);G->f=dm(n);Vv->f=dm(n);reg(P->f,G->f,Vv->f,n);
        A3(Wq,dd)A3(Wk,dd)A3(Wv,dd)A3(Wo,dd)A3(W1,df)A3(b1,DFF)A3(W2,df)A3(b2,D)A3(l1g,D)A3(l1b,D)A3(l2g,D)A3(l2b,D)
    }
    for(int l=0;l<=NL;l++)Hs[l]=dm(L*D);
    for(int l=0;l<NL;l++){ HR1[l]=dm(L*D);n1[l]=dm(L*D);m1[l]=dm(L);r1[l]=dm(L);m2[l]=dm(L);r2[l]=dm(L);qr[l]=dm(L*D);kr[l]=dm(L*D);vv[l]=dm(L*D);Pt[l]=dm(L*L);ao[l]=dm(L*D);n2[l]=dm(L*D);f1[l]=dm(L*DFF);gg[l]=dm(L*DFF); }
    nf=dm(L*D);mf=dm(L);rf=dm(L);logit=dm(L*V);
    dS=dm(L*L);dq=dm(L*D);dk=dm(L*D);dv=dm(L*D);dqr=dm(L*D);dkr=dm(L*D);dn=dm(L*D);dn2=dm(L*D);dtmp=dm(L*DFF);dtmp2=dm(L*DFF);dh=dm(L*D);dhr=dm(L*D);ddlog=dm(L*V);rowloss=dm(L);dnf=dm(L*D);qtmp=dm(L*D);ktmp=dm(L*D);dao=dm(L*D);daout=dm(L*D);dff=dm(L*DFF);
    dtg=(int*)dm(L);dmask=(int*)dm(L);
}
static void zero_grads(){ for(int i=0;i<NPT;i++) Z(PL[i].g,PL[i].n); }
static void opt_step(float lr){ for(int i=0;i<NPT;i++) krmsprop<<<(PL[i].n+255)/256,256>>>(PL[i].n,PL[i].p,PL[i].g,PL[i].v,lr,0.9f,1e-8f); }

/* ================= CHECKPOINT FORMAT (documented) =================
 * Little-endian binary, written/read by save_ckpt()/load_ckpt():
 *   char   magic[4]   = "GPT0"
 *   int32  version    = 1
 *   int32  V, D, DFF, NL, L      (model config; a reader must match these)
 *   int64  step                  (training step; 0 for a fresh/inference ckpt)
 *   float32 params[]  : every parameter tensor, in registration order
 *                       (E, lnfg, lnfb, then per layer l: Wq,Wk,Wv,Wo,W1,b1,
 *                        W2,b2,l1g,l1b,l2g,l2b), each row-major, sizes implied
 *                        by the config above.
 *   float32 optv[]    : RMSProp running-mean-square state, same order+sizes.
 * The REPL reads through the params block and ignores optv. Bit-exact
 * round-trip is proven by `gpt ckpttest`.                                    */
static int MAXTEN(){ int m=V*D; if(D*DFF>m)m=D*DFF; if(D*D>m)m=D*D; return m; }
static void save_ckpt(const char*path,long long step){
    FILE*f=fopen(path,"wb"); if(!f){printf("ckpt open fail %s\n",path);exit(1);}
    int ver=1; fwrite("GPT0",1,4,f); fwrite(&ver,4,1,f);
    fwrite(&V,4,1,f);fwrite(&D,4,1,f);fwrite(&DFF,4,1,f);fwrite(&NL,4,1,f);fwrite(&L,4,1,f);
    fwrite(&step,8,1,f);
    float*t=(float*)malloc((size_t)MAXTEN()*4);
    for(int i=0;i<NPT;i++){ CK(cudaMemcpy(t,PL[i].p,(size_t)PL[i].n*4,cudaMemcpyDeviceToHost)); fwrite(t,4,PL[i].n,f); }
    for(int i=0;i<NPT;i++){ CK(cudaMemcpy(t,PL[i].v,(size_t)PL[i].n*4,cudaMemcpyDeviceToHost)); fwrite(t,4,PL[i].n,f); }
    free(t); fclose(f);
}
static long long load_ckpt(const char*path,int want_optv){
    FILE*f=fopen(path,"rb"); if(!f){printf("ckpt open fail %s\n",path);exit(1);}
    char mg[4]; int ver,cv,cd,cf,cn,cl; long long step;
    if(fread(mg,1,4,f)!=4||memcmp(mg,"GPT0",4)){printf("bad ckpt magic\n");exit(1);}
    fread(&ver,4,1,f); fread(&cv,4,1,f);fread(&cd,4,1,f);fread(&cf,4,1,f);fread(&cn,4,1,f);fread(&cl,4,1,f); fread(&step,8,1,f);
    if(cv!=V||cd!=D||cf!=DFF||cn!=NL||cl!=L){ printf("ckpt config mismatch (%d %d %d %d %d)\n",cv,cd,cf,cn,cl); exit(1); }
    float*t=(float*)malloc((size_t)MAXTEN()*4);
    for(int i=0;i<NPT;i++){ if(fread(t,4,PL[i].n,f)!=(size_t)PL[i].n){printf("ckpt short read\n");exit(1);} CK(cudaMemcpy(PL[i].p,t,(size_t)PL[i].n*4,cudaMemcpyHostToDevice)); }
    if(want_optv) for(int i=0;i<NPT;i++){ if(fread(t,4,PL[i].n,f)!=(size_t)PL[i].n){printf("ckpt short read v\n");exit(1);} CK(cudaMemcpy(PL[i].v,t,(size_t)PL[i].n*4,cudaMemcpyHostToDevice)); }
    free(t); fclose(f); return step;
}

static void up(float*dev,const float*src,int n){ CK(cudaMemcpy(dev,src,(size_t)n*4,cudaMemcpyHostToDevice)); }
static void initw(float*dev,int n,double s){ float*t=(float*)malloc(n*4); for(int i=0;i<n;i++)t[i]=(float)((2*urand()-1)*s); up(dev,t,n); free(t); }
static void fillv(float*dev,int n,float val){ float*t=(float*)malloc(n*4); for(int i=0;i<n;i++)t[i]=val; up(dev,t,n); free(t); }
static void init_model(){
    initw(E,V*D,0.5/sqrt((double)D)); fillv(lnfg,D,1.f); Z(lnfb,D);
    for(int l=0;l<NL;l++){ Layer*P=&P_[l];
        initw(P->Wq,D*D,0.5/sqrt((double)D));initw(P->Wk,D*D,0.5/sqrt((double)D));initw(P->Wv,D*D,0.5/sqrt((double)D));initw(P->Wo,D*D,0.5/sqrt((double)D));
        initw(P->W1,D*DFF,0.5/sqrt((double)D));Z(P->b1,DFF);initw(P->W2,D*DFF,0.5/sqrt((double)DFF));Z(P->b2,D);
        fillv(P->l1g,D,1.f);Z(P->l1b,D);fillv(P->l2g,D,1.f);Z(P->l2b,D); }
    for(int i=0;i<NPT;i++)Z(PL[i].v,PL[i].n);
}

/* ---- forward: Xoh[L,V] one-hot on device; fills logit; returns loss ---- */
static double forward(const float*Xoh,const int*tg,const int*mask,int nact){
    GEMM(L,D,V,Xoh,0,E,0,Hs[0],0);
    for(int l=0;l<NL;l++){ Layer*P=&P_[l];
        kln_fwd<<<(L+63)/64,64>>>(Hs[l],P->l1g,n1[l],m1[l],r1[l],L,D);
        kaddb_fwd<<<(L*D+255)/256,256>>>(n1[l],P->l1b,n1[l],L,D);
        GEMM(L,D,D,n1[l],0,P->Wq,0,qtmp,0); GEMM(L,D,D,n1[l],0,P->Wk,0,ktmp,0); GEMM(L,D,D,n1[l],0,P->Wv,0,vv[l],0);
        krope<<<L,D/2>>>(qtmp,qr[l],L,D,RB,0); krope<<<L,D/2>>>(ktmp,kr[l],L,D,RB,0);
        kattn_fwd<<<(L+63)/64,64>>>(SCALE,qr[l],kr[l],vv[l],ao[l],Pt[l],L,D);
        GEMM(L,D,D,ao[l],0,P->Wo,0,dtmp,0);
        kadd<<<(L*D+255)/256,256>>>(Hs[l],dtmp,HR1[l],L*D);
        kln_fwd<<<(L+63)/64,64>>>(HR1[l],P->l2g,n2[l],m2[l],r2[l],L,D);
        kaddb_fwd<<<(L*D+255)/256,256>>>(n2[l],P->l2b,n2[l],L,D);
        GEMM(L,DFF,D,n2[l],0,P->W1,0,f1[l],0); kaddb_fwd<<<(L*DFF+255)/256,256>>>(f1[l],P->b1,f1[l],L,DFF);
        kgelu_fwd<<<(L*DFF+255)/256,256>>>(f1[l],gg[l],L*DFF);
        GEMM(L,D,DFF,gg[l],0,P->W2,0,dtmp,0); kaddb_fwd<<<(L*D+255)/256,256>>>(dtmp,P->b2,dtmp,L,D);
        kadd<<<(L*D+255)/256,256>>>(HR1[l],dtmp,Hs[l+1],L*D);
    }
    kln_fwd<<<(L+63)/64,64>>>(Hs[NL],lnfg,nf,mf,rf,L,D);
    kaddb_fwd<<<(L*D+255)/256,256>>>(nf,lnfb,nf,L,D);
    GEMM(L,V,D,nf,0,E,1,logit,0);
    kxent<<<(L+63)/64,64>>>(logit,tg,mask,nact,ddlog,rowloss,L,V);
    float rl[8192]; CK(cudaMemcpy(rl,rowloss,L*4,cudaMemcpyDeviceToHost)); double s=0; for(int i=0;i<L;i++)s+=rl[i]; return nact>0?s/nact:0;
}
/* ---- backward: assumes forward() just ran; ddlog holds dlogits; fills grads ---- */
static void backward(const float*Xoh){
    /* tied head: logit=nf@E^T ; dnf = ddlog@E ; dE += ddlog^T@nf */
    GEMM(L,D,V,ddlog,0,E,0,dnf,0);
    GEMM(V,D,L,ddlog,1,nf,0,gE,1);
    /* final LN (bias then gain). bias grad from dnf (addb), then gain LN bwd */
    kaddb_db<<<(D+255)/256,256>>>(dnf,glnfb,L,D);
    kln_bwd<<<(L+63)/64,64>>>(Hs[NL],lnfg,dnf,dh,glnfg,mf,rf,L,D);
    for(int l=NL-1;l>=0;l--){ Layer*P=&P_[l],*G=&Gr[l];
        /* h_{l+1} = hr1 + ffn ; dh is grad of h_{l+1}. residual: dhr1 = dh ; dff = dh */
        /* ffn: dtmp=g@W2+b2 ; db2 from dh ; dW2 += g^T dh ; dg = dh@W2^T */
        kaddb_db<<<(D+255)/256,256>>>(dh,G->b2,L,D);
        GEMM(DFF,D,L,gg[l],1,dh,0,G->W2,1);
        GEMM(L,DFF,D,dh,0,P->W2,1,dff,0);
        kgelu_bwd<<<(L*DFF+255)/256,256>>>(f1[l],dff,dff,L*DFF);   /* dff = dgelu */
        kaddb_db<<<(DFF+255)/256,256>>>(dff,G->b1,L,DFF);
        GEMM(D,DFF,L,n2[l],1,dff,0,G->W1,1);
        GEMM(L,D,DFF,dff,0,P->W1,1,dn2,0);                          /* dn2 = dff@W1^T */
        /* LN2 (bias then gain): db from dn2, then gain bwd -> dhr1 (add to residual dh) */
        kaddb_db<<<(D+255)/256,256>>>(dn2,G->l2b,L,D);
        kln_bwd<<<(L+63)/64,64>>>(HR1[l],P->l2g,dn2,dhr,G->l2g,m2[l],r2[l],L,D);
        kadd<<<(L*D+255)/256,256>>>(dh,dhr,dh,L*D);                 /* dh now = grad of hr1 */
        /* attn residual: hr1 = h_l + aout ; daout = dh ; dh_l accumulates dh */
        /* aout = ao@Wo : dWo += ao^T dh ; dao = dh@Wo^T */
        GEMM(D,D,L,ao[l],1,dh,0,G->Wo,1);
        GEMM(L,D,D,dh,0,P->Wo,1,dao,0);
        /* attention bwd: dqr,dkr,dv from dao (O=ao) */
        kattn_bwd_row<<<(L+63)/64,64>>>(SCALE,kr[l],vv[l],Pt[l],dao,dS,dqr,L,D);
        kattn_bwd_col<<<(L+63)/64,64>>>(qr[l],Pt[l],dao,dS,dkr,dv,L,D);
        krope<<<L,D/2>>>(dqr,dq,L,D,RB,1); krope<<<L,D/2>>>(dkr,dk,L,D,RB,1);  /* inverse-rotate grads */
        /* q=n1@Wq etc: dWq += n1^T dq ; dn1 = dq@Wq^T + dk@Wk^T + dv@Wv^T */
        GEMM(D,D,L,n1[l],1,dq,0,G->Wq,1); GEMM(D,D,L,n1[l],1,dk,0,G->Wk,1); GEMM(D,D,L,n1[l],1,dv,0,G->Wv,1);
        GEMM(L,D,D,dq,0,P->Wq,1,dn,0); GEMM(L,D,D,dk,0,P->Wk,1,dn,1); GEMM(L,D,D,dv,0,P->Wv,1,dn,1);
        /* LN1 (bias then gain): db from dn, gain bwd -> dtmp2 (grad wrt h_l); add to dh (residual) */
        kaddb_db<<<(D+255)/256,256>>>(dn,G->l1b,L,D);
        kln_bwd<<<(L+63)/64,64>>>(Hs[l],P->l1g,dn,dtmp,G->l1g,m1[l],r1[l],L,D);
        kadd<<<(L*D+255)/256,256>>>(dh,dtmp,dh,L*D);                /* dh now = grad of h_l */
    }
    /* embedding: Hs[0]=Xoh@E ; dE += Xoh^T@dh */
    GEMM(V,D,L,Xoh,1,dh,0,gE,1);
}

/* ================= host fp64 reference forward (loss only, for gradcheck) ==== */
/* reads device params into host doubles once (snapshot), computes loss in fp64. */
struct HL{ double *Wq,*Wk,*Wv,*Wo,*W1,*b1,*W2,*b2,*l1g,*l1b,*l2g,*l2b; };
static HL HP[16]; static double *hE,*hlnfg,*hlnfb;
static double* hm(int n){ return (double*)malloc(n*8); }
static void snap(){
    static int done=0; if(!done){ hE=hm(V*D);hlnfg=hm(D);hlnfb=hm(D);
        for(int l=0;l<NL;l++){ HP[l].Wq=hm(D*D);HP[l].Wk=hm(D*D);HP[l].Wv=hm(D*D);HP[l].Wo=hm(D*D);HP[l].W1=hm(D*DFF);HP[l].b1=hm(DFF);HP[l].W2=hm(D*DFF);HP[l].b2=hm(D);HP[l].l1g=hm(D);HP[l].l1b=hm(D);HP[l].l2g=hm(D);HP[l].l2b=hm(D);} done=1; }
    int mxn=D*DFF; if(V*D>mxn)mxn=V*D; if(D*D>mxn)mxn=D*D;
    float* t=(float*)malloc((size_t)mxn*4);
    #define GET(dev,host,n) CK(cudaMemcpy(t,dev,(size_t)(n)*4,cudaMemcpyDeviceToHost)); for(int q=0;q<(n);q++)host[q]=t[q];
    GET(E,hE,V*D) GET(lnfg,hlnfg,D) GET(lnfb,hlnfb,D)
    for(int l=0;l<NL;l++){ Layer*P=&P_[l];
        GET(P->Wq,HP[l].Wq,D*D)GET(P->Wk,HP[l].Wk,D*D)GET(P->Wv,HP[l].Wv,D*D)GET(P->Wo,HP[l].Wo,D*D)
        GET(P->W1,HP[l].W1,D*DFF)GET(P->b1,HP[l].b1,DFF)GET(P->W2,HP[l].W2,D*DFF)GET(P->b2,HP[l].b2,D)
        GET(P->l1g,HP[l].l1g,D)GET(P->l1b,HP[l].l1b,D)GET(P->l2g,HP[l].l2g,D)GET(P->l2b,HP[l].l2b,D) }
    free(t);
}
static void lnf(const double*X,const double*g,const double*b,double*Y,int R,int C){
    for(int i=0;i<R;i++){ double mu=0; for(int j=0;j<C;j++)mu+=X[i*C+j]; mu/=C; double var=0; for(int j=0;j<C;j++){double d=X[i*C+j]-mu;var+=d*d;} var/=C; double is=1.0/sqrt(var+1e-5);
        for(int j=0;j<C;j++)Y[i*C+j]=((X[i*C+j]-mu)*is)*g[j]+b[j]; } }
static double ref_loss(const int*x,const int*tg,const int*mask,int nact){
    double *h=(double*)malloc(L*D*8),*n=(double*)malloc(L*D*8),*q=(double*)malloc(L*D*8),*k=(double*)malloc(L*D*8),*v=(double*)malloc(L*D*8),*a=(double*)malloc(L*D*8),*f=(double*)malloc(L*DFF*8),*g2=(double*)malloc(L*DFF*8),*t=(double*)malloc(L*D*8);
    for(int i=0;i<L;i++)for(int c=0;c<D;c++) h[i*D+c]=hE[x[i]*D+c];
    for(int l=0;l<NL;l++){ HL*P=&HP[l];
        lnf(h,P->l1g,P->l1b,n,L,D);
        for(int i=0;i<L;i++)for(int c=0;c<D;c++){ double qq=0,kk=0,vvv=0; for(int b=0;b<D;b++){ qq+=n[i*D+b]*P->Wq[b*D+c]; kk+=n[i*D+b]*P->Wk[b*D+c]; vvv+=n[i*D+b]*P->Wv[b*D+c]; } q[i*D+c]=qq;k[i*D+c]=kk;v[i*D+c]=vvv; }
        for(int i=0;i<L;i++)for(int kk=0;kk<D/2;kk++){ double th=(double)i*pow(RB,-2.0*kk/D),c=cos(th),s=sin(th);
            double q0=q[i*D+2*kk],q1=q[i*D+2*kk+1]; q[i*D+2*kk]=q0*c-q1*s; q[i*D+2*kk+1]=q0*s+q1*c;
            double k0=k[i*D+2*kk],k1=k[i*D+2*kk+1]; k[i*D+2*kk]=k0*c-k1*s; k[i*D+2*kk+1]=k0*s+k1*c; }
        for(int i=0;i<L;i++){ double mx=-1e300,row[4096]; for(int j=0;j<=i;j++){ double s=0; for(int c=0;c<D;c++)s+=q[i*D+c]*k[j*D+c]; s*=SCALE; row[j]=s; if(s>mx)mx=s; }
            double se=0; for(int j=0;j<=i;j++){ row[j]=exp(row[j]-mx); se+=row[j]; }
            for(int c=0;c<D;c++){ double o=0; for(int j=0;j<=i;j++)o+=(row[j]/se)*v[j*D+c]; a[i*D+c]=o; } }
        for(int i=0;i<L;i++)for(int c=0;c<D;c++){ double o=0; for(int b=0;b<D;b++)o+=a[i*D+b]*P->Wo[b*D+c]; h[i*D+c]+=o; }
        lnf(h,P->l2g,P->l2b,n,L,D);
        for(int i=0;i<L;i++)for(int c=0;c<DFF;c++){ double s=P->b1[c]; for(int b=0;b<D;b++)s+=n[i*D+b]*P->W1[b*DFF+c]; f[i*DFF+c]=s; g2[i*DFF+c]=0.5*s*(1.0+erf(s*0.70710678118654752)); }
        for(int i=0;i<L;i++)for(int c=0;c<D;c++){ double s=P->b2[c]; for(int b=0;b<DFF;b++)s+=g2[i*DFF+b]*P->W2[b*D+c]; h[i*D+c]+=s; }
    }
    lnf(h,hlnfg,hlnfb,n,L,D);
    double loss=0; for(int i=0;i<L;i++){ if(!mask[i])continue; double lg[4096],mx=-1e300; for(int vv2=0;vv2<V;vv2++){ double s=0; for(int c=0;c<D;c++)s+=n[i*D+c]*hE[vv2*D+c]; lg[vv2]=s; if(s>mx)mx=s; }
        double se=0; for(int vv2=0;vv2<V;vv2++)se+=exp(lg[vv2]-mx); loss+=(mx+log(se))-lg[tg[i]]; }
    free(h);free(n);free(q);free(k);free(v);free(a);free(f);free(g2);free(t);
    return nact>0?loss/nact:0;
}
/* map param index p, offset j -> host-double slot (for finite-diff perturbation) */
static double* hslot(float* devp,int j){
    if(devp==E)return &hE[j]; if(devp==lnfg)return &hlnfg[j]; if(devp==lnfb)return &hlnfb[j];
    for(int l=0;l<NL;l++){ Layer*P=&P_[l]; HL*H=&HP[l];
        if(devp==P->Wq)return &H->Wq[j]; if(devp==P->Wk)return &H->Wk[j]; if(devp==P->Wv)return &H->Wv[j]; if(devp==P->Wo)return &H->Wo[j];
        if(devp==P->W1)return &H->W1[j]; if(devp==P->b1)return &H->b1[j]; if(devp==P->W2)return &H->W2[j]; if(devp==P->b2)return &H->b2[j];
        if(devp==P->l1g)return &H->l1g[j]; if(devp==P->l1b)return &H->l1b[j]; if(devp==P->l2g)return &H->l2g[j]; if(devp==P->l2b)return &H->l2b[j]; }
    return NULL;
}

/* ================= BPE tokenizer (load + encode + decode) for the REPL ====
 * Reads the "BPE1" file written by zero/bpe.c. Chat tokens are ordinary text
 * strings (e.g. "<|end|>"), so no special vocab is needed. */
struct BPair{ int a,b; };
static int TNM=0, TVOC=0; static BPair TMG[8192];
static unsigned char* TVB[8192]; static int TVL[8192];
static void load_tok(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){printf("no tokenizer %s\n",path);exit(1);} char mg[4];
    if(fread(mg,1,4,f)!=4||memcmp(mg,"BPE1",4)){printf("bad tok magic\n");exit(1);}
    fread(&TVOC,4,1,f); fread(&TNM,4,1,f);
    for(int i=0;i<TNM;i++){ fread(&TMG[i].a,4,1,f); fread(&TMG[i].b,4,1,f); } fclose(f);
    for(int i=0;i<256;i++){ TVB[i]=(unsigned char*)malloc(1); TVB[i][0]=(unsigned char)i; TVL[i]=1; }
    for(int k=0;k<TNM;k++){ int id=256+k,a=TMG[k].a,b=TMG[k].b,l=TVL[a]+TVL[b]; TVB[id]=(unsigned char*)malloc(l); memcpy(TVB[id],TVB[a],TVL[a]); memcpy(TVB[id]+TVL[a],TVB[b],TVL[b]); TVL[id]=l; }
}
static int bpe_encode(const char* s,int slen,int* out){
    int N=slen; for(int i=0;i<slen;i++)out[i]=(unsigned char)s[i];
    for(int k=0;k<TNM;k++){ int a=TMG[k].a,b=TMG[k].b,id=256+k,w=0; for(int i=0;i<N;){ if(i+1<N&&out[i]==a&&out[i+1]==b){out[w++]=id;i+=2;}else out[w++]=out[i++]; } N=w; }
    return N;
}
static void bpe_decode(const int* t,int n,char* out,int* olen){ int w=0; for(int i=0;i<n;i++){ memcpy(out+w,TVB[t[i]],TVL[t[i]]); w+=TVL[t[i]]; } out[w]=0; *olen=w; }

/* sample next token from a host logit row [V]: temperature + top-k */
static int sample_next(const float* lg,float temp,int topk){
    static int* idx=NULL; if(!idx) idx=(int*)malloc(V*sizeof(int));
    double* p=(double*)malloc(V*sizeof(double));
    for(int v=0;v<V;v++){ p[v]=lg[v]/(temp>1e-4?temp:1e-4); idx[v]=v; }
    /* top-k: partial selection of k largest */
    if(topk>0 && topk<V){ for(int i=0;i<topk;i++){ int mx=i; for(int j=i+1;j<V;j++) if(p[idx[j]]>p[idx[mx]])mx=j; int t=idx[i];idx[i]=idx[mx];idx[mx]=t; }
        double thr=p[idx[topk-1]]; for(int v=0;v<V;v++) if(p[v]<thr)p[v]=-1e30; }
    double mx=-1e300; for(int v=0;v<V;v++)if(p[v]>mx)mx=p[v]; double se=0; for(int v=0;v<V;v++){ p[v]=exp(p[v]-mx); se+=p[v]; }
    double r=urand()*se, c=0; int out=V-1; for(int v=0;v<V;v++){ c+=p[v]; if(c>=r){ out=v; break; } }
    free(p); return out;
}
/* run the transformer on the last min(len,L) tokens; return next-token logits (host) */
static float* GX=NULL; static float* GdX=NULL;
static void logits_at(const int* toks,int len,float* out){
    if(!GX){ GX=(float*)malloc((size_t)L*V*4); GdX=dm(L*V); }
    int k=len<L?len:L; const int* w=toks+(len-k);
    memset(GX,0,(size_t)L*V*4); for(int i=0;i<k;i++)GX[i*V+w[i]]=1.f;
    static int tg[512],ms[512]; for(int i=0;i<L;i++){tg[i]=0;ms[i]=0;}
    CK(cudaMemcpy(GdX,GX,(size_t)L*V*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtg,tg,L*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmask,ms,L*4,cudaMemcpyHostToDevice));
    forward(GdX,dtg,dmask,0);
    CK(cudaMemcpy(out,logit+(k-1)*V,(size_t)V*4,cudaMemcpyDeviceToHost));
}

static void gen_seq(int*x,int*tg,int*mask,int*nact,float*Xoh){
    for(int i=0;i<L;i++)x[i]=irnd(V);
    memset(Xoh,0,(size_t)L*V*4); for(int i=0;i<L;i++)Xoh[i*V+x[i]]=1.f;
    *nact=0; for(int i=0;i<L;i++){ if(i<L-1){ tg[i]=x[i+1]; mask[i]=1; (*nact)++; } else { tg[i]=0; mask[i]=0; } }
}

int main(int argc,char**argv){
    if(argc>=2 && !strcmp(argv[1],"check")){
        V=16;D=32;DFF=64;NL=2;L=12;SCALE=1.f/sqrtf((float)D); seed(20260707ULL); alloc_model();
        /* init params on host, upload */
        auto initw=[&](float*dev,int n,double sc){ float*t=(float*)malloc(n*4); for(int i=0;i<n;i++)t[i]=(float)((2*urand()-1)*sc); CK(cudaMemcpy(dev,t,(size_t)n*4,cudaMemcpyHostToDevice)); free(t); };
        auto ones=[&](float*dev,int n){ float*t=(float*)malloc(n*4); for(int i=0;i<n;i++)t[i]=1.f; CK(cudaMemcpy(dev,t,(size_t)n*4,cudaMemcpyHostToDevice)); free(t); };
        initw(E,V*D,0.5/sqrt((double)D)); ones(lnfg,D); Z(lnfb,D);
        for(int l=0;l<NL;l++){ Layer*P=&P_[l]; initw(P->Wq,D*D,0.5/sqrt((double)D));initw(P->Wk,D*D,0.5/sqrt((double)D));initw(P->Wv,D*D,0.5/sqrt((double)D));initw(P->Wo,D*D,0.5/sqrt((double)D));
            initw(P->W1,D*DFF,0.5/sqrt((double)D));Z(P->b1,DFF);initw(P->W2,D*DFF,0.5/sqrt((double)DFF));Z(P->b2,D); ones(P->l1g,D);Z(P->l1b,D);ones(P->l2g,D);Z(P->l2b,D); }
        for(int l=0;l<NPT;l++)Z(PL[l].v,PL[l].n);
        int x[64],tg[64],mask[64],nact; float Xoh[64*16]; float*dX=dm(L*V);
        gen_seq(x,tg,mask,&nact,Xoh); CK(cudaMemcpy(dX,Xoh,(size_t)L*V*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtg,tg,L*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmask,mask,L*4,cudaMemcpyHostToDevice));
        zero_grads(); double gl=forward(dX,dtg,dmask,nact); backward(dX); CK(cudaDeviceSynchronize());
        snap();
        /* gradcheck: fp32 analytic (device grads) vs fp64 finite diff */
        double worstabs=0,worstrel=0; double h=1e-4; int wp=-1; double wn=0,wa=0; int wsz=0;
        for(int pidx=0;pidx<NPT;pidx++){ float* dp=PL[pidx].p; float* dg=PL[pidx].g; int n=PL[pidx].n;
            for(int s=0;s<3;s++){ int j=irnd(n); double* hs=hslot(dp,j); if(!hs)continue;
                float ana; CK(cudaMemcpy(&ana,dg+j,4,cudaMemcpyDeviceToHost));
                double keep=*hs; *hs=keep+h; double Lp=ref_loss(x,tg,mask,nact); *hs=keep-h; double Lm=ref_loss(x,tg,mask,nact); *hs=keep;
                double num=(Lp-Lm)/(2*h); double ab=fabs(num-ana); if(ab>worstabs)worstabs=ab;
                if(fabs(num)+fabs(ana)>1e-3){ double rel=ab/(fabs(num)+fabs(ana)); if(rel>worstrel){worstrel=rel; wp=pidx; wn=num; wa=ana; wsz=n;} } } }
        printf("  [worst-rel param PL#%d (size %d): num=%.6e ana=%.6e]\n", wp, wsz, wn, wa);
        int gc = (worstabs<1e-3 && worstrel<1e-2);
        printf("GRADCHECK (fp32 analytic vs fp64 finite-diff): max-abs %.2e, max-rel(sig) %.2e -> %s\n", worstabs,worstrel, gc?"PASS":"FAIL");
        printf("(gpu loss %.6f vs fp64 ref %.6f)\n", gl, ref_loss(x,tg,mask,nact));
        /* loss parity: train GPU 100 steps; at each step compare the GPU-fp32 loss
         * to the fp64 reference loss on the SAME (evolving) weights. Confirms the
         * GPU forward stays fp64-consistent along the real optimization path. */
        double worstpar=0, firstL=0, lastL=0;
        for(int step=1;step<=100;step++){
            gen_seq(x,tg,mask,&nact,Xoh); CK(cudaMemcpy(dX,Xoh,(size_t)L*V*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dtg,tg,L*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmask,mask,L*4,cudaMemcpyHostToDevice));
            snap(); double rl=ref_loss(x,tg,mask,nact);
            zero_grads(); double gl=forward(dX,dtg,dmask,nact); backward(dX); opt_step(0.01f); CK(cudaDeviceSynchronize());
            double d=fabs(gl-rl); if(d>worstpar)worstpar=d; if(step==1)firstL=gl; lastL=gl;
        }
        int par = (worstpar<=2e-2);
        printf("LOSS PARITY (GPU-fp32 vs fp64 ref along 100 training steps): worst |d| %.2e -> %s (loss %.3f -> %.3f)\n",
               worstpar, par?"PASS":"FAIL", firstL, lastL);
        int pass = gc && par;
        printf("JSON {\"gradcheck_abs\":%.2e,\"gradcheck_rel\":%.2e,\"gradcheck_pass\":%s,\"parity_worst\":%.2e,\"parity_pass\":%s,\"gate_pass\":%s}\n",
               worstabs,worstrel,gc?"true":"false",worstpar,par?"true":"false",pass?"true":"false");
        return pass?0:1;
    }
    if(argc>=2 && !strcmp(argv[1],"ckpttest")){
        V=16;D=32;DFF=64;NL=2;L=12;SCALE=1.f/sqrtf((float)D); seed(42); alloc_model(); init_model();
        /* snapshot params+v to host A */
        int tot=0; for(int i=0;i<NPT;i++)tot+=PL[i].n;
        float*A=(float*)malloc((size_t)tot*2*4); int off=0;
        for(int i=0;i<NPT;i++){ CK(cudaMemcpy(A+off,PL[i].p,(size_t)PL[i].n*4,cudaMemcpyDeviceToHost)); off+=PL[i].n; }
        for(int i=0;i<NPT;i++){ CK(cudaMemcpy(A+off,PL[i].v,(size_t)PL[i].n*4,cudaMemcpyDeviceToHost)); off+=PL[i].n; }
        save_ckpt("/tmp/ckpt_test.bin", 12345);
        for(int i=0;i<NPT;i++){ Z(PL[i].p,PL[i].n); Z(PL[i].v,PL[i].n); }         /* corrupt */
        long long st=load_ckpt("/tmp/ckpt_test.bin",1);
        float*B=(float*)malloc((size_t)tot*2*4); off=0;
        for(int i=0;i<NPT;i++){ CK(cudaMemcpy(B+off,PL[i].p,(size_t)PL[i].n*4,cudaMemcpyDeviceToHost)); off+=PL[i].n; }
        for(int i=0;i<NPT;i++){ CK(cudaMemcpy(B+off,PL[i].v,(size_t)PL[i].n*4,cudaMemcpyDeviceToHost)); off+=PL[i].n; }
        int ok = (st==12345) && (memcmp(A,B,(size_t)tot*2*4)==0);
        printf("checkpoint round-trip: step %lld, %d floats (params+optv) -> %s\n", st, tot*2, ok?"BIT-EXACT":"MISMATCH");
        printf("JSON {\"ckpt_step\":%lld,\"ckpt_bitexact\":%s}\n", st, ok?"true":"false");
        return ok?0:1;
    }
    if(argc>=5 && !strcmp(argv[1],"train")){
        const char*tokf=argv[2]; const char*ckpt=argv[3]; long steps=atol(argv[4]); int resume=(argc>=6&&!strcmp(argv[5],"resume"));
        /* milestone config */
        V=2048;D=256;DFF=1024;NL=4;L=128;SCALE=1.f/sqrtf((float)D); int B=8; float lr=3e-4f;
        seed(777); alloc_model();
        /* load tokens (uint16) */
        FILE*tf=fopen(tokf,"rb"); if(!tf){printf("no tokfile %s\n",tokf);return 1;} fseek(tf,0,SEEK_END); long fb=ftell(tf); fseek(tf,0,SEEK_SET);
        long NT=fb/2; unsigned short*u=(unsigned short*)malloc(fb); if(fread(u,2,NT,tf)!=(size_t)NT){printf("tok read fail\n");return 1;} fclose(tf);
        int*toks=(int*)malloc(NT*sizeof(int)); for(long i=0;i<NT;i++)toks[i]=u[i]; free(u);
        long NVal=NT/10, NTrain=NT-NVal;
        printf("tokens: %ld (train %ld, val %ld), config V=%d D=%d DFF=%d NL=%d L=%d B=%d\n",NT,NTrain,NVal,V,D,DFF,NL,L,B);
        int np=0; for(int i=0;i<NPT;i++)np+=PL[i].n; printf("params: %d (%.2fM)\n",np,np/1e6);
        /* bigram baseline val CE (add-1 smoothing) */
        double bigCE=0; { long*rc=(long*)calloc(V,8); long*cc=(long*)calloc((size_t)V*V,8);
            for(long i=0;i<NTrain-1;i++){ cc[(long)toks[i]*V+toks[i+1]]++; rc[toks[i]]++; }
            double s=0; long cnt=0; for(long i=NTrain;i<NT-1;i++){ int a=toks[i],b=toks[i+1]; double p=(cc[(long)a*V+b]+1.0)/(rc[a]+(double)V); s+=-log(p); cnt++; }
            bigCE=cnt>0?s/cnt:0; free(rc);free(cc); }
        printf("bigram baseline val CE = %.4f nats  (bar: model must beat by >=0.3)\n", bigCE);
        long step0=0;
        if(resume){ FILE*c=fopen(ckpt,"rb"); if(c){fclose(c); step0=load_ckpt(ckpt,1); printf("RESUMED from step %ld\n",step0);} else { init_model(); printf("no ckpt to resume; fresh init\n"); } }
        else init_model();
        float*Xoh=(float*)malloc((size_t)L*V*4); float*dX=dm(L*V); int tg[512],mask[512];
        /* val eval helper (sample ns windows) */
        auto eval=[&](int ns)->double{ double s=0; int c=0; for(int q=0;q<ns;q++){ long p=NTrain+(long)(urand()*(NVal-L-1)); if(p<0)p=0;
            memset(Xoh,0,(size_t)L*V*4); int nact=0; for(int i=0;i<L;i++){ Xoh[i*V+toks[p+i]]=1.f; tg[i]=toks[p+i+1]; mask[i]=1; nact++; }
            CK(cudaMemcpy(dX,Xoh,(size_t)L*V*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtg,tg,L*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmask,mask,L*4,cudaMemcpyHostToDevice));
            s+=forward(dX,dtg,dmask,nact); c++; } return c?s/c:0; };
        double t_ema=0;
        for(long step=step0+1; step<=steps; step++){
            zero_grads(); double tl=0;
            for(int b=0;b<B;b++){ long p=(long)(urand()*(NTrain-L-1)); if(p<0)p=0;
                memset(Xoh,0,(size_t)L*V*4); int nact=0; for(int i=0;i<L;i++){ Xoh[i*V+toks[p+i]]=1.f; tg[i]=toks[p+i+1]; mask[i]=1; nact++; }
                CK(cudaMemcpy(dX,Xoh,(size_t)L*V*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtg,tg,L*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dmask,mask,L*4,cudaMemcpyHostToDevice));
                tl+=forward(dX,dtg,dmask,nact); backward(dX); }
            opt_step(lr); CK(cudaDeviceSynchronize());
            tl/=B; t_ema = t_ema==0? tl : 0.98*t_ema+0.02*tl;
            if(step%100==0){ printf("step %ld/%ld  train %.4f (ema %.4f)\n",step,steps,tl,t_ema); fflush(stdout); }
            if(step%1000==0 || step==steps){ double vl=eval(64); printf("  [step %ld] VAL %.4f  (bigram %.4f, gap %.4f)\n",step,vl,bigCE,bigCE-vl); fflush(stdout);
                save_ckpt(ckpt,step); }
        }
        double vl=eval(256); printf("\nFINAL val CE %.4f  bigram %.4f  gap %.4f  -> %s\n",vl,bigCE,bigCE-vl,(bigCE-vl>=0.3)?"BEATS BASELINE (bar met)":"below bar");
        printf("JSON {\"final_val_ce\":%.4f,\"bigram_ce\":%.4f,\"gap\":%.4f,\"bar_met\":%s}\n",vl,bigCE,bigCE-vl,(bigCE-vl>=0.3)?"true":"false");
        save_ckpt(ckpt,steps);
        return 0;
    }
    if(argc>=4 && !strcmp(argv[1],"talk")){
        const char*ckpt=argv[2]; const char*tokf=argv[3];
        float temp=argc>=5?atof(argv[4]):0.8f; int topk=argc>=6?atoi(argv[5]):40;
        V=2048;D=256;DFF=1024;NL=4;L=128;SCALE=1.f/sqrtf((float)D); seed(12345); alloc_model();
        load_ckpt(ckpt,0); load_tok(tokf);
        static int hist[16384]; int hlen=0; char line[8192];
        fprintf(stderr,"loom chat REPL (temp %.2f, top-k %d). Type a message; Ctrl-D to quit.\n",temp,topk);
        while(fgets(line,sizeof(line),stdin)){
            int n=strlen(line); while(n>0&&(line[n-1]=='\n'||line[n-1]=='\r'))line[--n]=0;
            if(n==0) continue;
            char buf[16384]; snprintf(buf,sizeof(buf),"<|user|>\n%s\n<|assistant|>\n",line);
            static int enc[16384]; int ne=bpe_encode(buf,strlen(buf),enc);
            for(int i=0;i<ne&&hlen<15000;i++)hist[hlen++]=enc[i];
            int gstart=hlen; static char dec[65536]; int dl=0;
            for(int step=0;step<220 && hlen<15500;step++){
                float lg[2048]; logits_at(hist,hlen,lg);
                int nt=sample_next(lg,temp,topk); hist[hlen++]=nt;
                bpe_decode(hist+gstart,hlen-gstart,dec,&dl);
                if(strstr(dec,"<|end|>"))break;
            }
            char* e=strstr(dec,"<|end|>"); if(e)*e=0;
            printf("%s\n",dec); fflush(stdout);
        }
        return 0;
    }
    printf("usage: gpt check | gpt ckpttest | gpt train <tok> <ckpt> <steps> [resume] | gpt talk <ckpt> <tok> [temp] [topk]\n"); return 2;
}
