/*
 * day4.c — byte discipline, part a: the ZERO engine ported to fp32, with a
 * dual-precision gradcheck.
 *
 * Foundations: hardware, OS, C compiler, mathematics. No ML/numerics libs.
 * The forward+backward here run in float (fp32) — the precision the GPU path
 * will use. To prove the fp32 backward is correct we compare its ANALYTIC
 * gradients against FINITE DIFFERENCES computed in fp64 by an independent
 * double-precision forward (ref_loss_f64), evaluated at the exact fp32 point
 * (double represents any float exactly). Two things are proven at once: the
 * fp32 backward matches calculus, and the fp32 forward matches a separate
 * fp64 forward.
 *
 * Bars (pre-registered, ZERO.md): fp32 analytic vs fp64 finite-diff max rel
 * err < 1e-3 on significant-gradient entries; the fp64 reference build
 * (zero/day3.c) still hits < 1e-6.
 *
 * Compile: gcc -O2 -std=c11 -Wall -Wno-misleading-indentation -o day4 day4.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef float f32;
/* ===================== fp32 engine ===================================== */
static unsigned long long RNG=0x9E3779B97F4A7C15ULL;
static void rng_seed(unsigned long long s){ RNG=s?s:0x9E3779B97F4A7C15ULL; }
static f32 rnd(void){ RNG^=RNG>>12; RNG^=RNG<<25; RNG^=RNG>>27;
    return (f32)((double)((RNG*0x2545F4914F6CDD1DULL)>>11)/(double)(1ULL<<53)); }
static int irand(int n){ return (int)((double)rnd()*n); }

typedef struct T T;
struct T { int r,c; f32 *d,*g; T *s0,*s1; void(*bw)(T*); int *it; f32 *aux; int *msk; f32 p0; int vis; };
#define MAXN 200000
static T *NODES[MAXN]; static int NN=0;
static T *PARAMS[64];  static int NP=0;
static T *t_alloc(int r,int c){ T*t=calloc(1,sizeof(T)); t->r=r;t->c=c;
    t->d=calloc((size_t)r*c,sizeof(f32)); t->g=calloc((size_t)r*c,sizeof(f32)); return t; }
static T *t_node(int r,int c){ T*t=t_alloc(r,c); NODES[NN++]=t; return t; }
static T *t_param(int r,int c,f32 scale){ T*t=t_alloc(r,c);
    for(int i=0;i<r*c;i++) t->d[i]=scale*(2.0f*rnd()-1.0f); PARAMS[NP++]=t; return t; }
static void arena_reset(void){ for(int i=0;i<NN;i++){ free(NODES[i]->d);free(NODES[i]->g);
    free(NODES[i]->it);free(NODES[i]->aux);free(NODES[i]->msk);free(NODES[i]); } NN=0; }

static void bw_matmul(T*y){ T*A=y->s0,*B=y->s1;
    for(int i=0;i<A->r;i++) for(int k=0;k<A->c;k++){ f32 s=0;
        for(int j=0;j<y->c;j++) s+=y->g[i*y->c+j]*B->d[k*B->c+j]; A->g[i*A->c+k]+=s; }
    for(int k=0;k<B->r;k++) for(int j=0;j<B->c;j++){ f32 s=0;
        for(int i=0;i<A->r;i++) s+=A->d[i*A->c+k]*y->g[i*y->c+j]; B->g[k*B->c+j]+=s; } }
static T *t_matmul(T*A,T*B){ T*y=t_node(A->r,B->c);
    for(int i=0;i<A->r;i++) for(int j=0;j<B->c;j++){ f32 s=0;
        for(int k=0;k<A->c;k++) s+=A->d[i*A->c+k]*B->d[k*B->c+j]; y->d[i*y->c+j]=s; }
    y->s0=A;y->s1=B;y->bw=bw_matmul; return y; }
static void bw_addb(T*y){ T*X=y->s0,*b=y->s1;
    for(int i=0;i<y->r*y->c;i++) X->g[i]+=y->g[i];
    for(int j=0;j<y->c;j++){ f32 s=0; for(int i=0;i<y->r;i++) s+=y->g[i*y->c+j]; b->g[j]+=s; } }
static T *t_addb(T*X,T*b){ T*y=t_node(X->r,X->c);
    for(int i=0;i<X->r;i++) for(int j=0;j<X->c;j++) y->d[i*X->c+j]=X->d[i*X->c+j]+b->d[j];
    y->s0=X;y->s1=b;y->bw=bw_addb; return y; }
static void bw_transpose(T*y){ T*A=y->s0;
    for(int i=0;i<A->r;i++) for(int j=0;j<A->c;j++) A->g[i*A->c+j]+=y->g[j*y->c+i]; }
static T *t_transpose(T*A){ T*y=t_node(A->c,A->r);
    for(int i=0;i<A->r;i++) for(int j=0;j<A->c;j++) y->d[j*y->c+i]=A->d[i*A->c+j];
    y->s0=A;y->bw=bw_transpose; return y; }
static void bw_scale(T*y){ T*A=y->s0; f32 s=y->p0; for(int i=0;i<A->r*A->c;i++) A->g[i]+=s*y->g[i]; }
static T *t_scale(T*A,f32 s){ T*y=t_node(A->r,A->c);
    for(int i=0;i<A->r*A->c;i++) y->d[i]=s*A->d[i]; y->s0=A;y->p0=s;y->bw=bw_scale; return y; }
#define NEG (-1e30f)
static void bw_cmask(T*y){ T*A=y->s0;
    for(int i=0;i<y->r;i++) for(int j=0;j<y->c;j++) if(j<=i) A->g[i*y->c+j]+=y->g[i*y->c+j]; }
static T *t_cmask(T*A){ T*y=t_node(A->r,A->c);
    for(int i=0;i<A->r;i++) for(int j=0;j<A->c;j++) y->d[i*A->c+j]=(j<=i)?A->d[i*A->c+j]:NEG;
    y->s0=A;y->bw=bw_cmask; return y; }
static void bw_rowsm(T*y){ T*A=y->s0; int R=y->r,C=y->c;
    for(int i=0;i<R;i++){ f32 dot=0; for(int j=0;j<C;j++) dot+=y->aux[i*C+j]*y->g[i*C+j];
        for(int j=0;j<C;j++) A->g[i*C+j]+=y->aux[i*C+j]*(y->g[i*C+j]-dot); } }
static T *t_rowsm(T*A){ T*y=t_node(A->r,A->c); int R=A->r,C=A->c;
    y->aux=malloc((size_t)R*C*sizeof(f32));
    for(int i=0;i<R;i++){ f32 mx=A->d[i*C]; for(int j=1;j<C;j++) if(A->d[i*C+j]>mx) mx=A->d[i*C+j];
        f32 se=0; for(int j=0;j<C;j++) se+=expf(A->d[i*C+j]-mx);
        for(int j=0;j<C;j++){ f32 p=expf(A->d[i*C+j]-mx)/se; y->d[i*C+j]=p; y->aux[i*C+j]=p; } }
    y->s0=A;y->bw=bw_rowsm; return y; }
static double ROPE_BASE=10000.0; static int USE_ROPE=1;
static void bw_rope(T*y){ T*X=y->s0; int R=y->r,C=y->c;
    for(int i=0;i<R;i++) for(int k=0;k<C/2;k++){
        double th=(double)i*pow(ROPE_BASE,-2.0*k/C); f32 c=cosf((f32)th),s=sinf((f32)th);
        f32 d0=y->g[i*C+2*k], d1=y->g[i*C+2*k+1];
        X->g[i*C+2*k]+=d0*c+d1*s; X->g[i*C+2*k+1]+=-d0*s+d1*c; } }
static T *t_rope(T*X){ T*y=t_node(X->r,X->c); int R=X->r,C=X->c;
    for(int i=0;i<R;i++) for(int k=0;k<C/2;k++){
        double th=(double)i*pow(ROPE_BASE,-2.0*k/C); f32 c=cosf((f32)th),s=sinf((f32)th);
        f32 x0=X->d[i*C+2*k], x1=X->d[i*C+2*k+1];
        y->d[i*C+2*k]=x0*c-x1*s; y->d[i*C+2*k+1]=x0*s+x1*c; }
    y->s0=X;y->bw=bw_rope; return y; }
static T *t_xent_masked(T*z,const int*tg,const int*mask){ int R=z->r,Vv=z->c; T*L=t_node(1,1);
    L->s0=z;L->it=malloc(R*sizeof(int));L->msk=malloc(R*sizeof(int));
    L->aux=malloc((size_t)R*Vv*sizeof(f32));
    memcpy(L->it,tg,R*sizeof(int)); memcpy(L->msk,mask,R*sizeof(int));
    int nact=0; f32 total=0;
    for(int i=0;i<R;i++){ f32 mx=z->d[i*Vv]; for(int j=1;j<Vv;j++) if(z->d[i*Vv+j]>mx) mx=z->d[i*Vv+j];
        f32 se=0; for(int j=0;j<Vv;j++) se+=expf(z->d[i*Vv+j]-mx);
        for(int j=0;j<Vv;j++) L->aux[i*Vv+j]=expf(z->d[i*Vv+j]-mx)/se;
        if(mask[i]){ total+=(mx+logf(se))-z->d[i*Vv+tg[i]]; nact++; } }
    L->p0=(nact>0)?(1.0f/nact):0.0f; L->d[0]=(nact>0)?total/nact:0.0f;
    extern void bw_xent_masked(T*); L->bw=bw_xent_masked; return L; }
void bw_xent_masked(T*L){ T*z=L->s0; int R=z->r,Vv=z->c; f32 inv=L->p0;
    for(int i=0;i<R;i++) if(L->msk[i]) for(int j=0;j<Vv;j++)
        z->g[i*Vv+j]+=(L->aux[i*Vv+j]-(j==L->it[i]?1.0f:0.0f))*inv*L->g[0]; }
static T *TOPO[MAXN]; static int NT=0;
static void dfs(T*t){ if(!t||t->vis) return; t->vis=1; dfs(t->s0); dfs(t->s1); TOPO[NT++]=t; }
static void backward(T*loss){ NT=0; for(int i=0;i<NN;i++) NODES[i]->vis=0;
    for(int i=0;i<NP;i++) PARAMS[i]->vis=0; dfs(loss); loss->g[0]=1.0f;
    for(int i=NT-1;i>=0;i--) if(TOPO[i]->bw) TOPO[i]->bw(TOPO[i]); }
static void zero_grads(void){ for(int i=0;i<NP;i++)
    memset(PARAMS[i]->g,0,(size_t)PARAMS[i]->r*PARAMS[i]->c*sizeof(f32)); }

/* ===================== fp32 attention model =========================== */
#define VOC 8
#define L 24
#define D 32
static T *X_in;
static T *E,*Wq,*Wk,*Wv,*Wout,*bout;
static void build_model(void){
    for(int i=0;i<NP;i++){ free(PARAMS[i]->d);free(PARAMS[i]->g);free(PARAMS[i]); } NP=0;
    E=t_param(2*VOC,D,0.5f/sqrtf(2*VOC)); Wv=t_param(D,D,0.5f/sqrtf(D));
    Wout=t_param(D,VOC,0.5f/sqrtf(D)); bout=t_param(1,VOC,0.0f);
    Wq=t_param(D,D,0.5f/sqrtf(D)); Wk=t_param(D,D,0.5f/sqrtf(D));
}
static T *forward_logits(void){
    T *Xe=t_matmul(X_in,E), *Vv=t_matmul(Xe,Wv);
    T *Qm=t_matmul(Xe,Wq), *Km=t_matmul(Xe,Wk);
    T *Q=USE_ROPE?t_rope(Qm):Qm, *K=USE_ROPE?t_rope(Km):Km;
    T *S=t_scale(t_matmul(Q,t_transpose(K)),1.0f/sqrtf(D));
    T *P=t_rowsm(t_cmask(S));
    T *O=t_matmul(P,Vv);
    return t_addb(t_matmul(O,Wout),bout);
}
static int SEQ[L];
static void make_sequence(int *tg,int *mask){
    for(int i=0;i<L;i++) SEQ[i]=irand(VOC);
    memset(X_in->d,0,(size_t)L*2*VOC*sizeof(f32));
    for(int i=0;i<L;i++){ X_in->d[i*2*VOC+SEQ[i]]=1.0f;
        if(i>0) X_in->d[i*2*VOC+VOC+SEQ[i-1]]=1.0f; }
    for(int i=0;i<L;i++){ tg[i]=0; mask[i]=0; int prev=-1;
        for(int j=i-1;j>=0;j--) if(SEQ[j]==SEQ[i]){ prev=j; break; }
        if(prev>=0){ tg[i]=SEQ[prev+1]; mask[i]=1; } }
}

/* ===================== fp64 forward-only REFERENCE ====================== */
/* Replicates forward_logits + masked xent entirely in double, reading double
 * copies of the params. Used only to produce fp64 finite-difference grads. */
static double dE[2*VOC*D],dWv[D*D],dWq[D*D],dWk[D*D],dWout[D*VOC],dbout[VOC];
static void snapshot_params_to_double(void){
    for(int i=0;i<2*VOC*D;i++) dE[i]=(double)E->d[i];
    for(int i=0;i<D*D;i++){ dWv[i]=(double)Wv->d[i]; dWq[i]=(double)Wq->d[i]; dWk[i]=(double)Wk->d[i]; }
    for(int i=0;i<D*VOC;i++) dWout[i]=(double)Wout->d[i];
    for(int i=0;i<VOC;i++) dbout[i]=(double)bout->d[i];
}
static double ref_loss_f64(const int*tg,const int*mask){
    double Xe[L*D], Vv[L*D], Q[L*D], K[L*D], O[L*D], logit[L*VOC];
    /* Xe = X_in @ E ; Vv = Xe@Wv ; Q=Xe@Wq ; K=Xe@Wk */
    for(int i=0;i<L;i++) for(int a=0;a<D;a++){ double xe=0;
        for(int f=0;f<2*VOC;f++){ double x=(double)X_in->d[i*2*VOC+f]; if(x!=0) xe+=x*dE[f*D+a]; }
        Xe[i*D+a]=xe; }
    for(int i=0;i<L;i++) for(int a=0;a<D;a++){ double vv=0,q=0,kk=0;
        for(int b=0;b<D;b++){ double xe=Xe[i*D+b]; vv+=xe*dWv[b*D+a]; q+=xe*dWq[b*D+a]; kk+=xe*dWk[b*D+a]; }
        Vv[i*D+a]=vv; Q[i*D+a]=q; K[i*D+a]=kk; }
    if(USE_ROPE){ for(int i=0;i<L;i++) for(int k=0;k<D/2;k++){
        double th=(double)i*pow(ROPE_BASE,-2.0*k/D), c=cos(th), s=sin(th);
        double q0=Q[i*D+2*k],q1=Q[i*D+2*k+1]; Q[i*D+2*k]=q0*c-q1*s; Q[i*D+2*k+1]=q0*s+q1*c;
        double k0=K[i*D+2*k],k1=K[i*D+2*k+1]; K[i*D+2*k]=k0*c-k1*s; K[i*D+2*k+1]=k0*s+k1*c; } }
    /* scores, causal mask, softmax, O=P@Vv */
    for(int i=0;i<L;i++){ double row[L], mx=-1e300;
        for(int j=0;j<L;j++){ if(j<=i){ double s=0; for(int a=0;a<D;a++) s+=Q[i*D+a]*K[j*D+a];
            row[j]=s/sqrt((double)D); } else row[j]=-1e300; if(row[j]>mx) mx=row[j]; }
        double se=0; for(int j=0;j<L;j++){ row[j]=exp(row[j]-mx); se+=row[j]; }
        for(int a=0;a<D;a++){ double o=0; for(int j=0;j<L;j++) o+=(row[j]/se)*Vv[j*D+a]; O[i*D+a]=o; } }
    /* logits = O@Wout + bout */
    for(int i=0;i<L;i++) for(int v=0;v<VOC;v++){ double s=dbout[v];
        for(int a=0;a<D;a++) s+=O[i*D+a]*dWout[a*VOC+v]; logit[i*VOC+v]=s; }
    /* masked cross-entropy */
    int nact=0; double total=0;
    for(int i=0;i<L;i++){ if(!mask[i]) continue; double mx=logit[i*VOC];
        for(int v=1;v<VOC;v++) if(logit[i*VOC+v]>mx) mx=logit[i*VOC+v];
        double se=0; for(int v=0;v<VOC;v++) se+=exp(logit[i*VOC+v]-mx);
        total+=(mx+log(se))-logit[i*VOC+tg[i]]; nact++; }
    return nact? total/nact : 0.0;
}

/* map a global param index -> the matching double-array slot for perturbation */
static double* dslot(int pidx,int j){
    if(PARAMS[pidx]==E)   return &dE[j];
    if(PARAMS[pidx]==Wv)  return &dWv[j];
    if(PARAMS[pidx]==Wout)return &dWout[j];
    if(PARAMS[pidx]==bout)return &dbout[j];
    if(PARAMS[pidx]==Wq)  return &dWq[j];
    if(PARAMS[pidx]==Wk)  return &dWk[j];
    return NULL;
}

/* ===================== dual-precision gradcheck ======================== */
static int LTG[L],LMASK[L];
static void dual_gradcheck(double*out_abs,double*out_relsig){
    rng_seed(7ULL); build_model(); make_sequence(LTG,LMASK);
    for(int j=0;j<Wq->r*Wq->c;j++){ Wq->d[j]*=6.0f; Wk->d[j]*=6.0f; } /* sharpen off degenerate init */
    zero_grads(); T*z=forward_logits(); T*Lz=t_xent_masked(z,LTG,LMASK);
    backward(Lz); arena_reset();
    snapshot_params_to_double();
    double worstabs=0, worstrelsig=0; double h=1e-4;
    for(int p=0;p<NP;p++){ T*par=PARAMS[p]; int tot=par->r*par->c;
        for(int s=0;s<12;s++){ int j=irand(tot); double*ds=dslot(p,j); if(!ds) continue;
            double keep=*ds;
            *ds=keep+h; double Lp=ref_loss_f64(LTG,LMASK);
            *ds=keep-h; double Lm=ref_loss_f64(LTG,LMASK); *ds=keep;
            double num=(Lp-Lm)/(2*h);            /* fp64 finite difference */
            double ana=(double)par->g[j];        /* fp32 analytic */
            double ab=fabs(num-ana); if(ab>worstabs) worstabs=ab;
            if(fabs(num)+fabs(ana)>1e-3){ double rel=ab/(fabs(num)+fabs(ana));
                if(rel>worstrelsig) worstrelsig=rel; } } }
    *out_abs=worstabs; *out_relsig=worstrelsig;
}

/* fp32 vs fp64 forward loss agreement (sanity that the two forwards match) */
static double forward_agreement(void){
    rng_seed(101ULL); build_model(); make_sequence(LTG,LMASK);
    T*z=forward_logits(); T*Lz=t_xent_masked(z,LTG,LMASK); double lf32=(double)Lz->d[0]; arena_reset();
    snapshot_params_to_double(); double lf64=ref_loss_f64(LTG,LMASK);
    return fabs(lf32-lf64);
}

int main(void){
    X_in=t_alloc(L,2*VOC);
    printf("=== Day 4a: fp32 engine port + dual-precision gradcheck ===\n");
    double fa=forward_agreement();
    printf("fp32 vs fp64 forward loss agreement: |Δ| = %.3e\n", fa);
    double gabs,grel; dual_gradcheck(&gabs,&grel);
    int pass = (grel<1e-3);
    printf("DUAL GRADCHECK (fp32 analytic vs fp64 finite-diff):\n");
    printf("  max-abs err = %.3e ; max-rel err (significant grads) = %.3e\n", gabs, grel);
    printf("  bar rel<1e-3 -> %s\n", pass?"PASS":"FAIL");
    printf("\nJSON {\"fwd_agreement\":%.3e,\"gradcheck_abs\":%.3e,\"gradcheck_relsig\":%.3e,\"bar\":\"rel<1e-3\",\"pass\":%s}\n",
           fa, gabs, grel, pass?"true":"false");
    return pass?0:1;
}
