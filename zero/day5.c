/*
 * day5.c — the full transformer LM, assembled in the ZERO autodiff engine.
 * ARTIFACT (pure C, no libraries). Config-driven: embeddings (tied with the
 * output head), N x [pre-LayerNorm causal attention (RoPE) + residual;
 * pre-LayerNorm FFN (GELU) + residual], final LayerNorm, tied head, masked
 * next-token cross-entropy. Proofs: gradcheck (fp64 analytic vs central finite
 * differences < 1e-6) and BIT-EXACT causality on the assembled stack.
 *
 * Engine = day1..day4 machinery, plus three new ops with hand-derived backward
 * (add, gelu, layernorm), each covered by the whole-stack gradcheck.
 *
 * Build: gcc -O2 -std=c11 -Wall -Wno-misleading-indentation -o day5 day5.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned long long RNG=0x9E3779B97F4A7C15ULL;
static void seed(unsigned long long s){ RNG=s?s:0x9E3779B97F4A7C15ULL; }
static double rnd(void){ RNG^=RNG>>12; RNG^=RNG<<25; RNG^=RNG>>27; return (double)((RNG*0x2545F4914F6CDD1DULL)>>11)/(double)(1ULL<<53); }
static int irand(int n){ return (int)(rnd()*n); }

typedef struct T T;
struct T{ int r,c; double *d,*g; T *s0,*s1; void(*bw)(T*); int *it; double *aux; int *msk; double p0,p1; int vis; };
#define MAXN 400000
static T *NODES[MAXN]; static int NN=0;
static T *PARAMS[512]; static int NP=0;
static T *t_alloc(int r,int c){ T*t=calloc(1,sizeof(T)); t->r=r;t->c=c; t->d=calloc((size_t)r*c,sizeof(double)); t->g=calloc((size_t)r*c,sizeof(double)); return t; }
static T *t_node(int r,int c){ T*t=t_alloc(r,c); NODES[NN++]=t; return t; }
static T *t_param(int r,int c,double sc){ T*t=t_alloc(r,c); for(int i=0;i<r*c;i++) t->d[i]=sc*(2.0*rnd()-1.0); PARAMS[NP++]=t; return t; }
static void arena_reset(void){ for(int i=0;i<NN;i++){ free(NODES[i]->d);free(NODES[i]->g);free(NODES[i]->it);free(NODES[i]->aux);free(NODES[i]->msk);free(NODES[i]); } NN=0; }

/* matmul */
static void bw_mm(T*y){ T*A=y->s0,*B=y->s1;
    for(int i=0;i<A->r;i++)for(int k=0;k<A->c;k++){ double s=0; for(int j=0;j<y->c;j++)s+=y->g[i*y->c+j]*B->d[k*B->c+j]; A->g[i*A->c+k]+=s; }
    for(int k=0;k<B->r;k++)for(int j=0;j<B->c;j++){ double s=0; for(int i=0;i<A->r;i++)s+=A->d[i*A->c+k]*y->g[i*y->c+j]; B->g[k*B->c+j]+=s; } }
static T* mm(T*A,T*B){ T*y=t_node(A->r,B->c); for(int i=0;i<A->r;i++)for(int j=0;j<B->c;j++){ double s=0; for(int k=0;k<A->c;k++)s+=A->d[i*A->c+k]*B->d[k*B->c+j]; y->d[i*y->c+j]=s; } y->s0=A;y->s1=B;y->bw=bw_mm; return y; }
/* add bias [1,n] */
static void bw_addb(T*y){ T*X=y->s0,*b=y->s1; for(int i=0;i<y->r*y->c;i++)X->g[i]+=y->g[i]; for(int j=0;j<y->c;j++){ double s=0; for(int i=0;i<y->r;i++)s+=y->g[i*y->c+j]; b->g[j]+=s; } }
static T* addb(T*X,T*b){ T*y=t_node(X->r,X->c); for(int i=0;i<X->r;i++)for(int j=0;j<X->c;j++)y->d[i*X->c+j]=X->d[i*X->c+j]+b->d[j]; y->s0=X;y->s1=b;y->bw=bw_addb; return y; }
/* elementwise add (same shape) */
static void bw_add(T*y){ T*a=y->s0,*b=y->s1; for(int i=0;i<y->r*y->c;i++){ a->g[i]+=y->g[i]; b->g[i]+=y->g[i]; } }
static T* add(T*a,T*b){ T*y=t_node(a->r,a->c); for(int i=0;i<a->r*a->c;i++)y->d[i]=a->d[i]+b->d[i]; y->s0=a;y->s1=b;y->bw=bw_add; return y; }
/* transpose */
static void bw_tr(T*y){ T*A=y->s0; for(int i=0;i<A->r;i++)for(int j=0;j<A->c;j++)A->g[i*A->c+j]+=y->g[j*y->c+i]; }
static T* tr(T*A){ T*y=t_node(A->c,A->r); for(int i=0;i<A->r;i++)for(int j=0;j<A->c;j++)y->d[j*y->c+i]=A->d[i*A->c+j]; y->s0=A;y->bw=bw_tr; return y; }
/* scale const */
static void bw_sc(T*y){ T*A=y->s0; double s=y->p0; for(int i=0;i<A->r*A->c;i++)A->g[i]+=s*y->g[i]; }
static T* sc(T*A,double s){ T*y=t_node(A->r,A->c); for(int i=0;i<A->r*A->c;i++)y->d[i]=s*A->d[i]; y->s0=A;y->p0=s;y->bw=bw_sc; return y; }
/* causal mask */
#define NEG (-1e30)
static void bw_cm(T*y){ T*A=y->s0; for(int i=0;i<y->r;i++)for(int j=0;j<y->c;j++) if(j<=i) A->g[i*y->c+j]+=y->g[i*y->c+j]; }
static T* cm(T*A){ T*y=t_node(A->r,A->c); for(int i=0;i<A->r;i++)for(int j=0;j<A->c;j++)y->d[i*A->c+j]=(j<=i)?A->d[i*A->c+j]:NEG; y->s0=A;y->bw=bw_cm; return y; }
/* row softmax */
static void bw_sm(T*y){ T*A=y->s0; int R=y->r,C=y->c; for(int i=0;i<R;i++){ double dot=0; for(int j=0;j<C;j++)dot+=y->aux[i*C+j]*y->g[i*C+j]; for(int j=0;j<C;j++)A->g[i*C+j]+=y->aux[i*C+j]*(y->g[i*C+j]-dot); } }
static T* sm(T*A){ T*y=t_node(A->r,A->c); int R=A->r,C=A->c; y->aux=malloc((size_t)R*C*sizeof(double)); for(int i=0;i<R;i++){ double mx=A->d[i*C]; for(int j=1;j<C;j++) if(A->d[i*C+j]>mx)mx=A->d[i*C+j]; double se=0; for(int j=0;j<C;j++)se+=exp(A->d[i*C+j]-mx); for(int j=0;j<C;j++){ double p=exp(A->d[i*C+j]-mx)/se; y->d[i*C+j]=p; y->aux[i*C+j]=p; } } y->s0=A;y->bw=bw_sm; return y; }
/* RoPE */
static double ROPE_BASE=10000.0;
static void bw_rope(T*y){ T*X=y->s0; int R=y->r,C=y->c; for(int i=0;i<R;i++)for(int k=0;k<C/2;k++){ double th=(double)i*pow(ROPE_BASE,-2.0*k/C),c=cos(th),s=sin(th); double d0=y->g[i*C+2*k],d1=y->g[i*C+2*k+1]; X->g[i*C+2*k]+=d0*c+d1*s; X->g[i*C+2*k+1]+=-d0*s+d1*c; } }
static T* rope(T*X){ T*y=t_node(X->r,X->c); int R=X->r,C=X->c; for(int i=0;i<R;i++)for(int k=0;k<C/2;k++){ double th=(double)i*pow(ROPE_BASE,-2.0*k/C),c=cos(th),s=sin(th); double x0=X->d[i*C+2*k],x1=X->d[i*C+2*k+1]; y->d[i*C+2*k]=x0*c-x1*s; y->d[i*C+2*k+1]=x0*s+x1*c; } y->s0=X;y->bw=bw_rope; return y; }
/* GELU (erf exact): y=0.5x(1+erf(x/sqrt2)); dy/dx=0.5(1+erf(x/√2))+x*exp(-x^2/2)/√(2π) */
static void bw_gelu(T*y){ T*X=y->s0; for(int i=0;i<X->r*X->c;i++){ double x=X->d[i]; double c=0.5*(1.0+erf(x*0.70710678118654752)); double d=c + x*exp(-0.5*x*x)*0.39894228040143268; X->g[i]+=d*y->g[i]; } }
static T* gelu(T*X){ T*y=t_node(X->r,X->c); for(int i=0;i<X->r*X->c;i++){ double x=X->d[i]; y->d[i]=0.5*x*(1.0+erf(x*0.70710678118654752)); } y->s0=X;y->bw=bw_gelu; return y; }
/* LayerNorm over last dim with learned gain [1,d] (bias added separately via
 * addb). y_j = xhat_j * gain_j, xhat = (x-mu)/sqrt(var+eps). */
static void bw_ln(T*y){ T*X=y->s0,*G=y->s1; int R=X->r,C=X->c; double eps=1e-5;
    for(int i=0;i<R;i++){ double mu=0; for(int j=0;j<C;j++)mu+=X->d[i*C+j]; mu/=C; double var=0; for(int j=0;j<C;j++){ double dd=X->d[i*C+j]-mu; var+=dd*dd; } var/=C; double is=1.0/sqrt(var+eps);
        double mdxhat=0, mdxhatxhat=0; double xhat[512];
        for(int j=0;j<C;j++){ xhat[j]=(X->d[i*C+j]-mu)*is; double dxh=y->g[i*C+j]*G->d[j]; mdxhat+=dxh; mdxhatxhat+=dxh*xhat[j]; }
        mdxhat/=C; mdxhatxhat/=C;
        for(int j=0;j<C;j++){ double dxh=y->g[i*C+j]*G->d[j]; X->g[i*C+j]+= is*(dxh - mdxhat - xhat[j]*mdxhatxhat);
            G->g[j]+= y->g[i*C+j]*xhat[j]; }
    }
}
static T* layernorm(T*X,T*gain){ int R=X->r,C=X->c; T*y=t_node(R,C); double eps=1e-5;
    for(int i=0;i<R;i++){ double mu=0; for(int j=0;j<C;j++)mu+=X->d[i*C+j]; mu/=C; double var=0; for(int j=0;j<C;j++){ double dd=X->d[i*C+j]-mu; var+=dd*dd; } var/=C; double is=1.0/sqrt(var+eps);
        for(int j=0;j<C;j++) y->d[i*C+j]= ((X->d[i*C+j]-mu)*is)*gain->d[j]; }
    y->s0=X; y->s1=gain; y->bw=bw_ln; return y; }
/* masked next-token cross-entropy */
static T* xent(T*z,const int*tg,const int*mask){ int R=z->r,Vv=z->c; T*L=t_node(1,1); L->s0=z; L->it=malloc(R*sizeof(int)); L->aux=malloc((size_t)R*Vv*sizeof(double)); L->p1=0;
    memcpy(L->it,tg,R*sizeof(int)); int nact=0; double tot=0; double* mk=malloc(R*sizeof(double));
    for(int i=0;i<R;i++){ mk[i]=mask[i]; double mx=z->d[i*Vv]; for(int j=1;j<Vv;j++) if(z->d[i*Vv+j]>mx)mx=z->d[i*Vv+j]; double se=0; for(int j=0;j<Vv;j++)se+=exp(z->d[i*Vv+j]-mx); for(int j=0;j<Vv;j++)L->aux[i*Vv+j]=exp(z->d[i*Vv+j]-mx)/se; if(mask[i]){ tot+=(mx+log(se))-z->d[i*Vv+tg[i]]; nact++; } }
    L->p0=nact>0?1.0/nact:0; L->d[0]=nact>0?tot/nact:0; L->msk=malloc(R*sizeof(int)); memcpy(L->msk,mask,R*sizeof(int)); free(mk);
    extern void bw_xent(T*); L->bw=bw_xent; return L; }
void bw_xent(T*L){ T*z=L->s0; int R=z->r,Vv=z->c; double inv=L->p0; for(int i=0;i<R;i++) if(L->msk[i]) for(int j=0;j<Vv;j++) z->g[i*Vv+j]+=(L->aux[i*Vv+j]-(j==L->it[i]?1.0:0.0))*inv*L->g[0]; }

/* backward traversal */
static T* TOPO[MAXN]; static int NT=0;
static void dfs(T*t){ if(!t||t->vis)return; t->vis=1; dfs(t->s0); dfs(t->s1); TOPO[NT++]=t; }
static void backward(T*loss){ NT=0; for(int i=0;i<NN;i++)NODES[i]->vis=0; for(int i=0;i<NP;i++)PARAMS[i]->vis=0; dfs(loss); loss->g[0]=1.0; for(int i=NT-1;i>=0;i--) if(TOPO[i]->bw)TOPO[i]->bw(TOPO[i]); }
static void zero_grads(void){ for(int i=0;i<NP;i++) memset(PARAMS[i]->g,0,(size_t)PARAMS[i]->r*PARAMS[i]->c*sizeof(double)); }

/* ================= config-driven transformer ================= */
#define VOCAB 16
#define D 32
#define NHEAD 1
#define DFF 64
#define NLAYER 2
#define L 12
static T *E;                                  /* [VOCAB,D] tied */
static T *ln1g[NLAYER],*ln1b[NLAYER],*Wq[NLAYER],*Wk[NLAYER],*Wv[NLAYER],*Wo[NLAYER];
static T *ln2g[NLAYER],*ln2b[NLAYER],*W1[NLAYER],*b1[NLAYER],*W2[NLAYER],*b2[NLAYER];
static T *lnfg,*lnfb;
static T *Xin;                                /* [L,VOCAB] one-hot input */

static void build(void){
    for(int i=0;i<NP;i++){ free(PARAMS[i]->d);free(PARAMS[i]->g);free(PARAMS[i]); } NP=0;
    E=t_param(VOCAB,D,0.5/sqrt((double)D));
    for(int l=0;l<NLAYER;l++){
        ln1g[l]=t_param(1,D,0); for(int j=0;j<D;j++)ln1g[l]->d[j]=1.0; ln1b[l]=t_param(1,D,0);
        Wq[l]=t_param(D,D,0.5/sqrt((double)D)); Wk[l]=t_param(D,D,0.5/sqrt((double)D));
        Wv[l]=t_param(D,D,0.5/sqrt((double)D)); Wo[l]=t_param(D,D,0.5/sqrt((double)D));
        ln2g[l]=t_param(1,D,0); for(int j=0;j<D;j++)ln2g[l]->d[j]=1.0; ln2b[l]=t_param(1,D,0);
        W1[l]=t_param(D,DFF,0.5/sqrt((double)D)); b1[l]=t_param(1,DFF,0);
        W2[l]=t_param(DFF,D,0.5/sqrt((double)DFF)); b2[l]=t_param(1,D,0);
    }
    lnfg=t_param(1,D,0); for(int j=0;j<D;j++)lnfg->d[j]=1.0; lnfb=t_param(1,D,0);
}
static T* attn(T*x,int l){
    T*Q=rope(mm(x,Wq[l])), *K=rope(mm(x,Wk[l])), *Vv=mm(x,Wv[l]);
    T*S=sc(mm(Q,tr(K)),1.0/sqrt((double)D));
    T*P=sm(cm(S));
    return mm(mm(P,Vv),Wo[l]);
}
static T* ffn(T*x,int l){ return addb(mm(gelu(addb(mm(x,W1[l]),b1[l])),W2[l]),b2[l]); }
static T* forward_logits(void){
    T*h=mm(Xin,E);                             /* [L,D] */
    for(int l=0;l<NLAYER;l++){
        h=add(h, attn(addb(layernorm(h,ln1g[l]),ln1b[l]), l));
        h=add(h, ffn(addb(layernorm(h,ln2g[l]),ln2b[l]), l));
    }
    h=addb(layernorm(h,lnfg),lnfb);
    return mm(h, tr(E));                        /* tied head -> [L,VOCAB] */
}
static int SEQ[L];
static void make_seq(int *tg,int *mask){
    for(int i=0;i<L;i++) SEQ[i]=irand(VOCAB);
    memset(Xin->d,0,(size_t)L*VOCAB*sizeof(double));
    for(int i=0;i<L;i++) Xin->d[i*VOCAB+SEQ[i]]=1.0;
    for(int i=0;i<L;i++){ if(i<L-1){ tg[i]=SEQ[i+1]; mask[i]=1; } else { tg[i]=0; mask[i]=0; } }
}

static int TG[L],MASK[L];
static double loss_only(void){ T*z=forward_logits(); T*Lz=xent(z,TG,MASK); double v=Lz->d[0]; arena_reset(); return v; }
static int gradcheck(void){
    seed(3ULL); build(); make_seq(TG,MASK);
    zero_grads(); T*z=forward_logits(); T*Lz=xent(z,TG,MASK); backward(Lz); arena_reset();
    double worst=0,worstabs=0;
    for(int p=0;p<NP;p++){ T*par=PARAMS[p]; int tot=par->r*par->c;
        for(int s=0;s<4;s++){ int j=irand(tot); double keep=par->d[j],h=1e-5;
            par->d[j]=keep+h; double Lp=loss_only(); par->d[j]=keep-h; double Lm=loss_only(); par->d[j]=keep;
            double num=(Lp-Lm)/(2*h), ana=par->g[j]; double ab=fabs(num-ana); if(ab>worstabs)worstabs=ab;
            if(fabs(num)+fabs(ana)>1e-3){ double rel=ab/(fabs(num)+fabs(ana)); if(rel>worst)worst=rel; } } }
    printf("gradcheck (full transformer, %d params): max-abs %.2e, max-rel(sig) %.2e -> %s\n",
           NP, worstabs, worst, (worstabs<1e-6&&worst<1e-6)?"PASS":"FAIL");
    return (worstabs<1e-6&&worst<1e-6);
}
static int causality(void){
    seed(9ULL); build(); make_seq(TG,MASK);
    T*z=forward_logits(); double base[L*VOCAB]; memcpy(base,z->d,sizeof(base)); arena_reset();
    int ok=1;
    for(int p=1;p<L;p++){ double save[VOCAB]; memcpy(save,&Xin->d[p*VOCAB],VOCAB*sizeof(double));
        memset(&Xin->d[p*VOCAB],0,VOCAB*sizeof(double)); Xin->d[p*VOCAB+irand(VOCAB)]=1.0;
        T*z2=forward_logits(); for(int i=0;i<p;i++)for(int j=0;j<VOCAB;j++) if(z2->d[i*VOCAB+j]!=base[i*VOCAB+j]) ok=0; arena_reset();
        memcpy(&Xin->d[p*VOCAB],save,VOCAB*sizeof(double)); }
    printf("bit-exact causality (perturb pos p -> rows i<p identical): %s\n", ok?"PASS":"FAIL");
    return ok;
}
int main(void){
    Xin=t_alloc(L,VOCAB);
    printf("=== Day 5e: full transformer (config: d=%d layers=%d dff=%d vocab=%d L=%d, tied head) ===\n",D,NLAYER,DFF,VOCAB,L);
    int g=gradcheck(); int c=causality();
    int np=0; seed(3);build(); for(int i=0;i<NP;i++) np+=PARAMS[i]->r*PARAMS[i]->c;
    printf("param count (this config): %d\n", np);
    printf("JSON {\"gradcheck_pass\":%s,\"causality_pass\":%s,\"params\":%d}\n", g?"true":"false", c?"true":"false", np);
    return (g&&c)?0:1;
}
