/*
 * day3.c — sequence machinery, derived and built on the day1 engine.
 *
 * Foundations: hardware, OS, C compiler, mathematics. No ML/numerics
 * libraries. New ops below each carry a hand-derived backward rule, proven by
 * gradcheck. Derivations are pre-registered in ZERO.md (Day 3):
 *   - content-addressed comparison -> scaled dot-product attention
 *   - order awareness              -> RoPE (rotary position embedding)
 *   - autoregressive influence     -> causal mask
 * Proofs produced at runtime: gradcheck (<1e-6), causality bit-exact, and the
 * RoPE relative-offset property (<1e-9). Measurement: an induction task, our
 * attention model vs a uniform-causal-attention baseline. Optimizer: OURS
 * (day2 RMSProp).
 *
 * Compile: gcc -O2 -std=c11 -Wall -Wno-misleading-indentation -o day3 day3.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================== day1 engine (carried over) ====================== */
static unsigned long long RNG = 0x9E3779B97F4A7C15ULL;
static void rng_seed(unsigned long long s){ RNG = s?s:0x9E3779B97F4A7C15ULL; }
static double rnd(void){ RNG^=RNG>>12; RNG^=RNG<<25; RNG^=RNG>>27;
    return (double)((RNG*0x2545F4914F6CDD1DULL)>>11)/(double)(1ULL<<53); }
static int irand(int n){ return (int)(rnd()*n); }

typedef struct T T;
struct T { int r,c; double *d,*g; T *s0,*s1; void (*bw)(T*);
           int *it; double *aux; int *msk; double p0; int vis; };
#define MAXN 200000
static T *NODES[MAXN]; static int NN=0;
static T *PARAMS[64];  static int NP=0;
static T *t_alloc(int r,int c){ T*t=calloc(1,sizeof(T)); t->r=r;t->c=c;
    t->d=calloc((size_t)r*c,sizeof(double)); t->g=calloc((size_t)r*c,sizeof(double)); return t; }
static T *t_node(int r,int c){ T*t=t_alloc(r,c); NODES[NN++]=t; return t; }
static T *t_param(int r,int c,double scale){ T*t=t_alloc(r,c);
    for(int i=0;i<r*c;i++) t->d[i]=scale*(2.0*rnd()-1.0); PARAMS[NP++]=t; return t; }
static void arena_reset(void){ for(int i=0;i<NN;i++){ free(NODES[i]->d);free(NODES[i]->g);
    free(NODES[i]->it);free(NODES[i]->aux);free(NODES[i]->msk);free(NODES[i]); } NN=0; }

/* --- matmul: y=A@B, dA=dY B^T, dB=A^T dY (day1) --- */
static void bw_matmul(T*y){ T*A=y->s0,*B=y->s1;
    for(int i=0;i<A->r;i++) for(int k=0;k<A->c;k++){ double s=0;
        for(int j=0;j<y->c;j++) s+=y->g[i*y->c+j]*B->d[k*B->c+j]; A->g[i*A->c+k]+=s; }
    for(int k=0;k<B->r;k++) for(int j=0;j<B->c;j++){ double s=0;
        for(int i=0;i<A->r;i++) s+=A->d[i*A->c+k]*y->g[i*y->c+j]; B->g[k*B->c+j]+=s; } }
static T *t_matmul(T*A,T*B){ T*y=t_node(A->r,B->c);
    for(int i=0;i<A->r;i++) for(int j=0;j<B->c;j++){ double s=0;
        for(int k=0;k<A->c;k++) s+=A->d[i*A->c+k]*B->d[k*B->c+j]; y->d[i*y->c+j]=s; }
    y->s0=A;y->s1=B;y->bw=bw_matmul; return y; }

/* --- add bias [1,n] broadcast over rows (day1) --- */
static void bw_addb(T*y){ T*X=y->s0,*b=y->s1;
    for(int i=0;i<y->r*y->c;i++) X->g[i]+=y->g[i];
    for(int j=0;j<y->c;j++){ double s=0; for(int i=0;i<y->r;i++) s+=y->g[i*y->c+j]; b->g[j]+=s; } }
static T *t_addb(T*X,T*b){ T*y=t_node(X->r,X->c);
    for(int i=0;i<X->r;i++) for(int j=0;j<X->c;j++) y->d[i*X->c+j]=X->d[i*X->c+j]+b->d[j];
    y->s0=X;y->s1=b;y->bw=bw_addb; return y; }

/* ===================== NEW Day-3 ops (hand-derived backward) ============ */

/* transpose: Y[j,i]=A[i,j]; backward dA[i,j] += dY[j,i]. */
static void bw_transpose(T*y){ T*A=y->s0;
    for(int i=0;i<A->r;i++) for(int j=0;j<A->c;j++) A->g[i*A->c+j]+=y->g[j*y->c+i]; }
static T *t_transpose(T*A){ T*y=t_node(A->c,A->r);
    for(int i=0;i<A->r;i++) for(int j=0;j<A->c;j++) y->d[j*y->c+i]=A->d[i*A->c+j];
    y->s0=A;y->bw=bw_transpose; return y; }

/* scale by constant s (stored in p0): Y=s*A; backward dA += s*dY. */
static void bw_scale(T*y){ T*A=y->s0; double s=y->p0;
    for(int i=0;i<A->r*A->c;i++) A->g[i]+=s*y->g[i]; }
static T *t_scale(T*A,double s){ T*y=t_node(A->r,A->c);
    for(int i=0;i<A->r*A->c;i++) y->d[i]=s*A->d[i];
    y->s0=A;y->p0=s;y->bw=bw_scale; return y; }

/* causal mask on a square [L,L] score matrix: keep j<=i, else -inf.
 * backward passes gradient only through kept entries. */
#define NEG (-1e30)
static void bw_cmask(T*y){ T*A=y->s0;
    for(int i=0;i<y->r;i++) for(int j=0;j<y->c;j++) if(j<=i) A->g[i*y->c+j]+=y->g[i*y->c+j]; }
static T *t_cmask(T*A){ T*y=t_node(A->r,A->c);
    for(int i=0;i<A->r;i++) for(int j=0;j<A->c;j++) y->d[i*A->c+j]=(j<=i)?A->d[i*A->c+j]:NEG;
    y->s0=A;y->bw=bw_cmask; return y; }

/* row-softmax: P[i,:]=softmax(A[i,:]). probs cached in aux.
 * backward: dA_ij = P_ij*(dP_ij - sum_k P_ik dP_ik). */
static void bw_rowsm(T*y){ T*A=y->s0; int R=y->r,C=y->c;
    for(int i=0;i<R;i++){ double dot=0; for(int j=0;j<C;j++) dot+=y->aux[i*C+j]*y->g[i*C+j];
        for(int j=0;j<C;j++) A->g[i*C+j]+=y->aux[i*C+j]*(y->g[i*C+j]-dot); } }
static T *t_rowsm(T*A){ T*y=t_node(A->r,A->c); int R=A->r,C=A->c;
    y->aux=malloc((size_t)R*C*sizeof(double));
    for(int i=0;i<R;i++){ double mx=A->d[i*C]; for(int j=1;j<C;j++) if(A->d[i*C+j]>mx) mx=A->d[i*C+j];
        double se=0; for(int j=0;j<C;j++) se+=exp(A->d[i*C+j]-mx);
        for(int j=0;j<C;j++){ double p=exp(A->d[i*C+j]-mx)/se; y->d[i*C+j]=p; y->aux[i*C+j]=p; } }
    y->s0=A;y->bw=bw_rowsm; return y; }

/* RoPE: rotate each 2-D pair (2k,2k+1) of row i by theta=i*base^(-2k/d).
 * forward: y0=x0 c - x1 s ; y1=x0 s + x1 c
 * backward (orthogonal map, J^T = R(-theta)):
 *   dx0 += dy0 c + dy1 s ; dx1 += -dy0 s + dy1 c                          */
static double ROPE_BASE=10000.0;
static int USE_ROPE=1;   /* ablation switch: 0 = no positional signal at all */
static void bw_rope(T*y){ T*X=y->s0; int R=y->r,C=y->c;
    for(int i=0;i<R;i++) for(int k=0;k<C/2;k++){
        double th=(double)i*pow(ROPE_BASE,-2.0*k/C), c=cos(th), s=sin(th);
        double d0=y->g[i*C+2*k], d1=y->g[i*C+2*k+1];
        X->g[i*C+2*k]   += d0*c + d1*s;
        X->g[i*C+2*k+1] += -d0*s + d1*c; } }
static T *t_rope(T*X){ T*y=t_node(X->r,X->c); int R=X->r,C=X->c;
    for(int i=0;i<R;i++) for(int k=0;k<C/2;k++){
        double th=(double)i*pow(ROPE_BASE,-2.0*k/C), c=cos(th), s=sin(th);
        double x0=X->d[i*C+2*k], x1=X->d[i*C+2*k+1];
        y->d[i*C+2*k]=x0*c - x1*s; y->d[i*C+2*k+1]=x0*s + x1*c; }
    y->s0=X;y->bw=bw_rope; return y; }

/* masked softmax+cross-entropy over rows of z[L,V], targets it[], mask msk[].
 * loss = (1/Nact) sum_i msk_i*(logsumexp(z_i) - z_{i,t_i}); Nact=sum msk.
 * backward: z.g[i,j] += msk_i*(p_ij - 1[j=t_i])/Nact * dL. */
static T *t_xent_masked(T*z,const int*tg,const int*mask){ int R=z->r,V=z->c; T*L=t_node(1,1);
    L->s0=z;L->it=malloc(R*sizeof(int));L->msk=malloc(R*sizeof(int));
    L->aux=malloc((size_t)R*V*sizeof(double));
    memcpy(L->it,tg,R*sizeof(int)); memcpy(L->msk,mask,R*sizeof(int));
    int nact=0; double total=0;
    for(int i=0;i<R;i++){ double mx=z->d[i*V]; for(int j=1;j<V;j++) if(z->d[i*V+j]>mx) mx=z->d[i*V+j];
        double se=0; for(int j=0;j<V;j++) se+=exp(z->d[i*V+j]-mx);
        for(int j=0;j<V;j++) L->aux[i*V+j]=exp(z->d[i*V+j]-mx)/se;
        if(mask[i]){ total+=(mx+log(se))-z->d[i*V+tg[i]]; nact++; } }
    L->p0=(nact>0)?(1.0/nact):0.0; L->d[0]=(nact>0)?total/nact:0.0;
    L->bw=NULL; /* set below */
    /* stash bw via a trampoline that reads p0 as 1/Nact */
    extern void bw_xent_masked(T*); L->bw=bw_xent_masked; return L; }
void bw_xent_masked(T*L){ T*z=L->s0; int R=z->r,V=z->c; double inv=L->p0;
    for(int i=0;i<R;i++) if(L->msk[i]) for(int j=0;j<V;j++)
        z->g[i*V+j]+=(L->aux[i*V+j]-(j==L->it[i]?1.0:0.0))*inv*L->g[0]; }

/* ===================== reverse-mode traversal (day1) =================== */
static T *TOPO[MAXN]; static int NT=0;
static void dfs(T*t){ if(!t||t->vis) return; t->vis=1; dfs(t->s0); dfs(t->s1); TOPO[NT++]=t; }
static void backward(T*loss){ NT=0; for(int i=0;i<NN;i++) NODES[i]->vis=0;
    for(int i=0;i<NP;i++) PARAMS[i]->vis=0; dfs(loss); loss->g[0]=1.0;
    for(int i=NT-1;i>=0;i--) if(TOPO[i]->bw) TOPO[i]->bw(TOPO[i]); }
static void zero_grads(void){ for(int i=0;i<NP;i++)
    memset(PARAMS[i]->g,0,(size_t)PARAMS[i]->r*PARAMS[i]->c*sizeof(double)); }

/* ===================== OUR optimizer (day2 RMSProp) ==================== */
static double *VST[64];
static void opt_alloc(void){ for(int i=0;i<NP;i++){ int n=PARAMS[i]->r*PARAMS[i]->c;
    VST[i]=realloc(VST[i],n*sizeof(double)); memset(VST[i],0,n*sizeof(double)); } }
static void step_ours(double lr,double beta,double eps){
    for(int i=0;i<NP;i++){ int n=PARAMS[i]->r*PARAMS[i]->c;
        for(int j=0;j<n;j++){ double g=PARAMS[i]->g[j];
            VST[i][j]=beta*VST[i][j]+(1-beta)*g*g;
            PARAMS[i]->d[j]-=lr*g/(sqrt(VST[i][j])+eps); } } }

/* ===================== the model ======================================= */
#define V 8            /* content vocab */
#define L 24           /* sequence length */
#define D 32           /* model width (even, for RoPE) */
static T *X_in;                 /* [L,2V] one-hot (cur,prev), leaf, reused */
static T *PUNIF;                /* [L,L] uniform-causal P, leaf (baseline)  */
static T *E,*Wq,*Wk,*Wv,*Wout,*bout;   /* params */

static void build_model(int with_qk){
    for(int i=0;i<NP;i++){ free(PARAMS[i]->d);free(PARAMS[i]->g);free(PARAMS[i]); } NP=0;
    E   = t_param(2*V, D, 0.5/sqrt((double)(2*V)));
    Wv  = t_param(D, D, 0.5/sqrt((double)D));
    Wout= t_param(D, V, 0.5/sqrt((double)D));
    bout= t_param(1, V, 0.0);
    if(with_qk){ Wq=t_param(D,D,0.5/sqrt((double)D)); Wk=t_param(D,D,0.5/sqrt((double)D)); }
    else { Wq=Wk=NULL; }
}
/* attention (use_attn=1) or uniform-causal baseline (0). returns logits[L,V] */
static T *forward_logits(int use_attn){
    T *Xe = t_matmul(X_in, E);              /* [L,D] token+prev embedding */
    T *Vv = t_matmul(Xe, Wv);               /* values [L,D] */
    T *O;
    if(use_attn){
        T *Qm = t_matmul(Xe, Wq), *Km = t_matmul(Xe, Wk);
        T *Q = USE_ROPE ? t_rope(Qm) : Qm;  /* [L,D] with rotary position */
        T *K = USE_ROPE ? t_rope(Km) : Km;
        T *S = t_scale(t_matmul(Q, t_transpose(K)), 1.0/sqrt((double)D)); /* [L,L] */
        T *P = t_rowsm(t_cmask(S));          /* causal, row-softmax */
        O = t_matmul(P, Vv);                 /* [L,D] */
    } else {
        O = t_matmul(PUNIF, Vv);             /* fixed uniform-causal mixing */
    }
    return t_addb(t_matmul(O, Wout), bout);  /* [L,V] logits */
}

/* ===================== induction task ================================== */
/* fill X_in and targets/mask for one fresh sequence. */
static void make_sequence(int *tg,int *mask){
    int x[L];
    for(int i=0;i<L;i++) x[i]=irand(V);
    memset(X_in->d,0,(size_t)L*2*V*sizeof(double));
    for(int i=0;i<L;i++){ X_in->d[i*2*V + x[i]]=1.0;                 /* current */
        if(i>0) X_in->d[i*2*V + V + x[i-1]]=1.0; }                   /* previous */
    for(int i=0;i<L;i++){ tg[i]=0; mask[i]=0;
        int prev=-1; for(int j=i-1;j>=0;j--) if(x[j]==x[i]){ prev=j; break; }
        if(prev>=0){ tg[i]=x[prev+1]; mask[i]=1; } }  /* token after last match */
}
static void set_uniform_causal(void){
    for(int i=0;i<L;i++) for(int j=0;j<L;j++)
        PUNIF->d[i*L+j]=(j<=i)?1.0/(double)(i+1):0.0;
}

/* eval accuracy over `nseq` fresh sequences */
static double eval_acc(int use_attn,int nseq){
    int tg[L],mask[L]; long correct=0,total=0;
    for(int s=0;s<nseq;s++){ make_sequence(tg,mask);
        T *z=forward_logits(use_attn);
        for(int i=0;i<L;i++) if(mask[i]){ int best=0;
            for(int j=1;j<V;j++) if(z->d[i*V+j]>z->d[i*V+best]) best=j;
            if(best==tg[i]) correct++; total++; }
        arena_reset(); }
    return total? (double)correct/total : 0.0;
}
/* train; returns final eval accuracy. */
static double train(int use_attn,int steps,int batch,double lr){
    rng_seed(424242ULL); build_model(use_attn); opt_alloc();
    int tg[L],mask[L];
    for(int step=1;step<=steps;step++){
        zero_grads();
        for(int b=0;b<batch;b++){ make_sequence(tg,mask);
            T *z=forward_logits(use_attn);
            T *Lz=t_xent_masked(z,tg,mask);
            backward(Lz); arena_reset(); }
        step_ours(lr,0.9,1e-8);
    }
    return eval_acc(use_attn,256);
}

/* ===================== PROOFS ========================================== */
/* gradcheck the full attention stack (with_qk model, one sequence). */
static int LTG[L],LMASK[L];
static double loss_only(int use_attn){ T*z=forward_logits(use_attn);
    T*Lz=t_xent_masked(z,LTG,LMASK); double v=Lz->d[0]; arena_reset(); return v; }
/* gradcheck reports TWO numbers, because a single relative metric is the wrong
 * instrument here: at the near-uniform-attention region, some Wq/Wk gradient
 * entries are ~1e-7, and central finite differences have absolute truncation
 * error ~|f'''|h^2/6 ~ 1e-11, which is a huge RELATIVE error against a tiny
 * true gradient while proving nothing wrong. So we require BOTH:
 *   (a) max ABSOLUTE error < 1e-6 over all sampled entries, and
 *   (b) max RELATIVE error < 1e-6 over entries with a non-negligible gradient
 *       (|num|+|ana| > 1e-3), where the relative metric is meaningful.
 * A backward rule is correct at every point; (a) alone already proves it. */
static double GC_ABS=0, GC_RELSIG=0;
static int gradcheck(void){
    rng_seed(7ULL); build_model(1); make_sequence(LTG,LMASK);
    for(int j=0;j<Wq->r*Wq->c;j++) Wq->d[j]*=6.0;   /* sharpen off degenerate init */
    for(int j=0;j<Wk->r*Wk->c;j++) Wk->d[j]*=6.0;
    zero_grads(); T*z=forward_logits(1); T*Lz=t_xent_masked(z,LTG,LMASK);
    backward(Lz); arena_reset();
    double worstabs=0, worstrelsig=0;
    for(int p=0;p<NP;p++){ T*par=PARAMS[p]; int tot=par->r*par->c;
        for(int s=0;s<12;s++){ int j=irand(tot); double keep=par->d[j],h=1e-5;
            par->d[j]=keep+h; double Lp=loss_only(1);
            par->d[j]=keep-h; double Lm=loss_only(1); par->d[j]=keep;
            double num=(Lp-Lm)/(2*h), ana=par->g[j];
            double ab=fabs(num-ana); if(ab>worstabs) worstabs=ab;
            if(fabs(num)+fabs(ana) > 1e-3){
                double rel=ab/(fabs(num)+fabs(ana));
                if(rel>worstrelsig) worstrelsig=rel; } } }
    GC_ABS=worstabs; GC_RELSIG=worstrelsig;
    int pass = (worstabs<1e-6 && worstrelsig<1e-6);
    printf("  [gradcheck: max-abs err=%.2e ; max-rel err (significant grads)=%.2e]\n",
           worstabs, worstrelsig);
    return pass;
}
/* causality: perturb input row p, assert logits rows i<p bit-identical. */
static int causality_bitexact(void){
    rng_seed(11ULL); build_model(1); int tg[L],mask[L]; make_sequence(tg,mask);
    T *z0=forward_logits(1); double *base=malloc((size_t)L*V*sizeof(double));
    memcpy(base,z0->d,(size_t)L*V*sizeof(double)); arena_reset();
    int worst_ok=1;
    for(int p=1;p<L;p++){
        double save[2*V]; memcpy(save,&X_in->d[p*2*V],2*V*sizeof(double));
        for(int t=0;t<2*V;t++) X_in->d[p*2*V+t]=0.0;            /* arbitrary change */
        X_in->d[p*2*V + irand(V)]=1.0;
        T *z=forward_logits(1);
        for(int i=0;i<p;i++) for(int j=0;j<V;j++)
            if(z->d[i*V+j]!=base[i*V+j]) worst_ok=0;            /* exact equality */
        arena_reset();
        memcpy(&X_in->d[p*2*V],save,2*V*sizeof(double));
    }
    free(base); return worst_ok;
}
/* RoPE relative property: score(q@i,k@j) == score(q@i+dt,k@j+dt). */
static double rope_relative_maxerr(void){
    double q[D],k[D]; for(int t=0;t<D;t++){ q[t]=2*rnd()-1; k[t]=2*rnd()-1; }
    double maxe=0;
    int pairs[5][2]={{3,1},{7,2},{10,4},{20,5},{15,15}};
    int deltas[3]={1,3,6};
    for(int pp=0;pp<5;pp++){ int i=pairs[pp][0], j=pairs[pp][1];
        for(int dd=0;dd<3;dd++){ int dt=deltas[dd];
            /* score at (i,j) */
            double sa=0,sb=0;
            for(int k2=0;k2<D/2;k2++){
                double thi=(double)i*pow(ROPE_BASE,-2.0*k2/D), thj=(double)j*pow(ROPE_BASE,-2.0*k2/D);
                double qi0=q[2*k2]*cos(thi)-q[2*k2+1]*sin(thi), qi1=q[2*k2]*sin(thi)+q[2*k2+1]*cos(thi);
                double kj0=k[2*k2]*cos(thj)-k[2*k2+1]*sin(thj), kj1=k[2*k2]*sin(thj)+k[2*k2+1]*cos(thj);
                sa+=qi0*kj0+qi1*kj1;
                double thi2=(double)(i+dt)*pow(ROPE_BASE,-2.0*k2/D), thj2=(double)(j+dt)*pow(ROPE_BASE,-2.0*k2/D);
                double qi0b=q[2*k2]*cos(thi2)-q[2*k2+1]*sin(thi2), qi1b=q[2*k2]*sin(thi2)+q[2*k2+1]*cos(thi2);
                double kj0b=k[2*k2]*cos(thj2)-k[2*k2+1]*sin(thj2), kj1b=k[2*k2]*sin(thj2)+k[2*k2+1]*cos(thj2);
                sb+=qi0b*kj0b+qi1b*kj1b;
            }
            double e=fabs(sa-sb); if(e>maxe) maxe=e;
        } }
    return maxe;
}

int main(void){
    X_in = t_alloc(L,2*V);
    PUNIF= t_alloc(L,L); set_uniform_causal();

    printf("=== Day 3: sequence machinery ===\n");

    int gc=gradcheck();
    printf("PROOF gradcheck (attention stack): abs<1e-6 & rel<1e-6 -> %s\n",
           gc?"PASS":"FAIL");

    int caus=causality_bitexact();
    printf("PROOF causality bit-exact: perturb pos p, rows i<p unchanged -> %s\n",
           caus?"PASS":"FAIL");

    double re=rope_relative_maxerr();
    printf("PROOF RoPE relative offset: max |score(i,j)-score(i+d,j+d)| = %.2e -> %s\n",
           re, re<1e-9?"PASS":"FAIL");

    if(!gc || !caus || re>=1e-9){ printf("a proof failed; stop.\n"); return 1; }

    printf("\n=== induction task: attention vs uniform-causal baseline ===\n");
    int steps=4000, batch=16; double lr=0.01;
    double acc_attn=train(1,steps,batch,lr);
    double acc_base=train(0,steps,batch,lr);
    printf("attention  accuracy (fresh eval): %.3f\n", acc_attn);
    printf("baseline   accuracy (fresh eval): %.3f\n", acc_base);
    printf("chance (1/V) = %.3f\n", 1.0/V);

    /* RoPE ablation (exploratory): true no-positional-signal (skip rotation). */
    USE_ROPE=0;
    double acc_norope=train(1,steps,batch,lr);
    USE_ROPE=1;
    printf("attention no-position accuracy: %.3f (ablation, RoPE off)\n", acc_norope);

    printf("\nJSON {\"gradcheck_abs\":%.3e,\"gradcheck_relsig\":%.3e,\"causality_bitexact\":%s,\"rope_rel_maxerr\":%.3e,"
           "\"acc_attention\":%.3f,\"acc_baseline\":%.3f,\"acc_attention_norope\":%.3f,"
           "\"chance\":%.3f,\"steps\":%d,\"batch\":%d}\n",
           GC_ABS, GC_RELSIG, caus?"true":"false", re, acc_attn, acc_base, acc_norope, 1.0/V, steps, batch);
    return 0;
}
