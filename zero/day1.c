/*
 * day1.c — the machinery of learning, built from zero.
 *
 * Foundations permitted: hardware, OS, C compiler, mathematics.
 * Nothing else. No ML libraries, no numerics libraries, no copied
 * architecture. Every derivative below is derived by hand and then
 * PROVEN against central finite differences — pure calculus, so the
 * only authority being trusted is math itself.
 *
 * What this file contains, in dependency order:
 *   1. our own PRNG (xorshift64*)                      — randomness
 *   2. a tensor: a 2-D grid of doubles + a grad grid    — state
 *   3. reverse-mode differentiation                     — learning signal
 *        Derivation: for y = f(x), the chain rule dL/dx = dL/dy · dy/dx
 *        applied backwards over the computation graph visits each edge
 *        once. Cost of all gradients ≈ cost of one forward pass. This is
 *        why we differentiate in reverse; we take it because the math
 *        says so, not because frameworks do it.
 *   4. four operations with hand-derived backward rules:
 *        matmul     dA = dY·Bᵀ,  dB = Aᵀ·dY
 *        add_bias   dX = dY,     db_j = Σ_i dY_ij
 *        tanh       dx = (1 − y²)·dy
 *        softmax + cross-entropy, fused:
 *          L = mean_i [ logΣ_j e^{z_ij} − z_{i,t_i} ]
 *          ∂L/∂z_ij = (p_ij − 1[j = t_i]) / m,  p = softmax(z)
 *          (derived from d(logsumexp)/dz_j = p_j; the classical result —
 *           we re-derive it, we don't import it)
 *   5. PROOF 1: analytic gradients vs finite differences on every
 *      parameter class (max relative error printed; double precision
 *      should give ~1e-8)
 *   6. PROOF 2: a micro character-level language model — context of 2
 *      chars, one hidden layer — trained by plain gradient descent using
 *      only the machinery above, on one sentence of our own. Loss must
 *      fall from ln(V) toward the entropy floor of the data, and greedy
 *      generation must reproduce the learned loop.
 *
 * Compile:  gcc -O2 -std=c11 -Wall -o day1 day1.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----------------------------------------------------------- 1. PRNG */
static unsigned long long RNG = 0x9E3779B97F4A7C15ULL;
static double rnd(void) {                    /* xorshift64*, uniform [0,1) */
    RNG ^= RNG >> 12; RNG ^= RNG << 25; RNG ^= RNG >> 27;
    return (double)((RNG * 0x2545F4914F6CDD1DULL) >> 11)
           / (double)(1ULL << 53);
}

/* --------------------------------------------------------- 2. tensor */
typedef struct T T;
struct T {
    int r, c;            /* rows, cols                                  */
    double *d, *g;       /* values, gradients                           */
    T *s0, *s1;          /* graph sources (NULL for leaves)             */
    void (*bw)(T *);     /* backward rule for the op that produced this */
    int *it;             /* int targets (cross-entropy only)            */
    double *aux;         /* cached softmax probs (cross-entropy only)   */
    int vis;             /* topological-sort mark                       */
};

#define MAXN 8192
static T *NODES[MAXN]; static int NN = 0;    /* per-step activation arena */
static T *PARAMS[64];  static int NP = 0;    /* persistent parameters     */

static T *t_alloc(int r, int c) {
    T *t = calloc(1, sizeof(T));
    t->r = r; t->c = c;
    t->d = calloc((size_t)r * c, sizeof(double));
    t->g = calloc((size_t)r * c, sizeof(double));
    return t;
}
static T *t_node(int r, int c) { T *t = t_alloc(r, c); NODES[NN++] = t; return t; }
static T *t_param(int r, int c, double scale) {
    T *t = t_alloc(r, c);
    for (int i = 0; i < r * c; i++) t->d[i] = scale * (2.0 * rnd() - 1.0);
    PARAMS[NP++] = t;
    return t;
}
static void arena_reset(void) {              /* free the step's graph */
    for (int i = 0; i < NN; i++) {
        free(NODES[i]->d); free(NODES[i]->g);
        free(NODES[i]->it); free(NODES[i]->aux);
        free(NODES[i]);
    }
    NN = 0;
}

/* ---------------------------------------------- 4. ops + hand-derived bw */
static void bw_matmul(T *y) {
    T *A = y->s0, *B = y->s1;
    for (int i = 0; i < A->r; i++)               /* dA = dY · Bᵀ */
        for (int k = 0; k < A->c; k++) {
            double s = 0;
            for (int j = 0; j < y->c; j++)
                s += y->g[i * y->c + j] * B->d[k * B->c + j];
            A->g[i * A->c + k] += s;
        }
    for (int k = 0; k < B->r; k++)               /* dB = Aᵀ · dY */
        for (int j = 0; j < B->c; j++) {
            double s = 0;
            for (int i = 0; i < A->r; i++)
                s += A->d[i * A->c + k] * y->g[i * y->c + j];
            B->g[k * B->c + j] += s;
        }
}
static T *t_matmul(T *A, T *B) {
    T *y = t_node(A->r, B->c);
    for (int i = 0; i < A->r; i++)
        for (int j = 0; j < B->c; j++) {
            double s = 0;
            for (int k = 0; k < A->c; k++)
                s += A->d[i * A->c + k] * B->d[k * B->c + j];
            y->d[i * y->c + j] = s;
        }
    y->s0 = A; y->s1 = B; y->bw = bw_matmul;
    return y;
}

static void bw_addb(T *y) {
    T *X = y->s0, *b = y->s1;
    for (int i = 0; i < y->r * y->c; i++) X->g[i] += y->g[i];
    for (int j = 0; j < y->c; j++) {
        double s = 0;
        for (int i = 0; i < y->r; i++) s += y->g[i * y->c + j];
        b->g[j] += s;
    }
}
static T *t_addb(T *X, T *b) {                   /* X[m,n] + b[1,n] */
    T *y = t_node(X->r, X->c);
    for (int i = 0; i < X->r; i++)
        for (int j = 0; j < X->c; j++)
            y->d[i * X->c + j] = X->d[i * X->c + j] + b->d[j];
    y->s0 = X; y->s1 = b; y->bw = bw_addb;
    return y;
}

static void bw_tanh(T *y) {
    T *x = y->s0;
    for (int i = 0; i < y->r * y->c; i++)
        x->g[i] += (1.0 - y->d[i] * y->d[i]) * y->g[i];
}
static T *t_tanh(T *x) {
    T *y = t_node(x->r, x->c);
    for (int i = 0; i < x->r * x->c; i++) y->d[i] = tanh(x->d[i]);
    y->s0 = x; y->bw = bw_tanh;
    return y;
}

static void bw_xent(T *L) {
    T *z = L->s0;
    int m = z->r, V = z->c;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < V; j++)
            z->g[i * V + j] +=
                (L->aux[i * V + j] - (j == L->it[i] ? 1.0 : 0.0))
                * L->g[0] / m;
}
static T *t_softmax_xent(T *z, const int *tg) {  /* -> scalar mean loss */
    int m = z->r, V = z->c;
    T *L = t_node(1, 1);
    L->s0 = z; L->bw = bw_xent;
    L->it  = malloc(m * sizeof(int));
    L->aux = malloc((size_t)m * V * sizeof(double));
    memcpy(L->it, tg, m * sizeof(int));
    double total = 0;
    for (int i = 0; i < m; i++) {
        double mx = z->d[i * V];
        for (int j = 1; j < V; j++)
            if (z->d[i * V + j] > mx) mx = z->d[i * V + j];
        double se = 0;
        for (int j = 0; j < V; j++) se += exp(z->d[i * V + j] - mx);
        for (int j = 0; j < V; j++)
            L->aux[i * V + j] = exp(z->d[i * V + j] - mx) / se;
        total += (mx + log(se)) - z->d[i * V + tg[i]];
    }
    L->d[0] = total / m;
    return L;
}

/* ------------------------------------------- 3. reverse-mode traversal */
static T *TOPO[MAXN]; static int NT = 0;
static void dfs(T *t) {
    if (!t || t->vis) return;
    t->vis = 1;
    dfs(t->s0); dfs(t->s1);
    TOPO[NT++] = t;
}
static void backward(T *loss) {
    NT = 0;
    for (int i = 0; i < NN; i++) NODES[i]->vis = 0;
    for (int i = 0; i < NP; i++) PARAMS[i]->vis = 0;
    dfs(loss);
    loss->g[0] = 1.0;                            /* dL/dL = 1 */
    for (int i = NT - 1; i >= 0; i--)
        if (TOPO[i]->bw) TOPO[i]->bw(TOPO[i]);
}
static void zero_grads(void) {
    for (int i = 0; i < NP; i++)
        memset(PARAMS[i]->g, 0,
               (size_t)PARAMS[i]->r * PARAMS[i]->c * sizeof(double));
}
static void sgd(double lr) {
    for (int i = 0; i < NP; i++)
        for (int j = 0; j < PARAMS[i]->r * PARAMS[i]->c; j++)
            PARAMS[i]->d[j] -= lr * PARAMS[i]->g[j];
}

/* ============================ the first model ============================
 * One sentence of our own; context = previous 2 characters, one-hot,
 * concatenated -> hidden tanh layer -> logits over vocab. Data wraps
 * around so the mapping is a closed loop the model can fully learn
 * (up to the entropy of genuinely ambiguous contexts).                */
static const char *TEXT = "the loom weaves what the weaver wills. ";

static int  V;                 /* vocab size                */
static char VOCAB[64];         /* id -> char                */
static int  n_ex;              /* number of training pairs  */
static T   *X;                 /* [n_ex, 2V] one-hot pairs  */
static int  TG[256];           /* next-char targets         */

static int cid(char ch) {
    for (int i = 0; i < V; i++) if (VOCAB[i] == ch) return i;
    VOCAB[V] = ch; return V++;
}
static void build_data(void) {
    int n = (int)strlen(TEXT);
    int ids[256];
    for (int i = 0; i < n; i++) ids[i] = cid(TEXT[i]);
    n_ex = n;                                   /* circular: n pairs */
    X = t_alloc(n_ex, 2 * V);                   /* leaf, not in arena */
    for (int i = 0; i < n_ex; i++) {
        int a = ids[i], b = ids[(i + 1) % n], t = ids[(i + 2) % n];
        X->d[i * 2 * V + a]     = 1.0;
        X->d[i * 2 * V + V + b] = 1.0;
        TG[i] = t;
    }
}

static T *W1, *b1, *W2, *b2;
static T *model_loss(T *inp, const int *tg) {   /* builds graph in arena */
    T *h = t_tanh(t_addb(t_matmul(inp, W1), b1));
    T *z = t_addb(t_matmul(h, W2), b2);
    return t_softmax_xent(z, tg);
}

/* -------------------- PROOF 1: gradients vs finite differences ------- */
static double loss_value_only(void) {
    T *L = model_loss(X, TG);
    double v = L->d[0];
    arena_reset();
    return v;
}
static void proof_gradients(void) {
    zero_grads();
    T *L = model_loss(X, TG);
    backward(L);
    arena_reset();

    double worst = 0; int checked = 0;
    for (int p = 0; p < NP; p++) {
        T *par = PARAMS[p];
        int total = par->r * par->c;
        for (int s = 0; s < 6; s++) {            /* 6 random entries each */
            int j = (int)(rnd() * total);
            double keep = par->d[j], h = 1e-5;
            par->d[j] = keep + h; double Lp = loss_value_only();
            par->d[j] = keep - h; double Lm = loss_value_only();
            par->d[j] = keep;
            double num = (Lp - Lm) / (2 * h);
            double ana = par->g[j];
            double rel = fabs(num - ana) /
                         (fabs(num) + fabs(ana) + 1e-12);
            if (rel > worst) worst = rel;
            checked++;
        }
    }
    printf("PROOF 1  gradcheck: %d entries across %d parameter tensors, "
           "max relative error vs finite differences = %.2e  ->  %s\n",
           checked, NP, worst, worst < 1e-6 ? "PASS" : "FAIL");
    if (worst >= 1e-6) exit(1);
}

/* -------------------- PROOF 2: it learns language -------------------- */
static void proof_learning(void) {
    printf("PROOF 2  training the first model built on this machinery\n");
    printf("         data: \"%s\" (%d chars, vocab %d, %d examples)\n",
           TEXT, (int)strlen(TEXT), V, n_ex);
    int n_par = 0;
    for (int i = 0; i < NP; i++) n_par += PARAMS[i]->r * PARAMS[i]->c;
    printf("         parameters: %d   chance loss ln(V) = %.4f\n",
           n_par, log((double)V));

    for (int step = 0; step <= 3000; step++) {
        zero_grads();
        T *L = model_loss(X, TG);
        if (step % 500 == 0)
            printf("         step %4d  loss %.4f\n", step, L->d[0]);
        backward(L);
        arena_reset();
        sgd(0.5);
    }

    /* greedy generation from our engine's forward pass */
    char out[81]; int a = cid('t'), b = cid('h');
    out[0] = 't'; out[1] = 'h';
    T *one = t_alloc(1, 2 * V);
    for (int i = 2; i < 80; i++) {
        memset(one->d, 0, 2 * V * sizeof(double));
        one->d[a] = 1.0; one->d[V + b] = 1.0;
        T *h = t_tanh(t_addb(t_matmul(one, W1), b1));
        T *z = t_addb(t_matmul(h, W2), b2);
        int best = 0;
        for (int j = 1; j < V; j++)
            if (z->d[j] > z->d[best]) best = j;
        arena_reset();
        out[i] = VOCAB[best];
        a = b; b = best;
    }
    out[80] = 0;
    printf("         greedy sample (seed \"th\"): %s\n", out);
}

int main(void) {
    build_data();
    int H = 48;
    W1 = t_param(2 * V, H, 0.5 / sqrt((double)(2 * V)));
    b1 = t_param(1, H, 0.0);
    W2 = t_param(H, V, 0.5 / sqrt((double)H));
    b2 = t_param(1, V, 0.0);

    proof_gradients();
    proof_learning();
    printf("day one complete: differentiation and learning, from zero.\n");
    return 0;
}
