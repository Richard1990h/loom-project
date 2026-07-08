/* bpe.c — our byte-pair-merge tokenizer. ARTIFACT (pure C, no libraries).
 *
 * Derivation (see ZERO.md): the model needs discrete units that are denser than
 * raw bytes but keep a small vocabulary and stay LOSSLESS. Requirement: start
 * from bytes (so ANY input round-trips exactly), then repeatedly fuse the most
 * frequent adjacent pair of units into a new unit — greedy compression of the
 * corpus's own statistics. (Declared convergence: this is Byte-Pair Encoding.
 * We derive it from the compression requirement; base is bytes so it is exactly
 * invertible by construction.)
 *
 * Modes:
 *   train <corpus> <vocab_size> <out.bin>   learn merges, save tokenizer
 *   proof <corpus> <out.bin>                round-trip on held-out + chars/token
 *
 * Tokenizer file format (documented): see save()/load() below.
 * Build: gcc -O2 -std=c11 -Wall -o bpe bpe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VMAX 8192          /* hard cap on vocab */
typedef struct { int a,b; } Pair;

static uint8_t* read_file(const char* path, long* n){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",path); exit(1); }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* buf=malloc(sz); if(fread(buf,1,sz,f)!=(size_t)sz){ fprintf(stderr,"read fail\n"); exit(1);}
    fclose(f); *n=sz; return buf;
}

/* ---- tokenizer file: magic 'B''P''E''1', int32 vocab_size, int32 n_merges,
 *      then n_merges pairs of int32 (a,b) in learned order. base tokens 0..255
 *      are the byte values; token 256+k = merge(merges[k].a, merges[k].b).   */
static void save(const char* path,int vocab,int nm,Pair* mg){
    FILE* f=fopen(path,"wb"); const char magic[4]={'B','P','E','1'};
    fwrite(magic,1,4,f); fwrite(&vocab,4,1,f); fwrite(&nm,4,1,f);
    for(int i=0;i<nm;i++){ fwrite(&mg[i].a,4,1,f); fwrite(&mg[i].b,4,1,f); }
    fclose(f);
}
static int load(const char* path,int* vocab,Pair* mg){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"no %s\n",path); exit(1);}
    char magic[4]; if(fread(magic,1,4,f)!=4||memcmp(magic,"BPE1",4)){ fprintf(stderr,"bad magic\n"); exit(1);}
    int nm; if(fread(vocab,4,1,f)!=1||fread(&nm,4,1,f)!=1){exit(1);}
    for(int i=0;i<nm;i++){ if(fread(&mg[i].a,4,1,f)!=1||fread(&mg[i].b,4,1,f)!=1){exit(1);} }
    fclose(f); return nm;
}

/* ---- train: learn merges by iterated most-frequent-adjacent-pair fusion ---- */
static void train(const char* corpus,int vocab,const char* out){
    long n; uint8_t* raw=read_file(corpus,&n);
    int* tok=malloc((size_t)n*sizeof(int)); long N=n;
    for(long i=0;i<n;i++) tok[i]=raw[i];
    int VM=vocab>VMAX?VMAX:vocab;
    long* cnt=calloc((size_t)VM*VM,sizeof(long));   /* pair counts, flat */
    Pair* mg=malloc(sizeof(Pair)*(VM));
    int nm=0;
    for(int newid=256; newid<VM; newid++){
        memset(cnt,0,(size_t)VM*VM*sizeof(long));
        for(long i=0;i+1<N;i++){ int a=tok[i],b=tok[i+1]; cnt[(long)a*VM+b]++; }
        long best=0,bi=-1; for(long p=0;p<(long)VM*VM;p++) if(cnt[p]>best){ best=cnt[p]; bi=p; }
        if(bi<0||best<2) break;                     /* nothing worth merging */
        int a=(int)(bi/VM), b=(int)(bi%VM);
        mg[nm].a=a; mg[nm].b=b; nm++;
        long w=0; for(long i=0;i<N;){ if(i+1<N && tok[i]==a && tok[i+1]==b){ tok[w++]=newid; i+=2; } else tok[w++]=tok[i++]; }
        N=w;
        if(newid%256==0 || newid==VM-1) fprintf(stderr,"  merges %d, tokens %ld (last pair %d+%d x%ld)\n",nm,N,a,b,best);
    }
    save(out,VM,nm,mg);
    printf("trained: vocab=%d merges=%d corpus_bytes=%ld tokens_after=%ld chars_per_token=%.3f\n",
           VM,nm,n,N,(double)n/N);
    free(raw);free(tok);free(cnt);free(mg);
}

/* ---- build decode table: token -> byte string ---- */
static uint8_t** vocab_bytes; static int* vocab_len;
static void build_vocab(int vocab,int nm,Pair* mg){
    vocab_bytes=malloc(sizeof(uint8_t*)*vocab); vocab_len=malloc(sizeof(int)*vocab);
    for(int i=0;i<256;i++){ vocab_bytes[i]=malloc(1); vocab_bytes[i][0]=(uint8_t)i; vocab_len[i]=1; }
    for(int k=0;k<nm;k++){ int id=256+k,a=mg[k].a,b=mg[k].b; int l=vocab_len[a]+vocab_len[b];
        vocab_bytes[id]=malloc(l); memcpy(vocab_bytes[id],vocab_bytes[a],vocab_len[a]);
        memcpy(vocab_bytes[id]+vocab_len[a],vocab_bytes[b],vocab_len[b]); vocab_len[id]=l; }
}
/* encode: bytes -> tokens by applying merges in learned order (correctness-first) */
static long encode(const uint8_t* s,long n,int nm,Pair* mg,int* out){
    long N=n; for(long i=0;i<n;i++) out[i]=s[i];
    for(int k=0;k<nm;k++){ int a=mg[k].a,b=mg[k].b,id=256+k; long w=0;
        for(long i=0;i<N;){ if(i+1<N && out[i]==a && out[i+1]==b){ out[w++]=id; i+=2; } else out[w++]=out[i++]; }
        N=w; }
    return N;
}
/* decode: tokens -> bytes via vocab table */
static long decode(const int* t,long N,uint8_t* out){
    long w=0; for(long i=0;i<N;i++){ memcpy(out+w,vocab_bytes[t[i]],vocab_len[t[i]]); w+=vocab_len[t[i]]; } return w;
}

static void proof(const char* corpus,const char* binf){
    int vocab; Pair* mg=malloc(sizeof(Pair)*VMAX); int nm=load(binf,&vocab,mg);
    build_vocab(vocab,nm,mg);
    long n; uint8_t* raw=read_file(corpus,&n);
    /* held-out = last 20% of corpus */
    long ho_start=(long)(n*0.8), ho_n=n-ho_start; const uint8_t* ho=raw+ho_start;
    int* tk=malloc((size_t)ho_n*sizeof(int)); long ntok=encode(ho,ho_n,nm,mg,tk);
    uint8_t* back=malloc(ho_n+16); long nb=decode(tk,ntok,back);
    int ok = (nb==ho_n) && (memcmp(back,ho,ho_n)==0);
    long mism=0; if(nb==ho_n) for(long i=0;i<ho_n;i++) if(back[i]!=ho[i]) mism++;
    printf("round-trip on held-out (%ld bytes): %s  (decoded %ld bytes, %ld mismatches)\n",
           ho_n, ok?"LOSSLESS":"FAILED", nb, (nb==ho_n)?mism:-1);
    printf("chars_per_token(held-out) = %.3f  vocab=%d merges=%d\n",(double)ho_n/ntok,vocab,nm);
    printf("JSON {\"roundtrip_lossless\":%s,\"heldout_bytes\":%ld,\"mismatches\":%ld,\"chars_per_token\":%.3f,\"vocab\":%d,\"merges\":%d}\n",
           ok?"true":"false", ho_n, ok?0:mism, (double)ho_n/ntok, vocab, nm);
    if(!ok){ fprintf(stderr,"ROUND-TRIP NOT LOSSLESS\n"); exit(1); }
    free(raw);free(tk);free(back);free(mg);
}

int main(int argc,char** argv){
    if(argc>=5 && !strcmp(argv[1],"train")){ train(argv[2],atoi(argv[3]),argv[4]); return 0; }
    if(argc>=4 && !strcmp(argv[1],"proof")){ proof(argv[2],argv[3]); return 0; }
    fprintf(stderr,"usage: bpe train <corpus> <vocab> <out.bin> | bpe proof <corpus> <out.bin>\n");
    return 2;
}
