/* Read-only, loopback-only control plane. No inference or workload commands. */
#define _POSIX_C_SOURCE 200809L
#include <zmq.h>
#include <json-c/json.h>
#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <time.h>
#include <ctype.h>
#ifndef CKE_REVISION
#define CKE_REVISION "unknown"
#endif
#define LIMIT 8192
#define MAX_WORKERS 32
static volatile sig_atomic_t stopped;
static const char *node;
static char boot[33];
typedef struct { char id[65]; int port; } worker;
static worker workers[MAX_WORKERS];
static int count;
static void stop(int sig) { (void)sig; stopped=1; }
static const char *str(json_object *o,const char *key) {
    json_object *v=NULL;
    return json_object_object_get_ex(o,key,&v) && json_object_is_type(v,json_type_string)
        && strlen(json_object_get_string(v))==(size_t)json_object_get_string_len(v)
        ? json_object_get_string(v):NULL;
}
static int ident(const char *s) {
    if (!s || !*s || strlen(s)>64) return 0;
    for (;*s;s++) if (!isalnum((unsigned char)*s) && *s!='-' && *s!='_' && *s!='.') return 0;
    return 1;
}
static void put(json_object *o,const char *k,const char *v) {
    json_object_object_add(o,k,v?json_object_new_string(v):NULL);
}
static void stamp(json_object *o) {
    char b[32]; time_t t=time(NULL); struct tm tm;
    gmtime_r(&t,&tm); strftime(b,sizeof b,"%Y-%m-%dT%H:%M:%SZ",&tm); put(o,"observed_at",b);
}
static json_object *parse(const char *s,size_t n) {
    json_tokener *t=json_tokener_new();
    json_tokener_set_flags(t,JSON_TOKENER_STRICT|JSON_TOKENER_VALIDATE_UTF8);
    json_object *o=json_tokener_parse_ex(t,s,(int)n);
    size_t end=json_tokener_get_parse_end(t);
    if (json_tokener_get_error(t)!=json_tokener_success) { json_object_put(o); o=NULL; }
    while(end<n && isspace((unsigned char)s[end])) end++;
    if(end!=n) { json_object_put(o); o=NULL; }
    json_tokener_free(t); return o;
}
typedef struct { char data[LIMIT+1]; size_t n; } body;
static size_t receive(void *p,size_t size,size_t nm,void *ctx) {
    body *b=ctx;
    if (size && nm>SIZE_MAX/size) return 0;
    size_t n=size*nm; if(n>LIMIT-b->n) return 0;
    memcpy(b->data+b->n,p,n); b->n+=n; b->data[b->n]=0; return n;
}
static json_object *observe(worker *w) {
    json_object *o=json_object_new_object(); put(o,"worker_id",w->id); stamp(o);
    put(o,"boot_id",NULL); put(o,"cke_revision",NULL);
    put(o,"capabilities_status","unsupported"); json_object_object_add(o,"capabilities",NULL);
    json_object_object_add(o,"ready",NULL); json_object_object_add(o,"busy",NULL);
    json_object_object_add(o,"queued",NULL); json_object_object_add(o,"worker_generation",NULL);
    char url[96]; snprintf(url,sizeof url,"http://127.0.0.1:%d/health",w->port);
    body b={0}; CURL *c=curl_easy_init(); long code=0; CURLcode rc=CURLE_FAILED_INIT;
    if(c) {
        curl_easy_setopt(c,CURLOPT_URL,url); curl_easy_setopt(c,CURLOPT_PROXY,"");
        curl_easy_setopt(c,CURLOPT_CONNECTTIMEOUT_MS,200L); curl_easy_setopt(c,CURLOPT_TIMEOUT_MS,500L);
        curl_easy_setopt(c,CURLOPT_NOSIGNAL,1L); curl_easy_setopt(c,CURLOPT_FOLLOWLOCATION,0L);
        curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,receive); curl_easy_setopt(c,CURLOPT_WRITEDATA,&b);
        rc=curl_easy_perform(c); curl_easy_getinfo(c,CURLINFO_RESPONSE_CODE,&code); curl_easy_cleanup(c);
    }
    json_object *h=(rc==CURLE_OK && code==200)?parse(b.data,b.n):NULL;
    int valid=h && json_object_is_type(h,json_type_object);
    put(o,"observation_status",valid?"available":"unavailable");
    put(o,"observation_error",valid?NULL:(rc==CURLE_OPERATION_TIMEDOUT?"WORKER_TIMEOUT":"WORKER_UNAVAILABLE"));
    if(valid) {
        json_object *v=NULL;
        if(json_object_object_get_ex(h,"ready",&v) && json_object_is_type(v,json_type_boolean))
            json_object_object_add(o,"ready",json_object_get(v));
        if(json_object_object_get_ex(h,"active",&v) && (!v || json_object_is_type(v,json_type_string)))
            json_object_object_add(o,"busy",json_object_new_boolean(v && json_object_get_string_len(v)>0));
        const char *keys[]={"queued","worker_generation"};
        for(int i=0;i<2;i++) if(json_object_object_get_ex(h,keys[i],&v) && json_object_is_type(v,json_type_int) && json_object_get_int64(v)>=0)
            json_object_object_add(o,keys[i],json_object_get(v));
    }
    json_object_put(h); return o;
}
static json_object *response(json_object *q,const char *forced) {
    json_object *r=json_object_new_object(), *params=NULL,*data=NULL;
    const char *id=str(q,"request_id"), *op=str(q,"operation"), *err=forced;
    put(r,"schema","cke.control.v1"); put(r,"request_id",ident(id)?id:NULL); put(r,"node_id",node);
    put(r,"adapter_boot_id",boot); stamp(r);
    if(!err && (!json_object_is_type(q,json_type_object) || !ident(id) || !op || !str(q,"schema") || strcmp(str(q,"schema"),"cke.control.v1") || !str(q,"node_id") || strcmp(str(q,"node_id"),node) || !json_object_object_get_ex(q,"params",&params) || !json_object_is_type(params,json_type_object))) err="INVALID_REQUEST";
    if(!err && (!strcmp(op,"node.describe") || !strcmp(op,"workers.list"))) {
        if(json_object_object_length(params)) err="INVALID_PARAMS";
        else if(!strcmp(op,"node.describe")) {
            data=json_object_new_object(); put(data,"adapter_health","ready"); put(data,"cke_revision",CKE_REVISION);
            put(data,"transport","loopback-only"); put(data,"worker_discovery","explicit-registration");
            json_object_object_add(data,"worker_count",json_object_new_int(count));
        } else {
            data=json_object_new_object(); json_object *a=json_object_new_array();
            for(int i=0;i<count;i++) { json_object *w=json_object_new_object(); put(w,"worker_id",workers[i].id); json_object_array_add(a,w); }
            json_object_object_add(data,"workers",a);
        }
    } else if(!err && (!strcmp(op,"worker.status") || !strcmp(op,"worker.capabilities"))) {
        const char *wid=str(params,"worker_id");
        if(!ident(wid) || json_object_object_length(params)!=1) err="INVALID_PARAMS";
        else { int i; for(i=0;i<count;i++) if(!strcmp(wid,workers[i].id)) break;
            if(i==count) err="WORKER_NOT_FOUND"; else data=observe(&workers[i]); }
    } else if(!err) err="UNSUPPORTED_OPERATION";
    json_object_object_add(r,"ok",json_object_new_boolean(!err)); json_object_object_add(r,"data",data);
    json_object *e=NULL; if(err) { e=json_object_new_object(); put(e,"code",err); }
    json_object_object_add(r,"error",e); return r;
}
static int port_number(const char *s) {
    char *end; long n=strtol(s,&end,10); return *s && !*end && n>=1024 && n<=65535?(int)n:0;
}
int main(int argc,char **argv) {
    if(argc<3 || argc>3+MAX_WORKERS || !ident(argv[1]) || !port_number(argv[2])) {
        fprintf(stderr,"usage: node-adapter NODE_ID BIND_PORT [WORKER_ID=LOCAL_HTTP_PORT ...]\n"); return 2;
    }
    node=argv[1];
    for(int i=3;i<argc;i++) {
        char *eq=strchr(argv[i],'='); if(!eq) return 2; *eq=0;
        if(!ident(argv[i]) || !port_number(eq+1)) return 2;
        for(int j=0;j<count;j++) if(!strcmp(workers[j].id,argv[i])) return 2;
        strcpy(workers[count].id,argv[i]); workers[count++].port=port_number(eq+1);
    }
    FILE *f=fopen("/dev/urandom","rb"); unsigned char random[16];
    if(!f) return 2;
    size_t nr=fread(random,1,16,f); fclose(f); if(nr!=16) return 2;
    for(int i=0;i<16;i++) snprintf(boot+2*i,3,"%02x",random[i]);
    signal(SIGINT,stop); signal(SIGTERM,stop); if(curl_global_init(CURL_GLOBAL_DEFAULT)) return 2;
    void *ctx=zmq_ctx_new(), *s=NULL; if(!ctx) return 2;
    char endpoint[80]; snprintf(endpoint,sizeof endpoint,"tcp://127.0.0.1:%d",port_number(argv[2]));
rebind:
    s=zmq_socket(ctx,ZMQ_REP); if(!s) return 2;
    int zero=0,hwm=16,timeout=200; int64_t max=LIMIT;
    zmq_setsockopt(s,ZMQ_LINGER,&zero,sizeof zero); zmq_setsockopt(s,ZMQ_RCVHWM,&hwm,sizeof hwm);
    zmq_setsockopt(s,ZMQ_SNDHWM,&hwm,sizeof hwm); zmq_setsockopt(s,ZMQ_RCVTIMEO,&timeout,sizeof timeout);
    zmq_setsockopt(s,ZMQ_SNDTIMEO,&timeout,sizeof timeout); zmq_setsockopt(s,ZMQ_MAXMSGSIZE,&max,sizeof max);
    if(zmq_bind(s,endpoint)) { fprintf(stderr,"bind failed\n"); return 2; }
    while(!stopped) {
        zmq_msg_t m; zmq_msg_init(&m); int n=zmq_msg_recv(&m,s,0);
        if(n<0) { zmq_msg_close(&m); continue; }
        int more=zmq_msg_more(&m); json_object *q=(!more && n<=LIMIT)?parse(zmq_msg_data(&m),(size_t)n):NULL;
        zmq_msg_close(&m);
        /* Multipart is outside this protocol. Reset socket rather than unbounded drain. */
        if(more) { json_object_put(q); if(stopped) break; zmq_close(s); goto rebind; }
        json_object *r=response(q,q?NULL:"MALFORMED_MESSAGE");
        const char *encoded=json_object_to_json_string_ext(r,JSON_C_TO_STRING_PLAIN);
        int sent=zmq_send(s,encoded,strlen(encoded),0); json_object_put(r); json_object_put(q);
        if(sent<0) break;
    }
    zmq_close(s); zmq_ctx_term(ctx); curl_global_cleanup(); return 0;
}
