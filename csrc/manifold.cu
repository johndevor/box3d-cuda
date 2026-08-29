// SPDX-License-Identifier: MIT
// Fixed-topology persistent OBB manifolds: deterministic face clipping,
// topology feature IDs, warm starting, accumulated 2-D friction and repair.
#ifndef BOX3D_DEVICE_HELPERS_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int SW=13, MAXP=4, MAXCLIP=12;
constexpr float SAT_AXIS_TIE_EPSILON_M=1.0e-6f;
constexpr int64_t FACE_TAG=int64_t(1)<<60, EDGE_TAG=int64_t(2)<<60, SPECULATIVE_TAG=int64_t(3)<<60;

struct CV { float p[3]; int source, clip; };
struct MP { int64_t id; float p[3], depth, jn, jt1, jt2; };
struct MF { float n[3], t1[3], t2[3]; MP points[MAXP]; int count; };
struct SAT { bool hit; float n[3], depth, point[3]; int axis, kind; };

__device__ inline float dot3(const float*a,const float*b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ inline void cross3(const float*a,const float*b,float*o){o[0]=a[1]*b[2]-a[2]*b[1];o[1]=a[2]*b[0]-a[0]*b[2];o[2]=a[0]*b[1]-a[1]*b[0];}
__device__ inline void rotate3(const float*q,const float*v,float*o){float qv[3]={q[0],q[1],q[2]},t[3],u[3];cross3(qv,v,t);t[0]*=2;t[1]*=2;t[2]*=2;cross3(qv,t,u);for(int k=0;k<3;k++)o[k]=v[k]+q[3]*t[k]+u[k];}
__device__ inline void axes3(const float*q,float a[3][3]){float x[3]={1,0,0},y[3]={0,1,0},z[3]={0,0,1};rotate3(q,x,a[0]);rotate3(q,y,a[1]);rotate3(q,z,a[2]);}
__device__ inline void iworld(const float*q,const float*d,const float*v,float*o){float qc[4]={-q[0],-q[1],-q[2],q[3]},l[3];rotate3(qc,v,l);for(int k=0;k<3;k++)l[k]*=d[k];rotate3(q,l,o);}
__device__ inline void integrate_q(float*b,float h){float qx=b[3],qy=b[4],qz=b[5],qw=b[6],dx=b[10]*h,dy=b[11]*h,dz=b[12]*h;float x=qx+.5f*(dx*qw+dy*qz-dz*qy),y=qy+.5f*(-dx*qz+dy*qw+dz*qx),z=qz+.5f*(dx*qy-dy*qx+dz*qw),w=qw-.5f*(dx*qx+dy*qy+dz*qz);float l=x*x+y*y+z*z+w*w;if(l<=1e-20f){b[3]=b[4]=b[5]=0;b[6]=1;return;}l=rsqrtf(l);b[3]=x*l;b[4]=y*l;b[5]=z*l;b[6]=w*l;}
__device__ inline float radius(const float a[3][3],const float*h,const float*n){return h[0]*fabsf(dot3(a[0],n))+h[1]*fabsf(dot3(a[1],n))+h[2]*fabsf(dot3(a[2],n));}

__device__ inline void support(const float*b,const float a[3][3],const float*h,const float*d,float eps,float*o){o[0]=b[0];o[1]=b[1];o[2]=b[2];for(int k=0;k<3;k++){float x=dot3(a[k],d),s=x>eps?1:(x<-eps?-1:0);for(int j=0;j<3;j++)o[j]+=a[k][j]*h[k]*s;}}
__device__ inline SAT sat(const float*a,const float*ha,const float*b,const float*hb,float eps){SAT r{};float aa[3][3],ab[3][3];axes3(a+3,aa);axes3(b+3,ab);float t[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]},depths[15],normals[15][3],md=1e30f;bool valid[15]={};for(int index=0;index<15;index++){float c[3];if(index<3){for(int k=0;k<3;k++)c[k]=aa[index][k];}else if(index<6){for(int k=0;k<3;k++)c[k]=ab[index-3][k];}else{int edge=index-6;cross3(aa[edge/3],ab[edge%3],c);}float l=dot3(c,c);if(index>=6&&l<=eps*eps)continue;if(l<=1e-20f)continue;l=rsqrtf(l);for(int k=0;k<3;k++)c[k]*=l;float sd=dot3(t,c),dep=radius(aa,ha,c)+radius(ab,hb,c)-fabsf(sd);if(dep < -eps)return r;valid[index]=true;depths[index]=dep;md=fminf(md,dep);for(int k=0;k<3;k++)normals[index][k]=c[k]*(sd<0?-1:1);}int best=-1;for(int index=0;index<15;index++)if(valid[index]&&depths[index]<=md+SAT_AXIS_TIE_EPSILON_M){if(best<0||(index<6&&best>=6)||((index<6)==(best<6)&&index<best))best=index;}if(best<0)return r;r.hit=true;r.depth=fmaxf(0,depths[best]);r.axis=best;r.kind=best<3?0:(best<6?1:2);for(int k=0;k<3;k++)r.n[k]=normals[best][k];float pa[3],pb[3],opp[3]={-r.n[0],-r.n[1],-r.n[2]};support(a,aa,ha,r.n,eps,pa);support(b,ab,hb,opp,eps,pb);for(int k=0;k<3;k++)r.point[k]=.5f*(pa[k]+pb[k]);return r;}

__device__ inline void tangents(const float*n,float*t1,float*t2){int seed=0;float best=fabsf(n[0]);for(int k=1;k<3;k++)if(fabsf(n[k])<best){seed=k;best=fabsf(n[k]);}float s[3]={0,0,0};s[seed]=1;cross3(s,n,t1);float l=rsqrtf(fmaxf(dot3(t1,t1),1e-24f));for(int k=0;k<3;k++)t1[k]*=l;cross3(n,t1,t2);l=rsqrtf(fmaxf(dot3(t2,t2),1e-24f));for(int k=0;k<3;k++)t2[k]*=l;}
__device__ inline int vertex_bit(const int*s){int i=(s[0]>0?1:0)|(s[1]>0?2:0)|(s[2]>0?4:0);return 1<<i;}
__device__ inline int clip_poly(const CV*in,int count,CV*out,const float*origin,const float*axis,float bound,int bit,float eps){if(!count)return 0;int n=0;CV prev=in[count-1];float dp=(prev.p[0]-origin[0])*axis[0]+(prev.p[1]-origin[1])*axis[1]+(prev.p[2]-origin[2])*axis[2]-bound;bool ip=dp<=eps;for(int i=0;i<count;i++){CV cur=in[i];float dc=(cur.p[0]-origin[0])*axis[0]+(cur.p[1]-origin[1])*axis[1]+(cur.p[2]-origin[2])*axis[2]-bound;bool ic=dc<=eps;if(ic!=ip&&n<MAXCLIP){float den=dp-dc,f=fabsf(den)<=eps?.5f:fmaxf(0,fminf(1,dp/den));CV v{};for(int k=0;k<3;k++)v.p[k]=prev.p[k]+(cur.p[k]-prev.p[k])*f;v.source=prev.source|cur.source;v.clip=prev.clip|cur.clip|bit;out[n++]=v;}if(ic&&n<MAXCLIP)out[n++]=cur;prev=cur;dp=dc;ip=ic;}return n;}
__device__ inline int64_t face_id(int pair,bool rb,int ra,int rs,int ia,int is,int src,int clip){return FACE_TAG|(int64_t(pair&255)<<48)|(int64_t(rb?1:0)<<47)|(int64_t(ra&3)<<45)|(int64_t(rs>0?1:0)<<44)|(int64_t(ia&3)<<42)|(int64_t(is>0?1:0)<<41)|(int64_t(src&255)<<8)|int64_t(clip&255);}
__device__ inline int64_t edge_id(int pair,const SAT&s){int e=s.axis>=6?s.axis-6:0,ia=e/3,ib=e%3,bits=0;for(int k=0;k<3;k++)bits|=(s.n[k]>=0?1:0)<<k;return EDGE_TAG|(int64_t(pair&255)<<48)|(int64_t(ia&3)<<44)|(int64_t(ib&3)<<42)|bits|1;}
__device__ inline int64_t speculative_id(int pair,int axis,const float*n){int bits=0;for(int k=0;k<3;k++)bits|=(n[k]>=0?1:0)<<k;return SPECULATIVE_TAG|(int64_t(pair&255)<<48)|(int64_t(axis&15)<<40)|bits|1;}
__device__ inline void sort_points(MP*p,int n){for(int i=1;i<n;i++){MP v=p[i];int j=i-1;while(j>=0&&p[j].id>v.id){p[j+1]=p[j];j--;}p[j+1]=v;}}
__device__ inline bool has_feature(const MP*p,int n,int64_t id){for(int i=0;i<n;i++)if(p[i].id==id)return true;return false;}

__device__ inline bool manifold(const float*a,const float*ha,const float*b,const float*hb,int pair,float eps,MF*m){SAT s=sat(a,ha,b,hb,eps);if(!s.hit)return false;for(int k=0;k<3;k++)m->n[k]=s.n[k];tangents(m->n,m->t1,m->t2);m->count=0;if(s.kind==2){MP p{};p.id=edge_id(pair,s);p.depth=s.depth;for(int k=0;k<3;k++)p.p[k]=s.point[k];m->points[0]=p;m->count=1;return true;}bool rb=s.kind==1;const float*ref=rb?b:a,*inc=rb?a:b,*hr=rb?hb:ha,*hi=rb?ha:hb;float ar[3][3],ai[3][3];axes3(ref+3,ar);axes3(inc+3,ai);float outward[3];for(int k=0;k<3;k++)outward[k]=rb?-s.n[k]:s.n[k];int raxis=rb?s.axis-3:s.axis,rs=dot3(ar[raxis],outward)>=0?1:-1;float rc[3];for(int k=0;k<3;k++)rc[k]=ref[k]+ar[raxis][k]*rs*hr[raxis];int sides[2],si=0;for(int k=0;k<3;k++)if(k!=raxis)sides[si++]=k;int ia=0;for(int k=1;k<3;k++)if(fabsf(dot3(ai[k],outward))>fabsf(dot3(ai[ia],outward)))ia=k;int is=dot3(ai[ia],outward)>=0?-1:1, isides[2];si=0;for(int k=0;k<3;k++)if(k!=ia)isides[si++]=k;float ic[3];for(int k=0;k<3;k++)ic[k]=inc[k]+ai[ia][k]*is*hi[ia];int signs2[4][2]={{-1,-1},{1,-1},{1,1},{-1,1}};CV p1[MAXCLIP],p2[MAXCLIP];for(int v=0;v<4;v++){int signs[3]={0,0,0};signs[ia]=is;signs[isides[0]]=signs2[v][0];signs[isides[1]]=signs2[v][1];for(int k=0;k<3;k++)p1[v].p[k]=ic[k]+ai[isides[0]][k]*signs2[v][0]*hi[isides[0]]+ai[isides[1]][k]*signs2[v][1]*hi[isides[1]];p1[v].source=vertex_bit(signs);p1[v].clip=0;}int count=4,plane=0;for(int z=0;z<2;z++){int side=sides[z];count=clip_poly(p1,count,p2,rc,ar[side],hr[side],1<<plane++,eps);float neg[3]={-ar[side][0],-ar[side][1],-ar[side][2]};count=clip_poly(p2,count,p1,rc,neg,hr[side],1<<plane++,eps);}MP candidates[MAXCLIP];int cn=0;for(int v=0;v<count;v++){float d[3]={p1[v].p[0]-rc[0],p1[v].p[1]-rc[1],p1[v].p[2]-rc[2]},pd=dot3(d,outward);if(pd>eps)continue;MP q{};q.id=face_id(pair,rb,raxis,rs,ia,is,p1[v].source,p1[v].clip);q.depth=fmaxf(0,-pd);for(int k=0;k<3;k++)q.p[k]=p1[v].p[k]-outward[k]*pd*.5f;candidates[cn++]=q;}if(!cn){MP q{};q.id=edge_id(pair,s);q.depth=s.depth;for(int k=0;k<3;k++)q.p[k]=s.point[k];m->points[0]=q;m->count=1;return true;}if(cn<=4){for(int k=0;k<cn;k++)m->points[k]=candidates[k];m->count=cn;sort_points(m->points,m->count);return true;}int chosen=0;float*tv[4]={m->t1,m->t1,m->t2,m->t2};int sg[4]={-1,1,-1,1};for(int c=0;c<4;c++){int best=0;for(int i=1;i<cn;i++){float ka=sg[c]*dot3(candidates[i].p,tv[c]),kb=sg[c]*dot3(candidates[best].p,tv[c]);if(ka<kb||(ka==kb&&(candidates[i].depth>candidates[best].depth||(candidates[i].depth==candidates[best].depth&&candidates[i].id<candidates[best].id))))best=i;}if(!has_feature(m->points,chosen,candidates[best].id))m->points[chosen++]=candidates[best];}while(chosen<4){int best=-1;for(int i=0;i<cn;i++)if(!has_feature(m->points,chosen,candidates[i].id)&&(best<0||candidates[i].depth>candidates[best].depth||(candidates[i].depth==candidates[best].depth&&candidates[i].id<candidates[best].id)))best=i;if(best<0)break;m->points[chosen++]=candidates[best];}m->count=chosen;sort_points(m->points,m->count);return true;}

// Build a one-point velocity-only candidate from the true SAT gap. Unlike
// extent inflation, this is an isotropic pair distance in world space.
__device__ inline bool speculative_manifold(const float*a,const float*ha,const float*b,const float*hb,int pair,float pair_distance,float eps,MF*m,bool*actual){
  if(manifold(a,ha,b,hb,pair,eps,m)){*actual=true;return true;}
  *actual=false;if(pair_distance<=0)return false;
  float aa[3][3],ab[3][3];axes3(a+3,aa);axes3(b+3,ab);float delta[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
  float maximum=-1e30f,best_normal[3]={};int best=-1;
  for(int index=0;index<15;index++){float axis[3];if(index<3){for(int k=0;k<3;k++)axis[k]=aa[index][k];}else if(index<6){for(int k=0;k<3;k++)axis[k]=ab[index-3][k];}else{int edge=index-6;cross3(aa[edge/3],ab[edge%3],axis);}float length=dot3(axis,axis);if(index>=6&&length<=eps*eps)continue;if(length<=1e-20f)continue;length=rsqrtf(length);for(int k=0;k<3;k++)axis[k]*=length;float signed_distance=dot3(delta,axis),separation=fabsf(signed_distance)-radius(aa,ha,axis)-radius(ab,hb,axis);if(separation>pair_distance+eps)return false;bool prefer=best<0||separation>maximum+SAT_AXIS_TIE_EPSILON_M||(fabsf(separation-maximum)<=SAT_AXIS_TIE_EPSILON_M&&((index<6&&best>=6)||((index<6)==(best<6)&&index<best)));if(prefer){best=index;maximum=separation;for(int k=0;k<3;k++)best_normal[k]=axis[k]*(signed_distance<0?-1:1);}}
  if(best<0||maximum<=eps)return false;for(int k=0;k<3;k++)m->n[k]=best_normal[k];tangents(m->n,m->t1,m->t2);float pa[3],pb[3],opposite[3]={-m->n[0],-m->n[1],-m->n[2]};support(a,aa,ha,m->n,eps,pa);support(b,ab,hb,opposite,eps,pb);MP point{};point.id=speculative_id(pair,best,m->n);point.depth=-maximum;for(int k=0;k<3;k++)point.p[k]=.5f*(pa[k]+pb[k]);m->points[0]=point;m->count=1;return true;
}

__device__ inline void point_v(const float*b,const float*r,float*o){float w[3]={b[10],b[11],b[12]},c[3];cross3(w,r,c);for(int k=0;k<3;k++)o[k]=b[7+k]+c[k];}
__device__ inline float emass(const float*a,float ima,const float*ia,const float*ra,const float*b,float imb,const float*ib,const float*rb,const float*d){float c[3],x[3],y[3];cross3(ra,d,c);iworld(a+3,ia,c,x);cross3(x,ra,y);float v=ima+dot3(y,d);cross3(rb,d,c);iworld(b+3,ib,c,x);cross3(x,rb,y);return v+imb+dot3(y,d);}
__device__ inline void impulse(float*b,float im,const float*ii,const float*r,const float*j){if(im==0)return;for(int k=0;k<3;k++)b[7+k]+=j[k]*im;float c[3],x[3];cross3(r,j,c);iworld(b+3,ii,c,x);for(int k=0;k<3;k++)b[10+k]+=x[k];}
__device__ inline void pair_impulse(float*a,float ima,const float*ia,float*b,float imb,const float*ib,const MP&p,const float*j){float ra[3],rb[3],neg[3];for(int k=0;k<3;k++){ra[k]=p.p[k]-a[k];rb[k]=p.p[k]-b[k];neg[k]=-j[k];}impulse(a,ima,ia,ra,neg);impulse(b,imb,ib,rb,j);}
__device__ inline void seed(MF*m,const int64_t*ids,const float*js){for(int p=0;p<m->count;p++)for(int s=0;s<4;s++)if(ids[s]&&ids[s]==m->points[p].id){m->points[p].jn=fmaxf(0,js[s*3]);m->points[p].jt1=js[s*3+1];m->points[p].jt2=js[s*3+2];}}
__device__ inline void warm(float*a,float ima,const float*ia,float*b,float imb,const float*ib,MF*m){for(int p=0;p<m->count;p++){MP&q=m->points[p];float j[3];for(int k=0;k<3;k++)j[k]=m->n[k]*q.jn+m->t1[k]*q.jt1+m->t2[k]*q.jt2;pair_impulse(a,ima,ia,b,imb,ib,q,j);}}
__device__ inline void solve_normal(float*a,float ima,const float*ia,float*b,float imb,const float*ib,MF*m,MP&q,float e,float eps,float h){float ra[3],rb[3],va[3],vb[3],rel[3];for(int k=0;k<3;k++){ra[k]=q.p[k]-a[k];rb[k]=q.p[k]-b[k];}point_v(a,ra,va);point_v(b,rb,vb);for(int k=0;k<3;k++)rel[k]=vb[k]-va[k];float vn=dot3(rel,m->n),den=emass(a,ima,ia,ra,b,imb,ib,rb,m->n);if(den>eps){float bounce=q.depth>=0&&vn<-.5f?-e*vn:0,allowed=q.depth<0?q.depth/h:0,delta=(allowed+bounce-vn)/den,old=q.jn;q.jn=fmaxf(0,old+delta);float j[3];for(int k=0;k<3;k++)j[k]=m->n[k]*(q.jn-old);pair_impulse(a,ima,ia,b,imb,ib,q,j);}}
__device__ inline void solve_friction(float*a,float ima,const float*ia,float*b,float imb,const float*ib,MF*m,MP&q,float mu,float eps){float ra[3],rb[3],va[3],vb[3],rel[3];for(int k=0;k<3;k++){ra[k]=q.p[k]-a[k];rb[k]=q.p[k]-b[k];}point_v(a,ra,va);point_v(b,rb,vb);for(int k=0;k<3;k++)rel[k]=vb[k]-va[k];float proposed[2]={q.jt1,q.jt2};float*ts[2]={m->t1,m->t2};for(int z=0;z<2;z++){float den=emass(a,ima,ia,ra,b,imb,ib,rb,ts[z]);if(den>eps)proposed[z]+=-dot3(rel,ts[z])/den;}float limit=mu*q.jn,mag=hypotf(proposed[0],proposed[1]);if(mag>limit&&mag>eps){proposed[0]*=limit/mag;proposed[1]*=limit/mag;}float j[3];for(int k=0;k<3;k++)j[k]=m->t1[k]*(proposed[0]-q.jt1)+m->t2[k]*(proposed[1]-q.jt2);q.jt1=proposed[0];q.jt2=proposed[1];pair_impulse(a,ima,ia,b,imb,ib,q,j);}
__device__ inline void repair(float*a,float ima,float*b,float imb,const MF&m,float slop,float pc){float sum=ima+imb;if(sum<=0)return;float dep=0;for(int p=0;p<m.count;p++)dep=fmaxf(dep,m.points[p].depth);float r=fminf(.2f,fmaxf(0,dep-slop)*pc)/sum;for(int k=0;k<3;k++){a[k]-=m.n[k]*r*ima;b[k]+=m.n[k]*r*imb;}}
__device__ inline void write_cache(const MF*m,int64_t*ids,float*js){for(int s=0;s<4;s++){ids[s]=0;js[s*3]=js[s*3+1]=js[s*3+2]=0;}if(!m)return;for(int s=0;s<m->count;s++){ids[s]=m->points[s].id;js[s*3]=m->points[s].jn;js[s*3+1]=m->points[s].jt1;js[s*3+2]=m->points[s].jt2;}}

#ifndef BOX3D_DEVICE_HELPERS_ONLY
__global__ void kernel(float*state,const float*im,const float*half,const float*ii,
    const int64_t*pairs,int64_t*ids,float*cache,uint8_t*contacts,float*pen,
    int32_t*counts,int W,int B,int P,float dt,int sub,float gy,float rest,
    float friction,float slop,float pc,float damp,int iterations,float eps){
  int w=blockIdx.x*blockDim.x+threadIdx.x;
  if(w>=W)return;
  float h=dt/sub,df=fmaxf(0,1-damp*h);
  for(int ss=0;ss<sub;ss++){
    for(int bi=0;bi<B;bi++){
      int f=w*B+bi;
      if(im[f]==0)continue;
      float*b=state+f*SW;
      b[8]+=gy*h;
      for(int k=0;k<3;k++){b[k]+=b[7+k]*h;b[10+k]*=df;}
      integrate_q(b,h);
    }
    MF manif[16];
    bool active[16]={};
    for(int p=0;p<P;p++){
      int64_t ai=pairs[p*2],bi=pairs[p*2+1];
      int off=(w*P+p)*4;
      if(ai<0||bi<0||ai>=B||bi>=B||ai==bi){write_cache(nullptr,ids+off,cache+off*3);continue;}
      int fa=w*B+int(ai),fb=w*B+int(bi);
      active[p]=manifold(state+fa*SW,half+fa*3,state+fb*SW,half+fb*3,p,eps,&manif[p]);
      if(active[p]){
        contacts[w*P+p]=1;
        seed(&manif[p],ids+off,cache+off*3);
        warm(state+fa*SW,im[fa],ii+fa*3,state+fb*SW,im[fb],ii+fb*3,&manif[p]);
      }
    }
    for(int it=0;it<iterations;it++){
      for(int p=0;p<P;p++)if(active[p]){
        int ai=int(pairs[p*2]),bi=int(pairs[p*2+1]),fa=w*B+ai,fb=w*B+bi;
        for(int q=0;q<manif[p].count;q++)
          solve_normal(state+fa*SW,im[fa],ii+fa*3,state+fb*SW,im[fb],ii+fb*3,
              &manif[p],manif[p].points[q],rest,eps,h);
        for(int q=0;q<manif[p].count;q++)
          solve_friction(state+fa*SW,im[fa],ii+fa*3,state+fb*SW,im[fb],ii+fb*3,
              &manif[p],manif[p].points[q],friction,eps);
      }
    }
    for(int p=0;p<P;p++){
      int off=(w*P+p)*4;
      if(active[p]){
        int ai=int(pairs[p*2]),bi=int(pairs[p*2+1]);
        repair(state+(w*B+ai)*SW,im[w*B+ai],state+(w*B+bi)*SW,im[w*B+bi],manif[p],slop,pc);
        write_cache(&manif[p],ids+off,cache+off*3);
      }else write_cache(nullptr,ids+off,cache+off*3);
    }
  }
  for(int p=0;p<P;p++){
    int64_t ai=pairs[p*2],bi=pairs[p*2+1];
    int off=(w*P+p)*4;
    if(ai<0||bi<0||ai>=B||bi>=B||ai==bi){write_cache(nullptr,ids+off,cache+off*3);continue;}
    int fa=w*B+int(ai),fb=w*B+int(bi);
    MF m;
    if(!manifold(state+fa*SW,half+fa*3,state+fb*SW,half+fb*3,p,eps,&m)){
      write_cache(nullptr,ids+off,cache+off*3);continue;
    }
    seed(&m,ids+off,cache+off*3);
    write_cache(&m,ids+off,cache+off*3);
    counts[w*P+p]=m.count;
    float d=0;
    for(int q=0;q<m.count;q++)d=fmaxf(d,m.points[q].depth);
    pen[w*P+p]=d;
  }
}
#endif
}

#ifndef BOX3D_DEVICE_HELPERS_ONLY
std::vector<torch::Tensor> box3d_manifold_step_cuda(torch::Tensor state,torch::Tensor inverse_mass,torch::Tensor half_extents,torch::Tensor inverse_inertia,torch::Tensor pair_indices,torch::Tensor cache_feature_ids,torch::Tensor cache_impulses,double dt,int64_t substeps,double gravity_y,double restitution,double friction,double slop,double position_correction,double angular_damping,int64_t solver_iterations,double sat_epsilon){const c10::cuda::CUDAGuard guard(state.device());auto output=state.clone(),ids=cache_feature_ids.clone(),impulses=cache_impulses.clone();auto contacts=torch::zeros({state.size(0),pair_indices.size(0)},state.options().dtype(torch::kUInt8));auto penetration=torch::zeros({state.size(0),pair_indices.size(0)},state.options());auto counts=torch::zeros({state.size(0),pair_indices.size(0)},state.options().dtype(torch::kInt32));int W=state.size(0),B=state.size(1),P=pair_indices.size(0);constexpr int T=64;int blocks=(W+T-1)/T;kernel<<<blocks,T,0,at::cuda::getDefaultCUDAStream()>>>(output.data_ptr<float>(),inverse_mass.data_ptr<float>(),half_extents.data_ptr<float>(),inverse_inertia.data_ptr<float>(),pair_indices.data_ptr<int64_t>(),ids.data_ptr<int64_t>(),impulses.data_ptr<float>(),contacts.data_ptr<uint8_t>(),penetration.data_ptr<float>(),counts.data_ptr<int32_t>(),W,B,P,float(dt),int(substeps),float(gravity_y),float(restitution),float(friction),float(slop),float(position_correction),float(angular_damping),int(solver_iterations),float(sat_epsilon));C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,contacts,penetration,ids,impulses,counts};}
#endif
