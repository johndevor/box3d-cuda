// SPDX-License-Identifier: MIT
// Fixed-small articulated worlds. One CUDA lane owns one world, which keeps
// shared-body joint updates ordered and eliminates cross-world atomics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int SW=13, FIXED=0, REVOLUTE=1, PRISMATIC=2;

__device__ inline float dot3(const float*a,const float*b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ inline float norm3(const float*a){return sqrtf(fmaxf(0,dot3(a,a)));}
__device__ inline void cross3(const float*a,const float*b,float*o){o[0]=a[1]*b[2]-a[2]*b[1];o[1]=a[2]*b[0]-a[0]*b[2];o[2]=a[0]*b[1]-a[1]*b[0];}
__device__ inline void qmul(const float*a,const float*b,float*o){o[0]=a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1];o[1]=a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0];o[2]=a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3];o[3]=a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2];}
__device__ inline void qnorm(float*q){float l=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];if(l<=1e-20f){q[0]=q[1]=q[2]=0;q[3]=1;return;}l=rsqrtf(l);for(int k=0;k<4;k++)q[k]*=l;}
__device__ inline void rotate3(const float*q,const float*v,float*o){float t[3],u[3];cross3(q,v,t);for(int k=0;k<3;k++)t[k]*=2;cross3(q,t,u);for(int k=0;k<3;k++)o[k]=v[k]+q[3]*t[k]+u[k];}
__device__ inline void iworld(const float*q,const float*d,const float*v,float*o){float qc[4]={-q[0],-q[1],-q[2],q[3]},l[3];rotate3(qc,v,l);for(int k=0;k<3;k++)l[k]*=d[k];rotate3(q,l,o);}
__device__ inline void integrate_q(float*b,float h){float q[4]={b[3],b[4],b[5],b[6]},rv[4]={b[10]*h*.5f,b[11]*h*.5f,b[12]*h*.5f,1},out[4];qmul(rv,q,out);qnorm(out);for(int k=0;k<4;k++)b[3+k]=out[k];}
__device__ inline void rotvec(const float*q0,float*o){float q[4]={q0[0],q0[1],q0[2],q0[3]};qnorm(q);if(q[3]<0)for(int k=0;k<4;k++)q[k]=-q[k];float s=norm3(q);if(s<=1e-12f){for(int k=0;k<3;k++)o[k]=2*q[k];return;}float a=2*atan2f(s,fmaxf(0,q[3]));for(int k=0;k<3;k++)o[k]=q[k]*a/s;}
__device__ inline void deltaq(const float*rv,float*q){float a=norm3(rv);if(a<=1e-12f){q[0]=.5f*rv[0];q[1]=.5f*rv[1];q[2]=.5f*rv[2];q[3]=1;qnorm(q);return;}float s=sinf(.5f*a)/a;for(int k=0;k<3;k++)q[k]=rv[k]*s;q[3]=cosf(.5f*a);}

struct G {float rp[3],rc[3],axis[3],coord,lin[3],ang[3];};
__device__ inline G geometry(const float*p,const float*c,const float*pa,const float*ca,const float*axis0,const float*ref,int type){G g{};rotate3(p+3,pa,g.rp);rotate3(c+3,ca,g.rc);rotate3(p+3,axis0,g.axis);float al=norm3(g.axis);for(int k=0;k<3;k++)g.axis[k]/=fmaxf(al,1e-12f);float pp[3],cp[3],sep[3];for(int k=0;k<3;k++){pp[k]=p[k]+g.rp[k];cp[k]=c[k]+g.rc[k];sep[k]=cp[k]-pp[k];}float qp[4]={-p[3],-p[4],-p[5],p[6]},qr[4],qref[4]={-ref[0],-ref[1],-ref[2],ref[3]},qd[4],rl[3];qmul(qp,c+3,qr);qmul(qr,qref,qd);rotvec(qd,rl);float rw[3];rotate3(p+3,rl,rw);if(type==REVOLUTE){g.coord=dot3(rl,axis0);for(int k=0;k<3;k++){g.lin[k]=sep[k];g.ang[k]=rw[k]-g.axis[k]*dot3(rw,g.axis);}}else if(type==PRISMATIC){g.coord=dot3(sep,g.axis);for(int k=0;k<3;k++){g.lin[k]=sep[k]-g.axis[k]*g.coord;g.ang[k]=rw[k];}}else{g.coord=0;for(int k=0;k<3;k++){g.lin[k]=sep[k];g.ang[k]=rw[k];}}return g;}
__device__ inline void pointv(const float*b,const float*r,float*o){float x[3];cross3(b+10,r,x);for(int k=0;k<3;k++)o[k]=b[7+k]+x[k];}
__device__ inline float lemass(const float*p,float imp,const float*iip,const float*rp,const float*c,float imc,const float*iic,const float*rc,const float*d){float x[3],y[3],z[3];cross3(rp,d,x);iworld(p+3,iip,x,y);cross3(y,rp,z);float v=imp+dot3(z,d);cross3(rc,d,x);iworld(c+3,iic,x,y);cross3(y,rc,z);return v+imc+dot3(z,d);}
__device__ inline float aemass(const float*p,const float*iip,const float*c,const float*iic,const float*d){float a[3],b[3];iworld(p+3,iip,d,a);iworld(c+3,iic,d,b);return dot3(a,d)+dot3(b,d);}
__device__ inline void impulse(float*b,float im,const float*ii,const float*r,const float*j){if(im==0)return;for(int k=0;k<3;k++)b[7+k]+=j[k]*im;float x[3],y[3];cross3(r,j,x);iworld(b+3,ii,x,y);for(int k=0;k<3;k++)b[10+k]+=y[k];}
__device__ inline void aimpulse(float*b,float im,const float*ii,const float*j){if(im==0)return;float y[3];iworld(b+3,ii,j,y);for(int k=0;k<3;k++)b[10+k]+=y[k];}
__device__ inline void basis(const float*a,float*t1,float*t2){int s=0;if(fabsf(a[1])<fabsf(a[s]))s=1;if(fabsf(a[2])<fabsf(a[s]))s=2;float e[3]={0,0,0};e[s]=1;cross3(e,a,t1);float l=norm3(t1);for(int k=0;k<3;k++)t1[k]/=fmaxf(l,1e-12f);cross3(a,t1,t2);l=norm3(t2);for(int k=0;k<3;k++)t2[k]/=fmaxf(l,1e-12f);}
__device__ inline float linearrow(float*p,float imp,const float*iip,float*c,float imc,const float*iic,const float*rp,const float*rc,const float*d){float vp[3],vc[3],rel[3];pointv(p,rp,vp);pointv(c,rc,vc);for(int k=0;k<3;k++)rel[k]=vc[k]-vp[k];float den=lemass(p,imp,iip,rp,c,imc,iic,rc,d);if(den<=1e-12f)return 0.f;float l=-dot3(rel,d)/den,j[3],neg[3];for(int k=0;k<3;k++){j[k]=d[k]*l;neg[k]=-j[k];}impulse(p,imp,iip,rp,neg);impulse(c,imc,iic,rc,j);return l;}
__device__ inline float angularrow(float*p,float imp,const float*iip,float*c,float imc,const float*iic,const float*d){float rel[3];for(int k=0;k<3;k++)rel[k]=c[10+k]-p[10+k];float den=aemass(p,iip,c,iic,d);if(den<=1e-12f)return 0.f;float l=-dot3(rel,d)/den,j[3],neg[3];for(int k=0;k<3;k++){j[k]=d[k]*l;neg[k]=-j[k];}aimpulse(p,imp,iip,neg);aimpulse(c,imc,iic,j);return l;}
__device__ inline void orient(float*b,float im,const float*rv){if(im==0)return;float qd[4],out[4];deltaq(rv,qd);qmul(qd,b+3,out);qnorm(out);for(int k=0;k<4;k++)b[3+k]=out[k];}
__device__ inline void angularrepair(float*p,float imp,const float*iip,float*c,float imc,const float*iic,const float*err,float pc,float slop,float maxr){float m=norm3(err);if(m<=slop)return;float d[3];for(int k=0;k<3;k++)d[k]=err[k]/m;float wp[3],wc[3];iworld(p+3,iip,d,wp);iworld(c+3,iic,d,wc);float ap=dot3(wp,d),ac=dot3(wc,d),sum=ap+ac;if(sum<=1e-12f)return;float corr=fminf(maxr,(m-slop)*pc),rp[3],rc[3];for(int k=0;k<3;k++){rp[k]=d[k]*corr*ap/sum;rc[k]=-d[k]*corr*ac/sum;}orient(p,imp,rp);orient(c,imc,rc);}

#ifndef BOX3D_DEVICE_HELPERS_ONLY
__global__ void kernel(float*state,const float*im,const float*ii,const int64_t*ji,const int64_t*jt,
 const float*pa,const float*ca,const float*axis,const float*ref,const float*lo,const float*hi,
 const float*damp,const uint8_t*men,const float*target,const float*target_position,
 const float*stiffness,const float*effort,float*cache,float warm_factor,float*coordinate,
 float*linerr,float*angerr,float*limiterr,float*motorout,uint8_t*limitactive,
 int W,int B,int J,float dt,int sub,float gy,int iterations,float pc,float pslop,float aslop,float maxlin,float maxang){
 int w=blockIdx.x*blockDim.x+threadIdx.x;if(w>=W)return;float h=dt/sub;
 for(int control=0;control<1;control++)for(int ss=0;ss<sub;ss++){
  for(int b=0;b<B;b++){int f=w*B+b;if(im[f]==0)continue;float*x=state+f*SW;x[8]+=gy*h;for(int k=0;k<3;k++)x[k]+=x[7+k]*h;integrate_q(x,h);}
  float lambda[16*8]={};
  // Explicit world-indexed warm start. Slots are linear rows [0:3], angular
  // rows [3:6], motor [6], and unilateral limit [7].
  for(int j=0;j<J;j++){
   int pi=int(ji[j*2]),ci=int(ji[j*2+1]),fp=w*B+pi,fc=w*B+ci,type=int(jt[j]);float*p=state+fp*SW,*c=state+fc*SW;G g=geometry(p,c,pa+j*3,ca+j*3,axis+j*3,ref+j*4,type);
   float world[3][3]={{1,0,0},{0,1,0},{0,0,1}},t1[3],t2[3];basis(g.axis,t1,t2);float*lr=lambda+j*8;
   for(int row=0;row<8;row++)lr[row]=cache[(w*J+j)*8+row]*warm_factor;
   if(g.coord>=lo[j]&&g.coord<=hi[j])lr[7]=0.f;
   const float*ld[3];int ln;const float*ad[3];int an;
   if(type==PRISMATIC){ld[0]=t1;ld[1]=t2;ln=2;ad[0]=world[0];ad[1]=world[1];ad[2]=world[2];an=3;}
   else if(type==REVOLUTE){ld[0]=world[0];ld[1]=world[1];ld[2]=world[2];ln=3;ad[0]=t1;ad[1]=t2;an=2;}
   else{ld[0]=world[0];ld[1]=world[1];ld[2]=world[2];ln=3;ad[0]=world[0];ad[1]=world[1];ad[2]=world[2];an=3;}
   for(int row=0;row<ln;row++){float q[3],nq[3];for(int k=0;k<3;k++){q[k]=ld[row][k]*lr[row];nq[k]=-q[k];}impulse(p,im[fp],ii+fp*3,g.rp,nq);impulse(c,im[fc],ii+fc*3,g.rc,q);}
   for(int row=0;row<an;row++){float q[3],nq[3];for(int k=0;k<3;k++){q[k]=ad[row][k]*lr[3+row];nq[k]=-q[k];}aimpulse(p,im[fp],ii+fp*3,nq);aimpulse(c,im[fc],ii+fc*3,q);}
   if(type!=FIXED)for(int slot=6;slot<8;slot++){float q[3],nq[3];for(int k=0;k<3;k++){q[k]=g.axis[k]*lr[slot];nq[k]=-q[k];}if(type==REVOLUTE){aimpulse(p,im[fp],ii+fp*3,nq);aimpulse(c,im[fc],ii+fc*3,q);}else{impulse(p,im[fp],ii+fp*3,g.rp,nq);impulse(c,im[fc],ii+fc*3,g.rc,q);}}
  }
  for(int it=0;it<iterations;it++)for(int j=0;j<J;j++){
   int pi=int(ji[j*2]),ci=int(ji[j*2+1]),fp=w*B+pi,fc=w*B+ci,type=int(jt[j]);float*p=state+fp*SW,*c=state+fc*SW;G g=geometry(p,c,pa+j*3,ca+j*3,axis+j*3,ref+j*4,type);
   float world[3][3]={{1,0,0},{0,1,0},{0,0,1}},t1[3],t2[3];basis(g.axis,t1,t2);
   float*lr=lambda+j*8;
   if(type==PRISMATIC){lr[0]+=linearrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,g.rp,g.rc,t1);lr[1]+=linearrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,g.rp,g.rc,t2);for(int k=0;k<3;k++)lr[3+k]+=angularrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,world[k]);}
   else if(type==REVOLUTE){for(int k=0;k<3;k++)lr[k]+=linearrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,g.rp,g.rc,world[k]);lr[3]+=angularrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,t1);lr[4]+=angularrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,t2);}
   else{for(int k=0;k<3;k++){lr[k]+=linearrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,g.rp,g.rc,world[k]);lr[3+k]+=angularrow(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,world[k]);}}
   if(type!=FIXED&&effort[w*J+j]>0){float vp[3],vc[3],rel[3],speed,den;if(type==REVOLUTE){for(int k=0;k<3;k++)rel[k]=c[10+k]-p[10+k];speed=dot3(rel,g.axis);den=aemass(p,ii+fp*3,c,ii+fc*3,g.axis);}else{pointv(p,g.rp,vp);pointv(c,g.rc,vc);for(int k=0;k<3;k++)rel[k]=vc[k]-vp[k];speed=dot3(rel,g.axis);den=lemass(p,im[fp],ii+fp*3,g.rp,c,im[fc],ii+fc*3,g.rc,g.axis);}if(den>1e-12f){float old=lr[6],lim=effort[w*J+j]*h,desired;if(stiffness[j]>0){desired=(stiffness[j]*(target_position[w*J+j]-g.coord)-damp[j]*speed)*h;lr[6]=fmaxf(-lim,fminf(lim,desired));}else{float dl=-damp[j]*speed*h/den;if(men[j])dl+=(target[w*J+j]-speed)/den;lr[6]=fmaxf(-lim,fminf(lim,old+dl));}float dl=lr[6]-old,q[3],nq[3];for(int k=0;k<3;k++){q[k]=g.axis[k]*dl;nq[k]=-q[k];}if(type==REVOLUTE){aimpulse(p,im[fp],ii+fp*3,nq);aimpulse(c,im[fc],ii+fc*3,q);}else{impulse(p,im[fp],ii+fp*3,g.rp,nq);impulse(c,im[fc],ii+fc*3,g.rc,q);}}}
   if(type!=FIXED&&(g.coord<lo[j]||g.coord>hi[j])){float sign=g.coord<lo[j]?1.f:-1.f,violation=g.coord<lo[j]?lo[j]-g.coord:g.coord-hi[j],vp[3],vc[3],rel[3],speed,den;if(type==REVOLUTE){for(int k=0;k<3;k++)rel[k]=c[10+k]-p[10+k];speed=dot3(rel,g.axis);den=aemass(p,ii+fp*3,c,ii+fc*3,g.axis);}else{pointv(p,g.rp,vp);pointv(c,g.rc,vc);for(int k=0;k<3;k++)rel[k]=vc[k]-vp[k];speed=dot3(rel,g.axis);den=lemass(p,im[fp],ii+fp*3,g.rp,c,im[fc],ii+fc*3,g.rc,g.axis);}float dl=(sign*fminf(2.f,violation*.2f/h)-speed)/fmaxf(den,1e-12f),old=lr[7],next=sign*fmaxf(0.f,sign*(old+dl));dl=next-old;lr[7]=next;float q[3],nq[3];for(int k=0;k<3;k++){q[k]=g.axis[k]*dl;nq[k]=-q[k];}if(type==REVOLUTE){aimpulse(p,im[fp],ii+fp*3,nq);aimpulse(c,im[fc],ii+fc*3,q);}else{impulse(p,im[fp],ii+fp*3,g.rp,nq);impulse(c,im[fc],ii+fc*3,g.rc,q);}limitactive[w*J+j]=1;}
   else lr[7]=0.f;
  }
  for(int j=0;j<J;j++){float*lr=lambda+j*8;motorout[w*J+j]+=lr[6];for(int row=0;row<8;row++)cache[(w*J+j)*8+row]=lr[row];int pi=int(ji[j*2]),ci=int(ji[j*2+1]),fp=w*B+pi,fc=w*B+ci,type=int(jt[j]);float*p=state+fp*SW,*c=state+fc*SW;G g=geometry(p,c,pa+j*3,ca+j*3,axis+j*3,ref+j*4,type);float lm=norm3(g.lin),sum=im[fp]+im[fc];if(lm>pslop&&sum>0){float corr=fminf(maxlin,(lm-pslop)*pc);for(int k=0;k<3;k++){float d=g.lin[k]*corr/lm;p[k]+=d*im[fp]/sum;c[k]-=d*im[fc]/sum;}}angularrepair(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,g.ang,pc,aslop,maxang);G after=geometry(p,c,pa+j*3,ca+j*3,axis+j*3,ref+j*4,type);if(type==PRISMATIC&&(after.coord<lo[j]||after.coord>hi[j])){float excess=after.coord-(after.coord<lo[j]?lo[j]:hi[j]),corr=fmaxf(-maxlin,fminf(maxlin,excess*pc));if(sum>0)for(int k=0;k<3;k++){p[k]+=after.axis[k]*corr*im[fp]/sum;c[k]-=after.axis[k]*corr*im[fc]/sum;}}else if(type==REVOLUTE&&(after.coord<lo[j]||after.coord>hi[j])){float targetq=after.coord<lo[j]?lo[j]:hi[j],e[3];for(int k=0;k<3;k++)e[k]=after.axis[k]*(after.coord-targetq);angularrepair(p,im[fp],ii+fp*3,c,im[fc],ii+fc*3,e,pc,aslop,maxang);}}
 }
 for(int j=0;j<J;j++){int pi=int(ji[j*2]),ci=int(ji[j*2+1]),fp=w*B+pi,fc=w*B+ci;G g=geometry(state+fp*SW,state+fc*SW,pa+j*3,ca+j*3,axis+j*3,ref+j*4,int(jt[j]));coordinate[w*J+j]=g.coord;linerr[w*J+j]=norm3(g.lin);angerr[w*J+j]=norm3(g.ang);limiterr[w*J+j]=fmaxf(0,fmaxf(lo[j]-g.coord,g.coord-hi[j]));}
}
#endif
}

#ifndef BOX3D_DEVICE_HELPERS_ONLY
std::vector<torch::Tensor> box3d_joint_step_cuda(torch::Tensor state,torch::Tensor inverse_mass,torch::Tensor inverse_inertia,torch::Tensor joint_indices,torch::Tensor joint_types,torch::Tensor parent_anchor_local,torch::Tensor child_anchor_local,torch::Tensor axis_parent,torch::Tensor reference_quaternion,torch::Tensor lower_limit,torch::Tensor upper_limit,torch::Tensor damping,torch::Tensor motor_enabled,torch::Tensor motor_target_velocity,torch::Tensor motor_target_position,torch::Tensor stiffness,torch::Tensor maximum_effort,torch::Tensor warm_start_cache,double warm_start_factor,double dt,int64_t substeps,double gravity_y,int64_t solver_iterations,double position_correction,double position_slop,double angular_slop,double maximum_linear_repair_m,double maximum_angular_repair_rad){const c10::cuda::CUDAGuard guard(state.device());auto out=state.clone(),cache=warm_start_cache.clone();int W=state.size(0),B=state.size(1),J=joint_indices.size(0);auto scalar=torch::zeros({W,J},state.options()),coordinate=scalar.clone(),lin=scalar.clone(),ang=scalar.clone(),limit=scalar.clone(),motor=scalar.clone();auto active=torch::zeros({W,J},state.options().dtype(torch::kUInt8));constexpr int T=64;kernel<<<(W+T-1)/T,T,0,at::cuda::getDefaultCUDAStream()>>>(out.data_ptr<float>(),inverse_mass.data_ptr<float>(),inverse_inertia.data_ptr<float>(),joint_indices.data_ptr<int64_t>(),joint_types.data_ptr<int64_t>(),parent_anchor_local.data_ptr<float>(),child_anchor_local.data_ptr<float>(),axis_parent.data_ptr<float>(),reference_quaternion.data_ptr<float>(),lower_limit.data_ptr<float>(),upper_limit.data_ptr<float>(),damping.data_ptr<float>(),motor_enabled.data_ptr<uint8_t>(),motor_target_velocity.data_ptr<float>(),motor_target_position.data_ptr<float>(),stiffness.data_ptr<float>(),maximum_effort.data_ptr<float>(),cache.data_ptr<float>(),float(warm_start_factor),coordinate.data_ptr<float>(),lin.data_ptr<float>(),ang.data_ptr<float>(),limit.data_ptr<float>(),motor.data_ptr<float>(),active.data_ptr<uint8_t>(),W,B,J,float(dt),int(substeps),float(gravity_y),int(solver_iterations),float(position_correction),float(position_slop),float(angular_slop),float(maximum_linear_repair_m),float(maximum_angular_repair_rad));C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,coordinate,lin,ang,limit,motor,active,cache};}
#endif
