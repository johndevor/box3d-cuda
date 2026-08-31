// SPDX-License-Identifier: MIT
#include "coupled_impulse_v1.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <vector>
namespace {
struct Error { int code; };
void require(bool ok,int code=CIV1_INVALID){if(!ok)throw Error{code};}
bool finite(double x){return std::isfinite(x);}
double clip(double x,double lo,double hi){return std::max(lo,std::min(hi,x));}
struct Disk {double x,y;};
Disk disk(double a,double b,double c,double fx,double fy,double cap){
 require(finite(cap)&&cap>=0,CIV1_NUMERIC);
 if(cap==0)return {0,0};
 // Positive definite 2x2 local response; the global contact matrix may be PSD.
 const double determinant=a*c-b*b;
 require(a>0&&c>0&&determinant>0&&finite(determinant),CIV1_NUMERIC);
 auto at=[&](double l){double d=(a+l)*(c+l)-b*b;require(finite(d)&&d>0,CIV1_NUMERIC);double nx=-(c+l)*fx+b*fy,ny=b*fx-(a+l)*fy;require(finite(nx)&&finite(ny),CIV1_NUMERIC);Disk v{nx/d,ny/d};require(finite(v.x)&&finite(v.y)&&finite(std::hypot(v.x,v.y)),CIV1_NUMERIC);return v;};
 Disk free=at(0);if(std::hypot(free.x,free.y)<=cap)return free;
 double lo=0,hi=std::max({a,c,std::hypot(fx,fy)/cap,1e-30});
 require(finite(hi),CIV1_NUMERIC);
 for(int k=0;k<128&&std::hypot(at(hi).x,at(hi).y)>cap;k++){hi*=2;require(finite(hi),CIV1_NUMERIC);}
 require(std::hypot(at(hi).x,at(hi).y)<=cap,CIV1_NUMERIC);
 for(int k=0;k<64;k++){double mid=lo+(hi-lo)*.5;Disk v=at(mid);if(std::hypot(v.x,v.y)>cap)lo=mid;else hi=mid;}
 return at(hi);
}
int solve(const civ1_problem& p,civ1_result& out){
 const size_t N=p.dofs,R=p.rows;
 // Raised for multi-body cube-grid islands: N covers duck DOFs plus free
 // bodies (6 each); dense chol/K stay tractable at this scale on CPU.
 require(N&&N<=256&&R<=1536&&p.contacts<=512&&p.max_iterations&&p.max_iterations<=16384);
 require(finite(p.impulse_tolerance)&&p.impulse_tolerance>0&&p.impulse_tolerance<=1e-5);
 require(p.mass&&p.smooth_velocity&&out.velocity&&(!R||(p.jacobian&&p.target&&p.regularizer&&p.lower&&p.upper&&out.impulse))&&(!p.contacts||p.contact));
 std::vector<double> chol(N*N,0),response(R*N),K(R*R),lambda(R),base(R),v(N),residual(R);
 std::vector<int> kind(R,0);
 uint32_t end=0;
 for(uint32_t c=0;c<p.contacts;c++){auto x=p.contact[c];require(x.first_row>=end&&size_t(x.first_row)+3<=R&&finite(x.friction)&&x.friction>=0&&x.friction<=4);end=x.first_row+3;kind[x.first_row]=1;kind[x.first_row+1]=2;kind[x.first_row+2]=3;}
 for(size_t i=0;i<N;i++){
  require(finite(p.smooth_velocity[i]));
  for(size_t j=0;j<N;j++)require(finite(p.mass[i*N+j])&&std::fabs(p.mass[i*N+j]-p.mass[j*N+i])<=1e-12*(1+std::fabs(p.mass[i*N+j])));
  for(size_t j=0;j<=i;j++){double s=p.mass[i*N+j];for(size_t k=0;k<j;k++)s-=chol[i*N+k]*chol[j*N+k];if(i==j){require(s>0&&finite(s),CIV1_NUMERIC);chol[i*N+j]=std::sqrt(s);}else chol[i*N+j]=s/chol[j*N+j];}
 }
 for(size_t r=0;r<R;r++){
  require(finite(p.target[r])&&finite(p.regularizer[r])&&p.regularizer[r]>=0&&!std::isnan(p.lower[r])&&!std::isnan(p.upper[r])&&p.lower[r]<=p.upper[r]);
  require(p.lower[r]!=std::numeric_limits<double>::infinity()&&p.upper[r]!=-std::numeric_limits<double>::infinity());
  if(kind[r]){require(p.regularizer[r]==0);if(kind[r]==1)require(p.lower[r]==0&&p.upper[r]==std::numeric_limits<double>::infinity());else require(p.lower[r]==-std::numeric_limits<double>::infinity()&&p.upper[r]==std::numeric_limits<double>::infinity());}
  base[r]=-p.target[r];
  for(size_t j=0;j<N;j++){double x=p.jacobian[r*N+j];require(finite(x));base[r]+=x*p.smooth_velocity[j];double s=x;for(size_t k=0;k<j;k++)s-=chol[j*N+k]*response[r*N+k];response[r*N+j]=s/chol[j*N+j];}
  for(size_t j=N;j-->0;){double s=response[r*N+j];for(size_t k=j+1;k<N;k++)s-=chol[k*N+j]*response[r*N+k];response[r*N+j]=s/chol[j*N+j];}
  double w=p.warm?p.warm[r]:0;require(finite(w));lambda[r]=clip(w,p.lower[r],p.upper[r]);
 }
 for(size_t i=0;i<R;i++)for(size_t j=0;j<R;j++){double x=i==j?p.regularizer[i]:0;for(size_t k=0;k<N;k++)x+=p.jacobian[i*N+k]*response[j*N+k];require(finite(x),CIV1_NUMERIC);K[i*R+j]=x;}
 for(size_t i=0;i<R;i++)require(K[i*R+i]>0||(p.lower[i]==0&&p.upper[i]==0),CIV1_NUMERIC);
 for(uint32_t c=0;c<p.contacts;c++){size_t r=p.contact[c].first_row;double cap=p.contact[c].friction*lambda[r],norm=std::hypot(lambda[r+1],lambda[r+2]);if(norm>cap){lambda[r+1]*=cap/norm;lambda[r+2]*=cap/norm;}}
 auto residuals=[&]{for(size_t i=0;i<R;i++){double x=base[i];for(size_t j=0;j<R;j++)x+=K[i*R+j]*lambda[j];residual[i]=x;require(finite(x),CIV1_NUMERIC);}};
 auto scalar=[&](size_t r){if(p.lower[r]==p.upper[r])return p.lower[r];double candidate=lambda[r]-residual[r]/K[r*R+r];require(finite(candidate),CIV1_NUMERIC);return clip(candidate,p.lower[r],p.upper[r]);};
 auto scalar_error=[&](size_t r){
  if(p.lower[r]==p.upper[r])return std::fabs(lambda[r]-p.lower[r]);
  if(residual[r]==0)return 0.;
  double correction=residual[r]/K[r*R+r];require(finite(correction),CIV1_NUMERIC);
  double distance=correction>0?lambda[r]-p.lower[r]:p.upper[r]-lambda[r];
  require(!std::isnan(distance)&&distance>=0,CIV1_NUMERIC);
  double error=std::min(std::fabs(correction),distance);require(finite(error),CIV1_NUMERIC);return error;
 };
 auto conditional=[&](size_t r,double mu){size_t a=r+1,b=r+2;double aa=K[a*R+a],ab=.5*(K[a*R+b]+K[b*R+a]),bb=K[b*R+b];double fx=residual[a]-aa*lambda[a]-ab*lambda[b],fy=residual[b]-ab*lambda[a]-bb*lambda[b];return disk(aa,ab,bb,fx,fy,mu*lambda[r]);};
 auto update=[&](size_t r,double x){double delta=x-lambda[r];require(finite(x)&&finite(delta),CIV1_NUMERIC);lambda[r]=x;for(size_t i=0;i<R;i++){residual[i]+=K[i*R+r]*delta;require(finite(residual[i]),CIV1_NUMERIC);}};
 auto tangent_error=[&](size_t r,double mu){
  size_t a=r+1,b=r+2;double cap=mu*lambda[r],norm=std::hypot(lambda[a],lambda[b]);require(finite(cap)&&finite(norm),CIV1_NUMERIC);
  if(cap==0)return norm;
  double aa=K[a*R+a],ab=.5*(K[a*R+b]+K[b*R+a]),bb=K[b*R+b],det=aa*bb-ab*ab;
  double largest=.5*(aa+bb+std::hypot(aa-bb,2*ab)),smallest=det/largest;require(finite(smallest)&&smallest>0,CIV1_NUMERIC);
  // Stable conditional KKT certificate, without subtracting two large
  // impulses. MIN eigenvalue bounds anisotropic error conservatively; using
  // the maximum eigenvalue could hide a large low-inertia-axis correction.
  double gx=residual[a],gy=residual[b],stationarity=0,complement=0;
  if(norm==0)stationarity=std::hypot(gx,gy)/smallest;
  else {double ux=lambda[a]/norm,uy=lambda[b]/norm,radial=gx*ux+gy*uy;require(finite(radial),CIV1_NUMERIC);double m=std::max(0.,-radial);stationarity=std::hypot(gx+m*ux,gy+m*uy)/smallest;complement=(m/smallest)*(std::max(0.,cap-norm)/norm);}
  require(finite(stationarity)&&finite(complement),CIV1_NUMERIC);return std::max({stationarity,complement,std::max(0.,norm-cap)});
 };
 double jr=0,nr=0,tr=0;uint32_t iterations=0;bool converged=R==0;
 residuals();
 for(uint32_t it=0;it<p.max_iterations&&!converged;it++){
  size_t c=0;
  for(size_t r=0;r<R;r++){
   if(kind[r]==2||kind[r]==3)continue;
   update(r,scalar(r));
   if(kind[r]==1){require(c<p.contacts&&p.contact[c].first_row==r);Disk d=conditional(r,p.contact[c++].friction);update(r+1,d.x);update(r+2,d.y);}
  }
  // Recompute each sweep; accumulating updates alone can hide roundoff drift.
  residuals();jr=nr=tr=0;
  for(size_t r=0;r<R;r++)if(kind[r]==0)jr=std::max(jr,scalar_error(r));
  for(uint32_t c1=0;c1<p.contacts;c1++){size_t r=p.contact[c1].first_row;nr=std::max(nr,scalar_error(r));tr=std::max(tr,tangent_error(r,p.contact[c1].friction));}
  iterations=it+1;converged=std::max({jr,nr,tr})<=p.impulse_tolerance;
 }
 require(converged,CIV1_NO_CONVERGENCE);
 for(size_t k=0;k<N;k++){double x=p.smooth_velocity[k];for(size_t r=0;r<R;r++)x+=response[r*N+k]*lambda[r];require(finite(x),CIV1_NUMERIC);v[k]=x;}
 double mr=0;for(size_t k=0;k<N;k++){double x=0;for(size_t j=0;j<N;j++){double term=p.mass[k*N+j]*(v[j]-p.smooth_velocity[j]);require(finite(term),CIV1_NUMERIC);x+=term;require(finite(x),CIV1_NUMERIC);}for(size_t r=0;r<R;r++){double term=p.jacobian[r*N+k]*lambda[r];require(finite(term),CIV1_NUMERIC);x-=term;require(finite(x),CIV1_NUMERIC);}mr=std::max(mr,std::fabs(x));}
 require(finite(mr)&&mr<=1e-8,CIV1_NUMERIC);
 std::copy(v.begin(),v.end(),out.velocity);if(R)std::copy(lambda.begin(),lambda.end(),out.impulse);
 out.iterations=iterations;out.joint_residual=jr;out.normal_residual=nr;out.tangent_residual=tr;out.momentum_residual=mr;return CIV1_OK;
}
}
extern "C" int civ1_solve(const civ1_problem* p,civ1_result* out){if(!p||!out)return CIV1_INVALID;try{return solve(*p,*out);}catch(Error e){return e.code;}catch(const std::bad_alloc&){return CIV1_ALLOCATION;}catch(...){return CIV1_NUMERIC;}}
