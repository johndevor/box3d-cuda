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
 double jr=0,nr=0,tr=0;
 auto certify=[&]{
  jr=nr=tr=0;
  for(size_t r=0;r<R;r++)if(kind[r]==0)jr=std::max(jr,scalar_error(r));
  for(uint32_t c1=0;c1<p.contacts;c1++){size_t r=p.contact[c1].first_row;nr=std::max(nr,scalar_error(r));tr=std::max(tr,tangent_error(r,p.contact[c1].friction));}
  return std::max({jr,nr,tr});
 };
 // --- Degenerate-contact repairs. Both are gated behind an unconverged,
 // stalled sweep history, so well-posed problems keep the exact ordinary
 // arithmetic; all convergence/momentum certificates below stay unchanged.
 auto correlation=[&](size_t i,size_t j){
  // Signed: only positively collinear normals (near-duplicated contacts) are
  // the documented degenerate pattern; opposing rows keep the ordinary path.
  double d2=K[i*R+i]*K[j*R+j];if(!(d2>0&&finite(d2)))return 0.;
  double s=.5*(K[i*R+j]+K[j*R+i])/std::sqrt(d2);return finite(s)?s:0.;
 };
 auto coupled_pair=[&](size_t i,size_t j){
  // Joint nonnegative solve of two nearly dependent normal rows, tangents
  // held fixed: enumerate the 2x2 KKT active sets plus a least-norm rank-1
  // candidate and keep the smallest complementarity error.
  const double aii=K[i*R+i],ajj=K[j*R+j],aij=.5*(K[i*R+j]+K[j*R+i]);
  const double qi=residual[i]-aii*lambda[i]-aij*lambda[j],qj=residual[j]-aij*lambda[i]-ajj*lambda[j];
  if(!(aii>0&&ajj>0&&finite(aij)&&finite(qi)&&finite(qj)))return;
  double bi=lambda[i],bj=lambda[j],be=std::numeric_limits<double>::infinity();
  auto consider=[&](double xi,double xj){
   if(!(finite(xi)&&finite(xj)))return;
   xi=std::max(0.,xi);xj=std::max(0.,xj);
   double ri=qi+aii*xi+aij*xj,rj=qj+aij*xi+ajj*xj;
   if(!(finite(ri)&&finite(rj)))return;
   double e=std::max(std::fabs(std::min(xi,ri)),std::fabs(std::min(xj,rj)));
   if(finite(e)&&e<be){be=e;bi=xi;bj=xj;}
  };
  consider(0.,0.);consider(-qi/aii,0.);consider(0.,-qj/ajj);
  const double det=aii*ajj-aij*aij;
  if(det>1e-12*aii*ajj)consider((aij*qj-ajj*qi)/det,(aij*qi-aii*qj)/det);
  else{double ui=std::sqrt(aii),uj=aij<0?-std::sqrt(ajj):std::sqrt(ajj),uu=aii+ajj;
   if(uu>0){double al=-(ui*qi+uj*qj)/(uu*uu);consider(al*ui,al*uj);}}
  const double cap=1e6*(1+std::fabs(lambda[i])+std::fabs(lambda[j])+std::fabs(qi)/aii+std::fabs(qj)/ajj);
  if(finite(be)&&bi<=cap&&bj<=cap){update(i,bi);update(j,bj);}
 };
 auto null_move=[&]{
  // One bounded feasible boundary move along a verified self-stress direction
  // of the contact block. Only an exactly rank-1-deficient block qualifies;
  // nullness is verified against the original jacobian rows and all response
  // rows, and the move is kept only if it lowers the residual certificate.
  const size_t m=size_t(p.contacts)*3;
  if(m<2||m>64||m>R)return;
  std::vector<double> snapl(lambda),snapr(residual);
  const double sj0=jr,sn0=nr,st0=tr;
  try{
   std::vector<size_t> rowsS(m);
   for(uint32_t c2=0;c2<p.contacts;c2++)for(size_t k=0;k<3;k++)rowsS[c2*3+k]=p.contact[c2].first_row+k;
   std::vector<double> A(m*m),V(m*m,0.),d(m);
   double scale=1e-30;
   for(size_t i=0;i<m;i++){V[i*m+i]=1;for(size_t j=0;j<m;j++)A[i*m+j]=.5*(K[rowsS[i]*R+rowsS[j]]+K[rowsS[j]*R+rowsS[i]]);scale=std::max(scale,std::fabs(A[i*m+i]));}
   for(int sweep=0;sweep<100;sweep++){
    double off=0;
    for(size_t i=0;i<m;i++)for(size_t j=i+1;j<m;j++)off=std::max(off,std::fabs(A[i*m+j]));
    if(!(off>1e-15*scale))break;
    for(size_t q=0;q<m;q++)for(size_t r2=q+1;r2<m;r2++){
     double apq=A[q*m+r2];if(!(std::fabs(apq)>1e-300))continue;
     double th=(A[r2*m+r2]-A[q*m+q])/(2*apq);if(!finite(th))return;
     double t2=(th>=0?1.:-1.)/(std::fabs(th)+std::sqrt(th*th+1));
     double c3=1/std::sqrt(t2*t2+1),s3=t2*c3;
     for(size_t k=0;k<m;k++){double aq=A[q*m+k],ar=A[r2*m+k];A[q*m+k]=c3*aq-s3*ar;A[r2*m+k]=s3*aq+c3*ar;}
     for(size_t k=0;k<m;k++){double aq=A[k*m+q],ar=A[k*m+r2];A[k*m+q]=c3*aq-s3*ar;A[k*m+r2]=s3*aq+c3*ar;
      double vq=V[k*m+q],vr=V[k*m+r2];V[k*m+q]=c3*vq-s3*vr;V[k*m+r2]=s3*vq+c3*vr;}
    }
   }
   size_t k0=m;double e0=std::numeric_limits<double>::infinity(),e1=std::numeric_limits<double>::infinity(),emax=0;
   for(size_t i=0;i<m;i++){double e=std::fabs(A[i*m+i]);if(!finite(e))return;emax=std::max(emax,e);
    if(e<e0){e1=e0;e0=e;k0=i;}else e1=std::min(e1,e);}
   // Nonsingular block or more than rank-1 deficiency: keep the ordinary path.
   if(k0>=m||!(emax>0)||!(e0<=1e-8*emax)||!(e1>=1e-5*emax))return;
   double dn=0;for(size_t i=0;i<m;i++){d[i]=V[i*m+k0];if(!finite(d[i]))return;dn+=d[i]*d[i];}
   if(!(dn>0&&finite(dn)))return;dn=std::sqrt(dn);for(size_t i=0;i<m;i++)d[i]/=dn;
   double js=1e-30,rs=1e-30;
   for(size_t i=0;i<m;i++)for(size_t k=0;k<N;k++){js=std::max(js,std::fabs(p.jacobian[rowsS[i]*N+k]));rs=std::max(rs,std::fabs(response[rowsS[i]*N+k]));}
   double jm=0,rm=0;
   for(size_t k=0;k<N;k++){double sj=0,sr=0;
    for(size_t i=0;i<m;i++){sj+=d[i]*p.jacobian[rowsS[i]*N+k];sr+=d[i]*response[rowsS[i]*N+k];}
    if(!(finite(sj)&&finite(sr)))return;jm=std::max(jm,std::fabs(sj));rm=std::max(rm,std::fabs(sr));}
   if(jm>1e-9*js||rm>1e-9*rs)return;
   double lmax=0;for(size_t i=0;i<m;i++)lmax=std::max(lmax,std::fabs(lambda[rowsS[i]]));
   double mmax=1e-30;for(size_t i=0;i<N*N;i++)mmax=std::max(mmax,std::fabs(p.mass[i]));
   // Bounded step: cap so the verified null defect cannot disturb the 1e-8
   // momentum certificate, then intersect bounds and every friction disk.
   double tmax=std::min(1e3*(1+lmax),1e-10/std::max({jm,rm*(1+mmax)*double(N),1e-30}));
   if(!(finite(tmax)&&tmax>0))return;
   double tlo=-tmax,thi=tmax;
   const double slack=1e-10*(1+lmax);
   for(uint32_t c2=0;c2<p.contacts;c2++){
    size_t r=p.contact[c2].first_row,i0=size_t(c2)*3;
    double dnn=d[i0],dx=d[i0+1],dy=d[i0+2],mu=p.contact[c2].friction;
    double ln=lambda[r],lx=lambda[r+1],ly=lambda[r+2];
    if(ln<-slack)return;
    if(std::fabs(dnn)>1e-14){double tb=-ln/dnn;if(dnn>0)tlo=std::max(tlo,std::min(tb,0.));else thi=std::min(thi,std::max(tb,0.));}
    double a2=dx*dx+dy*dy-mu*mu*dnn*dnn,b2=2*(lx*dx+ly*dy-mu*mu*ln*dnn),c4=lx*lx+ly*ly-mu*mu*ln*ln;
    if(!(finite(a2)&&finite(b2)&&finite(c4)))return;
    if(c4>slack*(1+mu*mu)*(1+lmax))return; // disk currently violated: ordinary path
    if(std::fabs(a2)<=1e-30){
     if(std::fabs(b2)>1e-30){double tb=-c4/b2;if(b2>0)thi=std::min(thi,std::max(tb,0.));else tlo=std::max(tlo,std::min(tb,0.));}
    }else{
     double disc=b2*b2-4*a2*c4;
     if(a2>0){if(disc<0)return;double sq=std::sqrt(std::max(0.,disc)),x1=(-b2-sq)/(2*a2),x2=(-b2+sq)/(2*a2);
      tlo=std::max(tlo,std::min(std::min(x1,x2),0.));thi=std::min(thi,std::max(std::max(x1,x2),0.));}
     else if(disc>0){double sq=std::sqrt(disc),x1=(-b2-sq)/(2*a2),x2=(-b2+sq)/(2*a2),r1=std::min(x1,x2),r2b=std::max(x1,x2);
      if(r1>=-slack)thi=std::min(thi,std::max(r1,0.));
      else if(r2b<=slack)tlo=std::max(tlo,std::min(r2b,0.));
      else return;}
    }
   }
   const double tiny=1e-12*(1+lmax);
   if(!(finite(tlo)&&finite(thi)&&tlo<=0&&thi>=0&&thi-tlo>tiny))return;
   const double cert0=std::max({jr,nr,tr});
   double bestt=0,beste=cert0;
   const double cand[2]={tlo,thi};
   for(int ci=0;ci<2;ci++){
    double t5=cand[ci];
    if(!finite(t5)||std::fabs(t5)<=tiny)continue;
    for(size_t i=0;i<m;i++){size_t r=rowsS[i];lambda[r]=clip(snapl[r]+t5*d[i],p.lower[r],p.upper[r]);}
    residuals();
    double e=certify();
    if(finite(e)&&e<beste){beste=e;bestt=t5;}
    lambda=snapl;residual=snapr;
   }
   if(bestt!=0){
    for(size_t i=0;i<m;i++){size_t r=rowsS[i];lambda[r]=clip(snapl[r]+bestt*d[i],p.lower[r],p.upper[r]);}
    residuals();
   }
  }catch(const Error&){lambda=snapl;residual=snapr;}
  jr=sj0;nr=sn0;tr=st0;
 };
 uint32_t apgd_budget=524288;   // total inner projected-gradient iterations
 auto block_accelerate=[&]{
  // Tresca fixed-point acceleration for stalled sweeps. Multi-point foot
  // contact produces exactly rank-deficient blocks with inconsistent
  // depenetration targets: per-row sweeps then drift along self-stress
  // directions indefinitely without certifying. With friction caps frozen at
  // the current normals the remaining problem is a convex box/disk QP whose
  // exact KKT conditions are precisely this solver's unchanged certificates,
  // so solving it (accelerated projected gradient with adaptive restart) and
  // refreshing the caps converges to a certifiable point even on those
  // blocks. The trial is kept ONLY if the unchanged residual certificate
  // strictly improves; any other outcome restores the ordinary path.
  if(R<1||R>96||!apgd_budget)return;
  std::vector<double> snapl(lambda),snapr(residual);
  const double sj0=jr,sn0=nr,st0=tr,cert0=std::max({jr,nr,tr});
  bool kept=false;
  try{
   double L=0;
   for(size_t i=0;i<R;i++){double s=0;for(size_t j=0;j<R;j++)s+=std::fabs(.5*(K[i*R+j]+K[j*R+i]));L=std::max(L,s);}
   require(finite(L)&&L>0,CIV1_NUMERIC);
   const double step=1/L;
   std::vector<double> x(lambda),y(lambda),xn(R),grad(R),bestx(lambda),cap(p.contacts);
   double beste=cert0;
   for(int outer=0;outer<64&&apgd_budget;outer++){
    for(uint32_t c2=0;c2<p.contacts;c2++)cap[c2]=p.contact[c2].friction*std::max(0.,x[p.contact[c2].first_row]);
    double t=1;y=x;
    for(int k=0;k<8192&&apgd_budget;k++){
     apgd_budget--;
     for(size_t i=0;i<R;i++){double s=base[i];for(size_t j=0;j<R;j++)s+=K[i*R+j]*y[j];grad[i]=s;}
     for(size_t i=0;i<R;i++){xn[i]=clip(y[i]-step*grad[i],p.lower[i],p.upper[i]);require(finite(xn[i]),CIV1_NUMERIC);}
     for(uint32_t c2=0;c2<p.contacts;c2++){
      const size_t r=p.contact[c2].first_row;
      const double n2=std::hypot(xn[r+1],xn[r+2]);
      if(n2>cap[c2]){const double f=cap[c2]>0?cap[c2]/n2:0;xn[r+1]*=f;xn[r+2]*=f;}
     }
     double dot=0,dn=0,xs=0;
     for(size_t i=0;i<R;i++){const double dx=xn[i]-x[i];dot+=(y[i]-xn[i])*dx;dn=std::max(dn,std::fabs(dx));xs=std::max(xs,std::fabs(xn[i]));}
     if(dot>0)t=1;                                  // adaptive restart
     const double tn=.5*(1+std::sqrt(1+4*t*t)),mo=(t-1)/tn;
     for(size_t i=0;i<R;i++){y[i]=xn[i]+mo*(xn[i]-x[i]);x[i]=xn[i];}
     t=tn;
     if(!(dn>1e-16*(1+xs)))break;                   // fixed point reached
    }
    lambda=x;residuals();
    const double e=certify();
    if(finite(e)&&e<beste){beste=e;bestx=x;}
    lambda=snapl;residual=snapr;
    if(beste<=p.impulse_tolerance)break;
   }
   if(beste<cert0){lambda=bestx;residuals();certify();kept=true;}
  }catch(const Error&){}
  if(!kept){lambda=snapl;residual=snapr;jr=sj0;nr=sn0;tr=st0;}
 };
 std::vector<double> wprev,wprev2;
 auto extrapolate=[&]{
  // Richardson extrapolation of the window-to-window creep: stalled sweeps
  // contract geometrically along a few slow (near-null) modes, so jumping to
  // the projected geometric-series limit shortcuts thousands of sweeps. The
  // candidate is kept ONLY if the unchanged certificate strictly improves.
  if(wprev.size()!=R||wprev2.size()!=R)return;
  std::vector<double> snapl(lambda),snapr(residual);
  const double sj0=jr,sn0=nr,st0=tr,cert0=std::max({jr,nr,tr});
  bool kept=false;
  try{
   double n1=0,n2=0,dot=0;
   for(size_t i=0;i<R;i++){
    const double d1=lambda[i]-wprev[i],d2=wprev[i]-wprev2[i];
    n1+=d1*d1;n2+=d2*d2;dot+=d1*d2;
   }
   n1=std::sqrt(n1);n2=std::sqrt(n2);
   if(!(n1>0&&n2>0&&finite(n1)&&finite(n2)&&finite(dot)))return;
   const double rho=std::min(n1/n2,.9999),ca=dot/(n1*n2);
   if(!(ca>.5&&rho>.3))return;
   const double f=rho/(1-rho);
   for(size_t i=0;i<R;i++){
    const double x=lambda[i]+f*(lambda[i]-wprev[i]);
    require(finite(x),CIV1_NUMERIC);
    lambda[i]=clip(x,p.lower[i],p.upper[i]);
   }
   for(uint32_t c2=0;c2<p.contacts;c2++){
    const size_t r=p.contact[c2].first_row;
    const double cp=p.contact[c2].friction*lambda[r],n3=std::hypot(lambda[r+1],lambda[r+2]);
    if(n3>cp){const double sc=cp>0?cp/n3:0;lambda[r+1]*=sc;lambda[r+2]*=sc;}
   }
   residuals();
   const double e=certify();
   if(finite(e)&&e<cert0)kept=true;
  }catch(const Error&){}
  if(!kept){lambda=snapl;residual=snapr;jr=sj0;nr=sn0;tr=st0;}
 };
 uint32_t iterations=0;bool converged=R==0,stalled=false,moved=false;
 size_t pair_i=R,pair_j=R;uint32_t accelerations=16;
 double reference=std::numeric_limits<double>::infinity();
 residuals();
 for(uint32_t it=0;it<p.max_iterations&&!converged;it++){
  if(it==256&&!moved){moved=true;null_move();}
  size_t c=0;
  for(size_t r=0;r<R;r++){
   if(kind[r]==2||kind[r]==3)continue;
   update(r,scalar(r));
   if(kind[r]==1){require(c<p.contacts&&p.contact[c].first_row==r);Disk d=conditional(r,p.contact[c++].friction);update(r+1,d.x);update(r+2,d.y);}
  }
  if(stalled&&pair_i<R&&pair_j<R){
   std::vector<double> snapl(lambda),snapr(residual);
   try{coupled_pair(pair_i,pair_j);}catch(const Error&){lambda=snapl;residual=snapr;}
  }
  // Recompute each sweep; accumulating updates alone can hide roundoff drift.
  residuals();
  double err=certify();
  iterations=it+1;converged=err<=p.impulse_tolerance;
  if(!converged&&(it&31u)==31u){
   // A repair that fails to improve a full window disables itself for good.
   if(pair_i<R&&err>.5*reference){pair_i=pair_j=R;}
   if(it>=63&&!stalled&&err>.25*reference){
    // Scalar sweeps stalled: enable the joint solve for the most correlated
    // (nearly linearly dependent through K) pair of contact normal rows.
    stalled=true;double best=.99;
    for(uint32_t a2=0;a2<p.contacts;a2++)for(uint32_t b2=a2+1;b2<p.contacts;b2++){
     size_t i2=p.contact[a2].first_row,j2=p.contact[b2].first_row;
     double corr=correlation(i2,j2);
     if(corr>=best){best=corr;pair_i=i2;pair_j=j2;}
    }
   }
   if(stalled){
    extrapolate();
    converged=std::max({jr,nr,tr})<=p.impulse_tolerance;  // unchanged certificate
    if(!converged&&accelerations&&err>.25*reference){
     accelerations--;block_accelerate();
     converged=std::max({jr,nr,tr})<=p.impulse_tolerance; // unchanged certificate
    }
   }
   wprev2.swap(wprev);wprev=lambda;
   reference=err;
  }
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
