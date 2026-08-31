// SPDX-License-Identifier: MIT
// Experimental generalized dynamics. No legacy kernels, layouts or caches used.
#pragma once
#include <cmath>
#include <cstddef>

namespace box3d_joint_v1 {
constexpr int max_dofs=32, max_rows=64;
template<class T> inline T clamp(T x,T lo,T hi){return x<lo?lo:(x>hi?hi:x);}
template<class T> inline bool finite(T x){return std::isfinite(x);}

// Dense Cholesky, with an explicit relative conditioning rejection threshold.
// No diagonal-only response or isolated motor denominator approximation.
template<class T> bool factor(int n,const T* m,T* l){
 T scale=0;
 for(int i=0;i<n;i++){
  if(!finite(m[i*n+i])||m[i*n+i]<=0)return false;
  if(m[i*n+i]>scale)scale=m[i*n+i];
 }
 for(int i=0;i<n*n;i++)l[i]=0;
 for(int i=0;i<n;i++)for(int j=0;j<=i;j++){
  if(!finite(m[i*n+j])||!finite(m[j*n+i]) ||
     std::abs(m[i*n+j]-m[j*n+i])>T(2e-6)*scale)return false;
  T sum=m[i*n+j];
  for(int k=0;k<j;k++)sum-=l[i*n+k]*l[j*n+k];
  if(i==j){if(!finite(sum)||sum<=T(1e-9)*scale)return false;l[i*n+j]=std::sqrt(sum);}
  else {l[i*n+j]=sum/l[j*n+j];if(!finite(l[i*n+j]))return false;}
 }
 return true;
}
template<class T> void solve(int n,const T* l,const T* b,T* x){
 for(int i=0;i<n;i++){T v=b[i];for(int j=0;j<i;j++)v-=l[i*n+j]*x[j];x[i]=v/l[i*n+i];}
 for(int i=n-1;i>=0;i--){T v=x[i];for(int j=i+1;j<n;j++)v-=l[j*n+i]*x[j];x[i]=v/l[i*n+i];}
}
template<class T> bool inverse(int n,const T* m,T* out){
 T l[max_dofs*max_dofs],b[max_dofs],x[max_dofs];
 if(n<1||n>max_dofs||!factor(n,m,l))return false;
 for(int j=0;j<n;j++){
  for(int i=0;i<n;i++)b[i]=T(i==j);
  solve(n,l,b,x);for(int i=0;i<n;i++){if(!finite(x[i]))return false;out[i*n+j]=x[i];}
  if(out[j*n+j]<=T(0))return false;
 }
 return true;
}
template<class T> void matvec(int n,const T* m,const T* b,T* x){
 for(int i=0;i<n;i++){T sum=0;for(int j=0;j<n;j++)sum+=m[i*n+j]*b[j];x[i]=sum;}
}
template<class T> void armature_mass(int n,const T* body,const T* armature,T* total){
 for(int i=0;i<n*n;i++)total[i]=body[i];
 for(int i=0;i<n;i++)total[i*n+i]+=armature[i];
}

// Body spatial Jacobian: [body,6,dof], world linear xyz then world angular xyz.
// R rotates local principal-frame vectors into world. Body inertia is unchanged.
template<class T> void accumulate_body_mass(int n,T mass,const T* principal,
                                           const T* rotation,const T* jac,T* m){
 for(int i=0;i<n;i++)for(int j=0;j<n;j++){
  T sum=0;
  for(int k=0;k<3;k++)sum+=mass*jac[k*n+i]*jac[k*n+j];
  for(int k=0;k<3;k++){
   T wi=0,wj=0;
   for(int a=0;a<3;a++){wi+=rotation[a*3+k]*jac[(a+3)*n+i];wj+=rotation[a*3+k]*jac[(a+3)*n+j];}
   sum+=principal[k]*wi*wj;
  }
  m[i*n+j]+=sum;
 }
}
template<class T> void wrench_to_generalized(int n,const T* jac,const T* wrench,T* tau){
 for(int i=0;i<n;i++)for(int k=0;k<6;k++)tau[i]+=jac[k*n+i]*wrench[k];
}
template<class T> void body_velocity(int n,const T* jac,const T* velocity,T* spatial){
 for(int k=0;k<6;k++){spatial[k]=0;for(int i=0;i<n;i++)spatial[k]+=jac[k*n+i]*velocity[i];}
}

// Friction has zero position residual: d=d0; K=0; B=2/(dwidth*timeconst).
// invweight0 is independently computed diag((M_body(q0)+A)^-1), fixed at create.
// Only positive solref time constant, dampratio=1 and zero-residual d0 supported.
template<class T> void friction_coefficients(T dt,T d0,T dwidth,T timeconst,
                                             T invweight0,T velocity,T& r,T& aref){
 const T tc=timeconst<T(2)*dt?T(2)*dt:timeconst;
 r=(T(1)-d0)/d0*invweight0;
 aref=-T(2)/(dwidth*tc)*velocity;
}

// Solve min .5 f'Hf + b'f subject to lo<=f<=hi. Coupled PGS, deterministic
// row order, bounded iterations, explicit projected-coordinate residual.
template<class T> bool box_qp(int count,const T* h,const T* b,const T* lo,const T* hi,
                             int iterations,T tolerance,T* f,T& residual,int& used){
 for(int i=0;i<count;i++){
  if(!finite(h[i*count+i])||h[i*count+i]<=T(0))return false;
  f[i]=clamp(f[i],lo[i],hi[i]);
 }
 residual=0;used=0;
 for(int it=0;it<iterations;it++){
  for(int i=0;i<count;i++){
   double g=double(b[i]);for(int j=0;j<count;j++)g+=double(h[i*count+j])*double(f[j]);
   if(!finite(g))return false;
   f[i]=T(clamp(double(f[i])-g/double(h[i*count+i]),double(lo[i]),double(hi[i])));
   if(!finite(f[i]))return false;
  }
  double checked_residual=0;used=it+1;
  for(int i=0;i<count;i++){
   double g=double(b[i]);for(int j=0;j<count;j++)g+=double(h[i*count+j])*double(f[j]);
   if(!finite(g))return false;
   double err=std::abs(double(f[i])-clamp(double(f[i])-g/double(h[i*count+i]),double(lo[i]),double(hi[i])));
   if(!finite(err))return false;if(err>checked_residual)checked_residual=err;
  }
  residual=T(checked_residual);
  if(checked_residual<=double(tolerance))return true;
 }
 return count==0;
}
} // namespace box3d_joint_v1
