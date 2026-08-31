// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define BCV_HD __host__ __device__
#else
#define BCV_HD
#endif
namespace bcv1_math {
struct V { float x{},y{},z{}; BCV_HD float& operator[](int i){return i==0?x:i==1?y:z;}
 BCV_HD float operator[](int i)const{return i==0?x:i==1?y:z;} };
struct Q { float x{},y{},z{},w{1}; };
BCV_HD inline V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
BCV_HD inline V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
BCV_HD inline V operator-(V a){return {-a.x,-a.y,-a.z};}
BCV_HD inline V operator*(V a,float s){return {a.x*s,a.y*s,a.z*s};}
BCV_HD inline V operator/(V a,float s){return a*(1/s);}
BCV_HD inline float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
BCV_HD inline V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
BCV_HD inline float length(V a){return sqrtf(dot(a,a));}
BCV_HD inline Q conj(Q q){return {-q.x,-q.y,-q.z,q.w};}
BCV_HD inline Q mul(Q a,Q b){V av{a.x,a.y,a.z},bv{b.x,b.y,b.z};V v=bv*a.w+av*b.w+cross(av,bv);return {v.x,v.y,v.z,a.w*b.w-dot(av,bv)};}
BCV_HD inline V rotate(Q q,V v){V u{q.x,q.y,q.z};V t=cross(u,v)*2;return v+t*q.w+cross(u,t);}
BCV_HD inline V inertia(Q q,V inverse_diagonal,V v){V l=rotate(conj(q),v);return rotate(q,{l.x*inverse_diagonal.x,l.y*inverse_diagonal.y,l.z*inverse_diagonal.z});}
BCV_HD inline V point_velocity(V v,V omega,V r){return v+cross(omega,r);}
BCV_HD inline float row_inverse(float im,Q q,V ii,V r,V a,V b){return im*dot(a,b)+dot(cross(r,a),inertia(q,ii,cross(r,b)));}
BCV_HD inline V load(const float* p){return {p[0],p[1],p[2]};}
BCV_HD inline Q loadq(const float* p){return {p[0],p[1],p[2],p[3]};}
BCV_HD inline void save(V a,float* p){p[0]=a.x;p[1]=a.y;p[2]=a.z;}
BCV_HD inline void saveq(Q a,float* p){p[0]=a.x;p[1]=a.y;p[2]=a.z;p[3]=a.w;}
} // namespace
#undef BCV_HD
