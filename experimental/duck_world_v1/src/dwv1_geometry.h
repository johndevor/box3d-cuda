// SPDX-License-Identifier: MIT
// Internal manifold geometry for duck_world_v1, ported from
// experimental/contact_v1/src/contact_v1.cpp (same conventions: f32 geometry,
// SAT with face clipping, stable feature ids, <=4 point manifolds). Ported
// rather than linked because those functions live in contact_v1's anonymous
// namespace and its scene ABI caps bodies at 32/pairs at 16, far below a cube
// grid. Any behavioural divergence from contact_v1 is a bug here.
#pragma once
#include "contact_math.h"
#include "contact_v1.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <vector>
namespace dwv1geo {
using namespace bcv1_math;
constexpr float EPS=2e-6f;
struct Fail { bcv1_status status; };
inline void geo_need(bool b,bcv1_status s=BCV1_INVALID){if(!b)throw Fail{s};}
inline bool finite(float x){return std::isfinite(x);}
inline void valid(V v,float bound=1e4f){geo_need(finite(v.x)&&finite(v.y)&&finite(v.z)&&length(v)<=bound);}
inline V unit(V v){float n=length(v);geo_need(finite(n)&&n>1e-12f,BCV1_NUMERIC);return v/n;}
inline Q normalized(Q q){float n=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);geo_need(finite(n)&&n>1e-12f,BCV1_NUMERIC);return {q.x/n,q.y/n,q.z/n,q.w/n};}
inline V pos(const bcv1_body& b){return load(b.state);}
inline Q quat(const bcv1_body& b){return normalized(loadq(b.state+3));}
inline V world(const bcv1_body& b,V v){return pos(b)+rotate(quat(b),v);}
inline void basis(V n,V& u,V& v){V s=std::fabs(n.x)<=std::fabs(n.y)&&std::fabs(n.x)<=std::fabs(n.z)?V{1,0,0}:std::fabs(n.y)<=std::fabs(n.z)?V{0,1,0}:V{0,0,1};u=unit(cross(s,n));v=cross(n,u);}
struct Face { V n;std::vector<int> ids; };
struct Hull { std::vector<V> v;std::vector<Face> faces;std::vector<std::array<int,2>> edges; };
struct D {double x,y,z;};
inline D d(V a){return {a.x,a.y,a.z};}
inline D sub(D a,D b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline D cr(D a,D b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double dt(D a,D b){return a.x*b.x+a.y*b.y+a.z*b.z;}
// Registration-only convex hull construction (double predicates on copied
// float inputs; plane tolerance relative to diameter). Same as contact_v1.
inline Hull build_hull(const float (*vertex)[3],uint32_t vertex_count){
 geo_need(vertex_count>=4&&vertex_count<=32,BCV1_CAPACITY);Hull h;
 for(uint32_t i=0;i<vertex_count;i++){V p=load(vertex[i]);valid(p,100);h.v.push_back(p);}
 double diameter=0;for(auto a:h.v)for(auto b:h.v)diameter=std::max(diameter,double(length(a-b)));
 geo_need(diameter>=1e-4&&diameter<=100);double tol=diameter*2e-7;
 for(size_t i=0;i<h.v.size();i++)for(size_t j=i+1;j<h.v.size();j++)geo_need(length(h.v[i]-h.v[j])>tol);
 std::set<uint32_t> seen;
 for(int i=0;i<int(h.v.size());i++)for(int j=i+1;j<int(h.v.size());j++)for(int k=j+1;k<int(h.v.size());k++){
  D n=cr(sub(d(h.v[j]),d(h.v[i])),sub(d(h.v[k]),d(h.v[i])));double l=std::sqrt(dt(n,n));if(l<=diameter*diameter*1e-10)continue;
  n={n.x/l,n.y/l,n.z/l};double lo=0,hi=0;
  for(auto p:h.v){double x=dt(n,sub(d(p),d(h.v[i])));lo=std::min(lo,x);hi=std::max(hi,x);}
  if(lo < -tol && hi > tol)continue;
  if(hi<=tol){/* outward */}else n={-n.x,-n.y,-n.z};
  uint32_t mask=0;std::vector<int> ids;
  for(int a=0;a<int(h.v.size());a++)if(std::fabs(dt(n,sub(d(h.v[a]),d(h.v[i]))))<=tol){mask|=uint32_t(1)<<a;ids.push_back(a);}
  if(!seen.insert(mask).second)continue;
  V fn=unit({float(n.x),float(n.y),float(n.z)}),u,v;basis(fn,u,v);
  std::sort(ids.begin(),ids.end(),[&](int a,int b){float ax=dot(h.v[a],u),bx=dot(h.v[b],u);if(ax!=bx)return ax<bx;float ay=dot(h.v[a],v),by=dot(h.v[b],v);return ay!=by?ay<by:a<b;});
  auto turn=[&](int a,int b,int c){D ab=sub(d(h.v[b]),d(h.v[a])),ac=sub(d(h.v[c]),d(h.v[a]));return dt(cr(ab,ac),n);};
  std::vector<int> poly;
  for(int a:ids){while(poly.size()>=2&&turn(poly[poly.size()-2],poly.back(),a)<=tol*diameter)poly.pop_back();poly.push_back(a);}
  size_t lower=poly.size();for(int z=int(ids.size())-2;z>=0;z--){int a=ids[z];while(poly.size()>lower&&turn(poly[poly.size()-2],poly.back(),a)<=tol*diameter)poly.pop_back();poly.push_back(a);}if(poly.size()>1)poly.pop_back();
  if(poly.size()<3)continue;h.faces.push_back({fn,poly});geo_need(h.faces.size()<=60,BCV1_CAPACITY);
 }
 geo_need(h.faces.size()>=4);double volume=0;std::map<std::array<int,2>,int> edge_counts;
 for(auto& f:h.faces){V a=h.v[f.ids[0]];for(size_t i=1;i+1<f.ids.size();i++)volume+=dt(d(a),cr(d(h.v[f.ids[i]]),d(h.v[f.ids[i+1]])))/6;
  for(size_t i=0;i<f.ids.size();i++){int a1=f.ids[i],b=f.ids[(i+1)%f.ids.size()];edge_counts[{std::min(a1,b),std::max(a1,b)}]++;}}
 geo_need(volume>diameter*diameter*diameter*1e-7);for(auto x:edge_counts){geo_need(x.second==2);h.edges.push_back(x.first);}geo_need(h.edges.size()<=90,BCV1_CAPACITY);
 return h;
}
inline std::vector<V> vertices(const Hull& h,const bcv1_body& b){std::vector<V> r;for(auto p:h.v){V x=world(b,p);valid(x,2e4);r.push_back(x);}return r;}
inline uint64_t feature(uint64_t a,uint64_t b){return (a^b)*1099511628211ull;}
struct Candidate {V p;float depth;uint64_t id;};
inline bcv1_manifold reduce(V n,std::vector<Candidate> pts){
 bcv1_manifold m{};if(pts.empty())return m;V u,v;basis(n,u,v);save(n,m.normal);save(u,m.tangent1);save(v,m.tangent2);
 std::sort(pts.begin(),pts.end(),[](auto a,auto b){return a.id<b.id;});std::vector<Candidate> unique;
 for(auto p:pts){bool duplicate=false;for(auto q:unique)if(length(p.p-q.p)<EPS*.5f){duplicate=true;break;}if(!duplicate)unique.push_back(p);}
 geo_need(unique.size()<=64,BCV1_CAPACITY);std::vector<size_t> chosen;
 auto choose=[&](size_t x){if(std::find(chosen.begin(),chosen.end(),x)==chosen.end())chosen.push_back(x);};
 if(unique.size()<=4){for(size_t i=0;i<unique.size();i++)choose(i);}else{
  size_t deep=0;for(size_t i=1;i<unique.size();i++)if(unique[i].depth>unique[deep].depth)deep=i;choose(deep);
  while(chosen.size()<4){float best=-1;size_t bi=0;for(size_t i=0;i<unique.size();i++){if(std::find(chosen.begin(),chosen.end(),i)!=chosen.end())continue;float score=1e30f;for(auto j:chosen){V diff=unique[i].p-unique[j].p;score=std::min(score,dot(diff,diff));}if(score>best){best=score;bi=i;}}choose(bi);}
 }
 std::sort(chosen.begin(),chosen.end(),[&](size_t a,size_t b){return unique[a].id<unique[b].id;});m.count=uint32_t(chosen.size());
 for(size_t i=0;i<chosen.size();i++){auto p=unique[chosen[i]];m.points[i].feature=p.id;save(p.p,m.points[i].point);m.points[i].depth=p.depth;}return m;
}
// Hull against a fixed world half-space n.dot(x)<=offset (unit n). The
// returned manifold normal points from pair body A to pair body B; the hull
// is A when plane_is_a is false.
inline bcv1_manifold plane_manifold(const Hull& h,const bcv1_body& c,V n,float offset,bool plane_is_a){
 auto vs=vertices(h,c);std::vector<Candidate> pts;
 for(size_t i=0;i<vs.size();i++){float sep=dot(n,vs[i])-offset;if(sep<=EPS){float depth=std::max(0.f,-sep);pts.push_back({vs[i]-n*(sep*.5f),depth,0x100000000ull+i+1});}}
 return reduce(plane_is_a?n:-n,pts);
}
struct SAT {bool hit{};V n{};float depth{1e30f};int kind{},a{},b{};};
inline SAT sat(const Hull& ah,const bcv1_body& a,const std::vector<V>& av,const Hull& bh,const bcv1_body& b,const std::vector<V>& bv){
 SAT best;best.hit=true;
 auto axis=[&](V n,int kind,int ia,int ib){float len=length(n);if(len<1e-6f)return true;n=n/len;
  float amin=1e30f,amax=-1e30f,bmin=1e30f,bmax=-1e30f;for(auto v:av){float x=dot(n,v);amin=std::min(amin,x);amax=std::max(amax,x);}for(auto v:bv){float x=dot(n,v);bmin=std::min(bmin,x);bmax=std::max(bmax,x);}
  float positive=amax-bmin,negative=bmax-amin;if(positive < -EPS||negative < -EPS)return false;
  float depth=positive;if(kind==2&&negative<positive){depth=negative;n=-n;}
  if(depth<best.depth-EPS){best.n=n;best.depth=std::max(0.f,depth);best.kind=kind;best.a=ia;best.b=ib;}return true;};
 for(size_t i=0;i<ah.faces.size();i++)if(!axis(rotate(quat(a),ah.faces[i].n),0,int(i),0))return {};
 for(size_t i=0;i<bh.faces.size();i++)if(!axis(-rotate(quat(b),bh.faces[i].n),1,int(i),0))return {};
 for(size_t i=0;i<ah.edges.size();i++)for(size_t j=0;j<bh.edges.size();j++){auto x=ah.edges[i],y=bh.edges[j];if(!axis(cross(unit(av[x[1]]-av[x[0]]),unit(bv[y[1]]-bv[y[0]])),2,int(i),int(j)))return {};}
 return best;
}
struct Clip {V p;uint64_t id;};
inline std::vector<Clip> clip(const std::vector<Clip>& in,V n,float offset,uint64_t tag){
 std::vector<Clip> out;if(in.empty())return out;Clip prev=in.back();float dp=dot(n,prev.p)-offset;
 for(auto cur:in){float dc=dot(n,cur.p)-offset;bool ip=dp<=EPS,ic=dc<=EPS;if(ip!=ic){float f=dp/(dp-dc);f=std::max(0.f,std::min(1.f,f));out.push_back({prev.p+(cur.p-prev.p)*f,feature(feature(std::min(prev.id,cur.id),std::max(prev.id,cur.id)),tag)});}if(ic)out.push_back(cur);prev=cur;dp=dc;geo_need(out.size()<=64,BCV1_CAPACITY);}return out;
}
inline std::array<V,2> closest(V p,V q,V r,V s){
 D d1=sub(d(q),d(p)),d2=sub(d(s),d(r)),dr=sub(d(p),d(r));double a=dt(d1,d1),e=dt(d2,d2),b=dt(d1,d2),c=dt(d1,dr),f=dt(d2,dr),den=a*e-b*b;
 double u=den>a*e*1e-15?std::max(0.,std::min(1.,(b*f-c*e)/den)):0;
 double v=(b*u+f)/e;if(v<0){v=0;u=std::max(0.,std::min(1.,-c/a));}else if(v>1){v=1;u=std::max(0.,std::min(1.,(b-c)/a));}
 return {V{float(p.x+d1.x*u),float(p.y+d1.y*u),float(p.z+d1.z*u)},V{float(r.x+d2.x*v),float(r.y+d2.y*v),float(r.z+d2.z*v)}};
}
// Convex-convex SAT manifold; the returned normal points from A to B.
inline bcv1_manifold convex_contact(const Hull& ah,const bcv1_body& a,const Hull& bh,const bcv1_body& b){
 auto av=vertices(ah,a),bv=vertices(bh,b);SAT s=sat(ah,a,av,bh,b,bv);if(!s.hit)return {};std::vector<Candidate> pts;
 if(s.kind==2){
  float am=-1e30f,bm=1e30f;for(auto v:av)am=std::max(am,dot(s.n,v));for(auto v:bv)bm=std::min(bm,dot(s.n,v));
  float distance=1e30f;V cp{};uint64_t id=0;
  const float reach=s.depth+2e-4f;
  // contact_v1 uses the 4*EPS support band only. Near-parallel grazing
  // hull-hull contact (the duck foot edge-on a cube edge, walker corpus
  // 20260901T18*) starves it three ways: the band is empty (the sole facets
  // tilt a few mrad-0.3 rad off the axis, pushing edge endpoints of the true
  // witness outside any fixed band), the band holds only a FAR edge pair, or
  // a far pair found at a narrow rung used to end the search prematurely.
  // Repair discipline: the band is only a CANDIDATE prefilter, so widen it
  // progressively and finish with an exhaustive edge-pair scan; acceptance
  // stays the unchanged physical certificate (witness distance <= reach =
  // depth + 2e-4), which wider candidate sets can only satisfy honestly. The
  // global best is kept across rungs; a deep axis with NO edge pair within
  // reach anywhere remains the hard numeric failure it always was.
  auto scan=[&](float tol,bool banded){
   for(size_t i=0;i<ah.edges.size();i++){auto ae=ah.edges[i];if(banded&&(std::fabs(dot(s.n,av[ae[0]])-am)>tol||std::fabs(dot(s.n,av[ae[1]])-am)>tol))continue;
    for(size_t j=0;j<bh.edges.size();j++){auto be=bh.edges[j];if(banded&&(std::fabs(dot(s.n,bv[be[0]])-bm)>tol||std::fabs(dot(s.n,bv[be[1]])-bm)>tol))continue;
     auto x=closest(av[ae[0]],av[ae[1]],bv[be[0]],bv[be[1]]);float d2=dot(x[0]-x[1],x[0]-x[1]);if(d2<distance){distance=d2;cp=(x[0]+x[1])*.5f;id=0x200000000ull+(i<<16)+j+1;}}
   }
  };
  for(float tol:{4*EPS,64*EPS}){scan(tol,true);if(id&&distance<=reach*reach)break;}
  if(!(id&&distance<=reach*reach))scan(0,false);
  if(id==0||distance>reach*reach){geo_need(s.depth<=8*EPS,BCV1_NUMERIC);return {};}
  pts.push_back({cp,s.depth,id});return reduce(s.n,pts);
 }
 bool rb=s.kind==1;auto& rh=rb?bh:ah;auto& ih=rb?ah:bh;auto& rv=rb?bv:av;auto& iv=rb?av:bv;auto& rbody=rb?b:a;auto& ibody=rb?a:b;V outward=rb?-s.n:s.n;
 int rf=s.a,inf=0;geo_need(dot(rotate(quat(rbody),rh.faces[rf].n),outward)>1-1e-5f,BCV1_NUMERIC);
 for(size_t i=1;i<ih.faces.size();i++)if(dot(rotate(quat(ibody),ih.faces[i].n),outward)<dot(rotate(quat(ibody),ih.faces[inf].n),outward))inf=int(i);
 const auto& f=rh.faces[rf];std::vector<Clip> poly;for(int id:ih.faces[inf].ids)poly.push_back({iv[id],uint64_t(1)<<id});
 for(size_t i=0;i<f.ids.size();i++){V p=rv[f.ids[i]],q=rv[f.ids[(i+1)%f.ids.size()]];V side=unit(cross(q-p,outward));poly=clip(poly,side,dot(side,p),uint64_t(i+1)<<32);}
 float plane=dot(outward,rv[f.ids[0]]);
 for(auto p:poly){float separation=dot(outward,p.p)-plane;if(separation<=EPS){uint64_t id=feature(feature(p.id,0x300000000ull+rf),uint64_t(inf)*2+rb+1);if(id==0)id=1;pts.push_back({p.p-outward*(.5f*separation),std::max(0.f,-separation),id});}}
 // Corner grazes can clip the incident polygon away entirely although every
 // SAT interval overlaps; with tiny depth that is "no contact", while a deep
 // penetration without witness points remains a hard numeric failure.
 if(pts.empty()){
  // The most anti-parallel incident face is a heuristic; on many-faceted
  // hulls (the 18-vertex duck foot) it can clip away entirely although SAT
  // proved intersection. Fall back to the deepest incident vertex against
  // the reference plane (vertex-face contact, stable feature id).
  size_t deep=0;float best=1e30f;
  for(size_t i=0;i<iv.size();i++){float sep=dot(outward,iv[i])-plane;if(sep<best){best=sep;deep=i;}}
  if(best<=EPS)pts.push_back({iv[deep]-outward*(.5f*best),std::max(0.f,-best),feature(uint64_t(1)<<deep,0x400000000ull+uint64_t(rf))});
 }
 if(pts.empty()){geo_need(s.depth<=8*EPS,BCV1_NUMERIC);return {};}
 return reduce(s.n,pts);
}
} // namespace dwv1geo
