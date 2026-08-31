// SPDX-License-Identifier: MIT
#include "contact_v1.h"
#include "contact_transaction_v1.h"
#include "contact_math.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
using namespace bcv1_math;
namespace {
constexpr float EPS=2e-6f;
struct Failure { bcv1_status status; };
void need(bool b,bcv1_status s=BCV1_INVALID){if(!b)throw Failure{s};}
bool finite(float x){return std::isfinite(x);}
void valid(V v,float bound=1e4f){need(finite(v.x)&&finite(v.y)&&finite(v.z)&&length(v)<=bound);}
void valid(Q q){need(finite(q.x)&&finite(q.y)&&finite(q.z)&&finite(q.w));float n=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;need(std::fabs(n-1)<2e-5f);}
V unit(V v){float n=length(v);need(finite(n)&&n>1e-12f,BCV1_NUMERIC);return v/n;}
Q normalized(Q q){float n=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);need(finite(n)&&n>1e-12f,BCV1_NUMERIC);return {q.x/n,q.y/n,q.z/n,q.w/n};}
V pos(const bcv1_body& b){return load(b.state);}
Q quat(const bcv1_body& b){return normalized(loadq(b.state+3));}
V world(const bcv1_body& b,V v){return pos(b)+rotate(quat(b),v);}
void body_valid(const bcv1_body& b,bool fixed){
 valid(pos(b));valid(loadq(b.state+3));valid(load(b.state+7));valid(load(b.state+10));
 need(finite(b.inverse_mass)&&b.inverse_mass>=0&&b.inverse_mass<=1e6f);
 for(float v:b.inverse_inertia)need(finite(v)&&v>=0&&v<=1e8f);
 if(fixed){need(b.inverse_mass==0);for(float v:b.inverse_inertia)need(v==0);for(int i=7;i<13;i++)need(b.state[i]==0);}
 else {need(b.inverse_mass>0);for(float v:b.inverse_inertia)need(v>0);}
}
void basis(V n,V& u,V& v){V s=std::fabs(n.x)<=std::fabs(n.y)&&std::fabs(n.x)<=std::fabs(n.z)?V{1,0,0}:std::fabs(n.y)<=std::fabs(n.z)?V{0,1,0}:V{0,0,1};u=unit(cross(s,n));v=cross(n,u);}
struct Face { V n;std::vector<int> ids; };
struct Hull { std::vector<V> v;std::vector<Face> faces;std::vector<std::array<int,2>> edges; };
// Registration-only convex hull construction. Double predicates on copied
// float inputs; plane tolerance relative to diameter, not a metre-scale box.
struct D {double x,y,z;};
D d(V a){return {a.x,a.y,a.z};} D sub(D a,D b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
D cr(D a,D b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double dt(D a,D b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Hull build_hull(const bcv1_shape& s){
 need(s.vertex_count>=4&&s.vertex_count<=32,BCV1_CAPACITY);Hull h;
 for(uint32_t i=0;i<s.vertex_count;i++){V p=load(s.vertices[i]);valid(p,100);h.v.push_back(p);}
 double diameter=0;for(auto a:h.v)for(auto b:h.v)diameter=std::max(diameter,double(length(a-b)));
 need(diameter>=1e-4&&diameter<=100);double tol=diameter*2e-7;
 for(size_t i=0;i<h.v.size();i++)for(size_t j=i+1;j<h.v.size();j++)need(length(h.v[i]-h.v[j])>tol);
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
  // 2D monotone chain removes face-interior/collinear points; no diagonals.
  std::sort(ids.begin(),ids.end(),[&](int a,int b){float ax=dot(h.v[a],u),bx=dot(h.v[b],u);if(ax!=bx)return ax<bx;float ay=dot(h.v[a],v),by=dot(h.v[b],v);return ay!=by?ay<by:a<b;});
  auto turn=[&](int a,int b,int c){D ab=sub(d(h.v[b]),d(h.v[a])),ac=sub(d(h.v[c]),d(h.v[a]));return dt(cr(ab,ac),n);};
  std::vector<int> poly;
  for(int a:ids){while(poly.size()>=2&&turn(poly[poly.size()-2],poly.back(),a)<=tol*diameter)poly.pop_back();poly.push_back(a);}
  size_t lower=poly.size();for(int z=int(ids.size())-2;z>=0;z--){int a=ids[z];while(poly.size()>lower&&turn(poly[poly.size()-2],poly.back(),a)<=tol*diameter)poly.pop_back();poly.push_back(a);}if(poly.size()>1)poly.pop_back();
  if(poly.size()<3)continue;h.faces.push_back({fn,poly});need(h.faces.size()<=60,BCV1_CAPACITY);
 }
 need(h.faces.size()>=4);double volume=0;std::map<std::array<int,2>,int> edge_counts;
 for(auto& f:h.faces){V a=h.v[f.ids[0]];for(size_t i=1;i+1<f.ids.size();i++)volume+=dt(d(a),cr(d(h.v[f.ids[i]]),d(h.v[f.ids[i+1]])))/6;
  for(size_t i=0;i<f.ids.size();i++){int a1=f.ids[i],b=f.ids[(i+1)%f.ids.size()];edge_counts[{std::min(a1,b),std::max(a1,b)}]++;}}
 need(volume>diameter*diameter*diameter*1e-7);for(auto x:edge_counts){need(x.second==2);h.edges.push_back(x.first);}need(h.edges.size()<=90,BCV1_CAPACITY);
 return h;
}
struct Topology {uint32_t E{},B{},P{};std::vector<bcv1_shape> shapes;std::vector<bcv1_pair> pairs;std::vector<Hull> hulls;std::vector<uint32_t> key;};
struct Payload {std::vector<bcv1_body> bodies;std::vector<bcv1_manifold> cache;std::vector<float> gravity,mu;std::vector<double> clocks;};
void addfloat(std::vector<uint32_t>& key,float f){if(f==0)f=0;uint32_t x;std::memcpy(&x,&f,4);key.push_back(x);}
std::shared_ptr<const Topology> topology(const bcv1_registration& r){
 need(r.version==1);need(r.environments>0&&r.environments<=4096&&r.bodies>0&&r.bodies<=32&&r.pairs<=16,BCV1_CAPACITY);
 need(r.shapes&&(r.pairs==0||r.contact_pairs));auto t=std::make_shared<Topology>();t->E=r.environments;t->B=r.bodies;t->P=r.pairs;
 t->key={1,t->E,t->B,t->P};std::set<uint32_t> ids;
 for(uint32_t b=0;b<t->B;b++){auto s=r.shapes[b];need(ids.insert(s.caller_id).second);need(s.kind<=BCV1_PLANE&&s.fixed<=1);t->key.insert(t->key.end(),{s.caller_id,s.kind,s.fixed,s.vertex_count});
  if(s.kind==BCV1_CONVEX){t->hulls.push_back(build_hull(s));for(uint32_t v=0;v<s.vertex_count;v++)for(float x:s.vertices[v])addfloat(t->key,x);}
  else {need(s.vertex_count==0);t->hulls.push_back({});if(s.kind==BCV1_PLANE){need(s.fixed==1);V n=load(s.plane_normal);valid(n);need(std::fabs(length(n)-1)<2e-5f&&finite(s.plane_offset)&&std::fabs(s.plane_offset)<=1e4);for(float x:s.plane_normal)addfloat(t->key,x);addfloat(t->key,s.plane_offset);}}
  t->shapes.push_back(s);
 }
 ids.clear();std::set<std::array<uint32_t,2>> pairs;
 for(uint32_t p=0;p<t->P;p++){auto x=r.contact_pairs[p];need(x.body_a<t->B&&x.body_b<t->B&&x.body_a!=x.body_b);need(ids.insert(x.caller_id).second&&pairs.insert({std::min(x.body_a,x.body_b),std::max(x.body_a,x.body_b)}).second);
  need(t->shapes[x.body_a].kind&&t->shapes[x.body_b].kind);need(t->shapes[x.body_a].kind==1||t->shapes[x.body_b].kind==1);need(!(t->shapes[x.body_a].fixed&&t->shapes[x.body_b].fixed));
  t->pairs.push_back(x);t->key.insert(t->key.end(),{x.caller_id,x.body_a,x.body_b});}
 return t;
}
void payload_valid(const Topology& t,const Payload& p){
 need(p.bodies.size()==size_t(t.E)*t.B&&p.cache.size()==size_t(t.E)*t.P&&p.mu.size()==size_t(t.E)*t.P&&p.gravity.size()==size_t(t.E)*3&&p.clocks.size()==t.E);
 for(uint32_t e=0;e<t.E;e++){valid(load(p.gravity.data()+e*3),100);need(std::isfinite(p.clocks[e])&&p.clocks[e]>=0);for(uint32_t b=0;b<t.B;b++)body_valid(p.bodies[e*t.B+b],t.shapes[b].fixed);for(uint32_t i=0;i<t.P;i++)need(finite(p.mu[e*t.P+i])&&p.mu[e*t.P+i]>=0&&p.mu[e*t.P+i]<=4);}
 for(auto& m:p.cache){need(m.count<=4);if(!m.count)continue;valid(load(m.normal));valid(load(m.tangent1));valid(load(m.tangent2));need(std::fabs(length(load(m.normal))-1)<2e-5f);for(uint32_t j=0;j<m.count;j++){auto& x=m.points[j];valid(load(x.point),2e4);need(x.feature&&finite(x.depth)&&x.depth>=0&&finite(x.normal_impulse)&&x.normal_impulse>=0&&finite(x.tangent_impulse[0])&&finite(x.tangent_impulse[1]));}}
}
std::vector<V> vertices(const Hull& h,const bcv1_body& b){std::vector<V> r;for(auto p:h.v){V x=world(b,p);valid(x,2e4);r.push_back(x);}return r;}
uint64_t feature(uint64_t a,uint64_t b){return (a^b)*1099511628211ull;}
struct Candidate {V p;float depth;uint64_t id;};
bcv1_manifold reduce(V n,std::vector<Candidate> pts){
 bcv1_manifold m{};if(pts.empty())return m;V u,v;basis(n,u,v);save(n,m.normal);save(u,m.tangent1);save(v,m.tangent2);
 std::sort(pts.begin(),pts.end(),[](auto a,auto b){return a.id<b.id;});std::vector<Candidate> unique;
 for(auto p:pts){bool duplicate=false;for(auto q:unique)if(length(p.p-q.p)<EPS*.5f){duplicate=true;break;}if(!duplicate)unique.push_back(p);}
 need(unique.size()<=64,BCV1_CAPACITY);std::vector<size_t> chosen;
 auto choose=[&](size_t x){if(std::find(chosen.begin(),chosen.end(),x)==chosen.end())chosen.push_back(x);};
 if(unique.size()<=4){for(size_t i=0;i<unique.size();i++)choose(i);}else{
  size_t deep=0;for(size_t i=1;i<unique.size();i++)if(unique[i].depth>unique[deep].depth)deep=i;choose(deep);
  while(chosen.size()<4){float best=-1;size_t bi=0;for(size_t i=0;i<unique.size();i++){if(std::find(chosen.begin(),chosen.end(),i)!=chosen.end())continue;float score=1e30f;for(auto j:chosen){V diff=unique[i].p-unique[j].p;score=std::min(score,dot(diff,diff));}if(score>best){best=score;bi=i;}}choose(bi);}
 }
 std::sort(chosen.begin(),chosen.end(),[&](size_t a,size_t b){return unique[a].id<unique[b].id;});m.count=uint32_t(chosen.size());
 for(size_t i=0;i<chosen.size();i++){auto p=unique[chosen[i]];m.points[i].feature=p.id;save(p.p,m.points[i].point);m.points[i].depth=p.depth;}return m;
}
bcv1_manifold plane_contact(const Topology& t,uint32_t ca,const bcv1_body& c,uint32_t pl,const bcv1_body& p,bool plane_is_a){
 const auto& s=t.shapes[pl];float scale=length(load(s.plane_normal));V n=unit(rotate(quat(p),load(s.plane_normal)/scale));float offset=s.plane_offset/scale+dot(n,pos(p));auto vs=vertices(t.hulls[ca],c);std::vector<Candidate> pts;
 for(size_t i=0;i<vs.size();i++){float sep=dot(n,vs[i])-offset;if(sep<=EPS){float depth=std::max(0.f,-sep);pts.push_back({vs[i]-n*(sep*.5f),depth,0x100000000ull+i+1});}}
 return reduce(plane_is_a?n:-n,pts);
}
struct SAT {bool hit{};V n{};float depth{1e30f};int kind{},a{},b{};};
SAT sat(const Hull& ah,const bcv1_body& a,const std::vector<V>& av,const Hull& bh,const bcv1_body& b,const std::vector<V>& bv){
 SAT best;best.hit=true;
 auto axis=[&](V n,int kind,int ia,int ib){float len=length(n);if(len<1e-6f)return true;n=n/len;
  float amin=1e30f,amax=-1e30f,bmin=1e30f,bmax=-1e30f;for(auto v:av){float x=dot(n,v);amin=std::min(amin,x);amax=std::max(amax,x);}for(auto v:bv){float x=dot(n,v);bmin=std::min(bmin,x);bmax=std::max(bmax,x);}
  float positive=amax-bmin,negative=bmax-amin;if(positive < -EPS||negative < -EPS)return false;
  // Face candidates retain their actual outward reference face. Reversing a
  // face normal and then selecting a merely nearby opposite face is wrong for
  // tetrahedra/asymmetric hulls. B faces enter already negated; edge axes may
  // use either signed exit. All axes still test separation in BOTH directions.
  float depth=positive;if(kind==2&&negative<positive){depth=negative;n=-n;}
  if(depth<best.depth-EPS){best.n=n;best.depth=std::max(0.f,depth);best.kind=kind;best.a=ia;best.b=ib;}return true;};
 for(size_t i=0;i<ah.faces.size();i++)if(!axis(rotate(quat(a),ah.faces[i].n),0,int(i),0))return {};
 for(size_t i=0;i<bh.faces.size();i++)if(!axis(-rotate(quat(b),bh.faces[i].n),1,int(i),0))return {};
 for(size_t i=0;i<ah.edges.size();i++)for(size_t j=0;j<bh.edges.size();j++){auto x=ah.edges[i],y=bh.edges[j];if(!axis(cross(unit(av[x[1]]-av[x[0]]),unit(bv[y[1]]-bv[y[0]])),2,int(i),int(j)))return {};}
 return best;
}
struct Clip {V p;uint64_t id;};
std::vector<Clip> clip(const std::vector<Clip>& in,V n,float offset,uint64_t tag){
 std::vector<Clip> out;if(in.empty())return out;Clip prev=in.back();float dp=dot(n,prev.p)-offset;
 for(auto cur:in){float dc=dot(n,cur.p)-offset;bool ip=dp<=EPS,ic=dc<=EPS;if(ip!=ic){float f=dp/(dp-dc);f=std::max(0.f,std::min(1.f,f));out.push_back({prev.p+(cur.p-prev.p)*f,feature(feature(std::min(prev.id,cur.id),std::max(prev.id,cur.id)),tag)});}if(ic)out.push_back(cur);prev=cur;dp=dc;need(out.size()<=64,BCV1_CAPACITY);}return out;
}
std::array<V,2> closest(V p,V q,V r,V s){
 // Edge parameters are ill-conditioned near parallelism: float (a*e-b*b)
 // lost enough bits to move a rod contact by12.5mm at angle.002rad. Use
 // double predicates/parameters on exact copied float endpoints, retaining
 // f32 state/manifold output. This is a precision fix, not a looser gate.
 D d1=sub(d(q),d(p)),d2=sub(d(s),d(r)),dr=sub(d(p),d(r));double a=dt(d1,d1),e=dt(d2,d2),b=dt(d1,d2),c=dt(d1,dr),f=dt(d2,dr),den=a*e-b*b;
 double u=den>a*e*1e-15?std::max(0.,std::min(1.,(b*f-c*e)/den)):0;
 double v=(b*u+f)/e;if(v<0){v=0;u=std::max(0.,std::min(1.,-c/a));}else if(v>1){v=1;u=std::max(0.,std::min(1.,(b-c)/a));}
 return {V{float(p.x+d1.x*u),float(p.y+d1.y*u),float(p.z+d1.z*u)},V{float(r.x+d2.x*v),float(r.y+d2.y*v),float(r.z+d2.z*v)}};
}
bcv1_manifold convex_contact(const Hull& ah,const bcv1_body& a,const Hull& bh,const bcv1_body& b){
 auto av=vertices(ah,a),bv=vertices(bh,b);SAT s=sat(ah,a,av,bh,b,bv);if(!s.hit)return {};std::vector<Candidate> pts;
 if(s.kind==2){
  float am=-1e30f,bm=1e30f;for(auto v:av)am=std::max(am,dot(s.n,v));for(auto v:bv)bm=std::min(bm,dot(s.n,v));
  float distance=1e30f;V cp{};uint64_t id=0; // actual support edge segments, never arbitrary support midpoint
  for(size_t i=0;i<ah.edges.size();i++){auto ae=ah.edges[i];if(std::fabs(dot(s.n,av[ae[0]])-am)>4*EPS||std::fabs(dot(s.n,av[ae[1]])-am)>4*EPS)continue;
   for(size_t j=0;j<bh.edges.size();j++){auto be=bh.edges[j];if(std::fabs(dot(s.n,bv[be[0]])-bm)>4*EPS||std::fabs(dot(s.n,bv[be[1]])-bm)>4*EPS)continue;
    auto x=closest(av[ae[0]],av[ae[1]],bv[be[0]],bv[be[1]]);float d2=dot(x[0]-x[1],x[0]-x[1]);if(d2<distance){distance=d2;cp=(x[0]+x[1])*.5f;id=0x200000000ull+(i<<16)+j+1;}}
  }
  need(id!=0,BCV1_NUMERIC);pts.push_back({cp,s.depth,id});return reduce(s.n,pts);
 }
 bool rb=s.kind==1;auto& rh=rb?bh:ah;auto& ih=rb?ah:bh;auto& rv=rb?bv:av;auto& iv=rb?av:bv;auto& rbody=rb?b:a;auto& ibody=rb?a:b;V outward=rb?-s.n:s.n;
 int rf=s.a,inf=0;need(dot(rotate(quat(rbody),rh.faces[rf].n),outward)>1-1e-5f,BCV1_NUMERIC);
 for(size_t i=1;i<ih.faces.size();i++)if(dot(rotate(quat(ibody),ih.faces[i].n),outward)<dot(rotate(quat(ibody),ih.faces[inf].n),outward))inf=int(i);
 const auto& f=rh.faces[rf];std::vector<Clip> poly;for(int id:ih.faces[inf].ids)poly.push_back({iv[id],uint64_t(1)<<id});
 for(size_t i=0;i<f.ids.size();i++){V p=rv[f.ids[i]],q=rv[f.ids[(i+1)%f.ids.size()]];V side=unit(cross(q-p,outward));poly=clip(poly,side,dot(side,p),uint64_t(i+1)<<32);}
 float plane=dot(outward,rv[f.ids[0]]);
 for(auto p:poly){float separation=dot(outward,p.p)-plane;if(separation<=EPS){uint64_t id=feature(feature(p.id,0x300000000ull+rf),uint64_t(inf)*2+rb+1);if(id==0)id=1;pts.push_back({p.p-outward*(.5f*separation),std::max(0.f,-separation),id});}}
 need(!pts.empty(),BCV1_NUMERIC);return reduce(s.n,pts);
}
bcv1_manifold manifold(const Topology& t,const Payload& p,uint32_t e,uint32_t i){
 auto pair=t.pairs[i];auto& a=p.bodies[e*t.B+pair.body_a];auto& b=p.bodies[e*t.B+pair.body_b];
 if(t.shapes[pair.body_a].kind==2)return plane_contact(t,pair.body_b,b,pair.body_a,a,true);
 if(t.shapes[pair.body_b].kind==2)return plane_contact(t,pair.body_a,a,pair.body_b,b,false);
 return convex_contact(t.hulls[pair.body_a],a,t.hulls[pair.body_b],b);
}
V velocity(const bcv1_body& b,V p){return point_velocity(load(b.state+7),load(b.state+10),p-pos(b));}
float inverse(const bcv1_body& a,const bcv1_body& b,V p,V u,V v){return row_inverse(a.inverse_mass,quat(a),load(a.inverse_inertia),p-pos(a),u,v)+row_inverse(b.inverse_mass,quat(b),load(b.inverse_inertia),p-pos(b),u,v);}
void impulse(bcv1_body& b,V p,V j){if(b.inverse_mass==0)return;save(load(b.state+7)+j*b.inverse_mass,b.state+7);save(load(b.state+10)+inertia(quat(b),load(b.inverse_inertia),cross(p-pos(b),j)),b.state+10);}
void pair_impulse(bcv1_body& a,bcv1_body& b,V p,V j){impulse(a,p,-j);impulse(b,p,j);}
void solve(bcv1_body& a,bcv1_body& b,bcv1_manifold& m,float mu,float h){
 V n=load(m.normal),t=load(m.tangent1),u=load(m.tangent2);
 for(uint32_t j=0;j<m.count;j++){auto& c=m.points[j];V p=load(c.point);float k=inverse(a,b,p,n,n);need(finite(k)&&k>1e-12f,BCV1_NUMERIC);
  float vn=dot(velocity(b,p)-velocity(a,p),n),bias=std::min(1.f,.2f*std::max(0.f,c.depth-EPS)/h),old=c.normal_impulse;
  need(finite(vn)&&finite(bias)&&finite(old)&&old>=0,BCV1_NUMERIC);float proposed=old+(bias-vn)/k;need(finite(proposed),BCV1_NUMERIC);
  c.normal_impulse=std::max(0.f,proposed);pair_impulse(a,b,p,n*(c.normal_impulse-old));
  V rel=velocity(b,p)-velocity(a,p);float k11=inverse(a,b,p,t,t),k22=inverse(a,b,p,u,u),k12=inverse(a,b,p,t,u),det=k11*k22-k12*k12;
  need(finite(k11)&&finite(k22)&&finite(k12)&&k11>0&&k22>0&&finite(det)&&det>1e-20f,BCV1_NUMERIC);
  // Conditional 2D maximum-dissipation block on the Euclidean Coulomb disk.
  // Radially clipping K^-1*rhs is wrong for anisotropic effective inertia.
  float r1=dot(rel,t)-k11*c.tangent_impulse[0]-k12*c.tangent_impulse[1],r2=dot(rel,u)-k12*c.tangent_impulse[0]-k22*c.tangent_impulse[1];
  need(finite(r1)&&finite(r2),BCV1_NUMERIC);
  auto candidate=[&](float lambda){float a11=k11+lambda,a22=k22+lambda,dd=a11*a22-k12*k12;need(finite(dd)&&dd>0,BCV1_NUMERIC);V value{(-a22*r1+k12*r2)/dd,(k12*r1-a11*r2)/dd,0};need(finite(value.x)&&finite(value.y)&&finite(length(value)),BCV1_NUMERIC);return value;};
  float cap=mu*c.normal_impulse;need(finite(cap),BCV1_NUMERIC);V jt=candidate(0);
  if(cap==0)jt={};else if(length(jt)>cap){float low=0,high=std::max(k11,k22);int growth=0;while(length(candidate(high))>cap){high*=2;need(finite(high)&&++growth<=80,BCV1_NUMERIC);}for(int z=0;z<40;z++){float mid=low+(high-low)*.5f;if(length(candidate(mid))>cap)low=mid;else high=mid;}jt=candidate(high);}
  pair_impulse(a,b,p,t*(jt.x-c.tangent_impulse[0])+u*(jt.y-c.tangent_impulse[1]));c.tangent_impulse[0]=jt.x;c.tangent_impulse[1]=jt.y;
 }
}
void seed(bcv1_body& a,bcv1_body& b,bcv1_manifold& m,const bcv1_manifold& old,float mu){
 if(!m.count||!old.count||dot(load(m.normal),load(old.normal))<.9999f)return;
 for(uint32_t i=0;i<m.count;i++)for(uint32_t j=0;j<old.count;j++)if(m.points[i].feature==old.points[j].feature){auto& c=m.points[i];auto& p=old.points[j];if(length(load(c.point)-load(p.point))>.01f)continue;
  c.normal_impulse=p.normal_impulse;V tangent=load(old.tangent1)*p.tangent_impulse[0]+load(old.tangent2)*p.tangent_impulse[1];float x=dot(tangent,load(m.tangent1)),y=dot(tangent,load(m.tangent2)),r=std::sqrt(x*x+y*y),cap=mu*c.normal_impulse;if(r>cap){x*=cap/r;y*=cap/r;}c.tangent_impulse[0]=x;c.tangent_impulse[1]=y;
  pair_impulse(a,b,load(c.point),load(m.normal)*c.normal_impulse+load(m.tangent1)*x+load(m.tangent2)*y);break;}
}
void advance(const Topology& t,Payload& p,float h,uint32_t iterations){
 for(uint32_t e=0;e<t.E;e++){
  for(uint32_t b=0;b<t.B;b++){auto& v=p.bodies[e*t.B+b];if(v.inverse_mass)save(load(v.state+7)+load(p.gravity.data()+e*3)*h,v.state+7);}
  std::vector<bcv1_manifold> contacts;for(uint32_t i=0;i<t.P;i++)contacts.push_back(manifold(t,p,e,i));
  for(uint32_t i=0;i<t.P;i++){auto pair=t.pairs[i];seed(p.bodies[e*t.B+pair.body_a],p.bodies[e*t.B+pair.body_b],contacts[i],p.cache[e*t.P+i],p.mu[e*t.P+i]);}
  for(uint32_t it=0;it<iterations;it++)for(uint32_t i=0;i<t.P;i++){auto pair=t.pairs[i];solve(p.bodies[e*t.B+pair.body_a],p.bodies[e*t.B+pair.body_b],contacts[i],p.mu[e*t.P+i],h);}
  for(uint32_t i=0;i<t.P;i++)p.cache[e*t.P+i]=contacts[i];
  for(uint32_t b=0;b<t.B;b++){auto& v=p.bodies[e*t.B+b];if(!v.inverse_mass)continue;save(pos(v)+load(v.state+7)*h,v.state);V w=load(v.state+10),ii=load(v.inverse_inertia),mom=inertia(quat(v),{1/ii.x,1/ii.y,1/ii.z},w);Q dq=mul({w.x,w.y,w.z,0},quat(v)),q=quat(v);Q next=normalized({q.x+.5f*h*dq.x,q.y+.5f*h*dq.y,q.z+.5f*h*dq.z,q.w+.5f*h*dq.w});saveq(next,v.state+3);save(inertia(next,ii,mom),v.state+10);}
  p.clocks[e]+=h;
 }
 payload_valid(t,p);
}
template<class F> bcv1_status guarded(F f){try{f();return BCV1_OK;}catch(Failure e){return e.status;}catch(const std::bad_alloc&){return BCV1_CAPACITY;}catch(...){return BCV1_INTERNAL;}}
} // namespace
struct bcv1_scene {std::shared_ptr<const Topology> topology;Payload payload;uint64_t generation=0;};
struct bcv1_snapshot {std::shared_ptr<const Topology> topology;Payload payload;};
struct bcx1_stage {const bcv1_scene* owner;uint64_t generation;bool consumed=false;Payload payload;};
extern "C" {
bcv1_status bcv1_create(const bcv1_registration* r,bcv1_scene** out){if(!out)return BCV1_INVALID;*out=nullptr;return guarded([&]{need(r&&r->initial&&r->gravity_xyz&&(r->pairs==0||r->pair_friction));auto s=std::make_unique<bcv1_scene>();s->topology=topology(*r);auto& t=*s->topology;auto& p=s->payload;p.bodies.assign(r->initial,r->initial+size_t(t.E)*t.B);p.gravity.assign(r->gravity_xyz,r->gravity_xyz+size_t(t.E)*3);if(t.P)p.mu.assign(r->pair_friction,r->pair_friction+size_t(t.E)*t.P);p.cache.resize(size_t(t.E)*t.P);p.clocks.resize(t.E);payload_valid(t,p);*out=s.release();});}
void bcv1_destroy(bcv1_scene* s){delete s;}
bcv1_status bcv1_read(const bcv1_scene* s,bcv1_body* b,bcv1_manifold* m,double* c){return guarded([&]{need(s);auto& p=s->payload;if(b)std::copy(p.bodies.begin(),p.bodies.end(),b);if(m)std::copy(p.cache.begin(),p.cache.end(),m);if(c)std::copy(p.clocks.begin(),p.clocks.end(),c);});}
bcv1_status bcv1_query(const bcv1_scene* s,bcv1_manifold* out){return guarded([&]{need(s&&(out||s->topology->P==0));auto& t=*s->topology;std::vector<bcv1_manifold> r;for(uint32_t e=0;e<t.E;e++)for(uint32_t i=0;i<t.P;i++)r.push_back(manifold(t,s->payload,e,i));std::copy(r.begin(),r.end(),out);});}
bcv1_status bcv1_step(bcv1_scene* s,float dt,uint32_t iterations){return guarded([&]{need(s&&s->generation<UINT64_MAX&&finite(dt)&&dt>0&&dt<=.01f&&iterations>0&&iterations<=128);Payload next=s->payload;advance(*s->topology,next,dt,iterations);s->payload=std::move(next);++s->generation;});}
bcv1_status bcv1_capture(const bcv1_scene* s,bcv1_snapshot** out){if(!out)return BCV1_INVALID;*out=nullptr;return guarded([&]{need(s);auto x=std::make_unique<bcv1_snapshot>();x->topology=s->topology;x->payload=s->payload;*out=x.release();});}
void bcv1_snapshot_destroy(bcv1_snapshot* s){delete s;}
bcv1_status bcv1_restore(bcv1_scene* s,const bcv1_snapshot* snap,const uint8_t* mask){bcx1_stage* stage=nullptr;auto rc=bcx1_prepare_restore(s,snap,mask,&stage);if(rc)return rc;int commit=bcx1_commit(s,stage);bcx1_stage_destroy(stage);return static_cast<bcv1_status>(commit);}
bcv1_status bcv1_to_principal(const float* source,const float* com,const float* qpc,float* out){return guarded([&]{need(source&&com&&qpc&&out);V p=load(source),v=load(source+7),w=load(source+10),c=load(com);Q q=loadq(source+3),pc=loadq(qpc);valid(p);valid(v);valid(w);valid(c,100);valid(q);valid(pc);q=normalized(q);pc=normalized(pc);float r[13];V offset=rotate(q,c);save(p+offset,r);saveq(normalized(mul(q,pc)),r+3);save(v+cross(w,offset),r+7);save(w,r+10);valid(load(r));valid(load(r+7));std::copy_n(r,13,out);});}
bcv1_status bcv1_from_principal(const float* input,const float* com,const float* qpc,float* out){return guarded([&]{need(input&&com&&qpc&&out);V p=load(input),v=load(input+7),w=load(input+10),c=load(com);Q q=loadq(input+3),pc=loadq(qpc);valid(p);valid(v);valid(w);valid(c,100);valid(q);valid(pc);q=normalized(q);pc=normalized(pc);float r[13];Q qs=normalized(mul(q,conj(pc)));V offset=rotate(qs,c);save(p-offset,r);saveq(qs,r+3);save(v-cross(w,offset),r+7);save(w,r+10);valid(load(r));valid(load(r+7));std::copy_n(r,13,out);});}
bcv1_status bcv1_bake_convex(const float* pose,const float* vs,uint32_t count,float* out){return guarded([&]{need(pose&&vs&&out);need(count>=4&&count<=32,BCV1_CAPACITY);V p=load(pose);Q q=loadq(pose+3);valid(p,100);valid(q);q=normalized(q);std::array<float,96> next{};for(uint32_t i=0;i<count;i++){V v=load(vs+i*3);valid(v,100);V x=p+rotate(q,v);valid(x,100);save(x,next.data()+i*3);}std::copy_n(next.data(),count*3,out);});}
bcv1_status bcv1_support(const bcv1_shape* shape,const float* state,const float* direction,float* out,uint32_t* id){return guarded([&]{need(shape&&state&&direction&&out&&id);need(shape->kind==1&&shape->vertex_count>=4&&shape->vertex_count<=32);V p=load(state),n=load(direction);Q q=loadq(state+3);valid(p);valid(q);valid(n);need(length(n)>1e-12f);q=normalized(q);float best=-std::numeric_limits<float>::infinity(),r[4]{};uint32_t which=0;for(uint32_t i=0;i<shape->vertex_count;i++){V v=load(shape->vertices[i]);valid(v,100);V x=p+rotate(q,v);valid(x,2e4);float val=dot(x,n);need(finite(val),BCV1_NUMERIC);if(val>best){best=val;which=i;save(x,r);r[3]=val;}}std::copy_n(r,4,out);*id=which;});}
bcv1_status bcv1_joint_geometry(const float* pa,const float* ch,const float* ap,const float* ac,const float* axis,const float* ref,float* out){return guarded([&]{need(pa&&ch&&ap&&ac&&axis&&ref&&out);for(auto s:{pa,ch}){valid(load(s));valid(loadq(s+3));valid(load(s+7));valid(load(s+10));}V va=load(ap),vb=load(ac),ax=load(axis);valid(va,100);valid(vb,100);valid(ax);need(std::fabs(length(ax)-1)<2e-5f);Q r=loadq(ref);valid(r);Q pq=normalized(loadq(pa+3)),cq=normalized(loadq(ch+3));Q delta=normalized(mul(mul(conj(pq),cq),conj(normalized(r))));if(delta.w<0)delta={-delta.x,-delta.y,-delta.z,-delta.w};V dv{delta.x,delta.y,delta.z};float len=length(dv);V rv=len<1e-9f?dv*2:dv*(2*std::atan2(len,delta.w)/len);float result[5]={dot(rv,unit(ax)),dot(rotate(pq,unit(ax)),load(ch+10)-load(pa+10)),0,0,0};save(load(ch)+rotate(cq,vb)-load(pa)-rotate(pq,va),result+2);std::copy_n(result,5,out);});}
bcv1_status bcv1_contact_row(uint32_t B,uint32_t N,const bcv1_body* bs,uint32_t a,uint32_t b,const float* point,const float* direction,const float* J,float* out){return guarded([&]{need(B>0&&B<=32&&N>0&&N<=64,BCV1_CAPACITY);need(bs&&point&&direction&&J&&out&&a<B&&b<B&&a!=b);V p=load(point),d=load(direction);valid(p,2e4);valid(d);need(std::fabs(length(d)-1)<2e-5f);valid(pos(bs[a]));valid(pos(bs[b]));std::vector<float> g(N,0);for(uint32_t i=0;i<B*6*N;i++)need(finite(J[i])&&std::fabs(J[i])<=1e6f);for(uint32_t body:{a,b}){V torque=cross(p-pos(bs[body]),d);float sign=body==a?-1.f:1.f;for(uint32_t j=0;j<N;j++)for(uint32_t k=0;k<3;k++)g[j]+=sign*(d[k]*J[(body*6+k)*N+j]+torque[k]*J[(body*6+3+k)*N+j]);}for(float x:g)need(finite(x),BCV1_NUMERIC);std::copy(g.begin(),g.end(),out);});}
bcv1_status bcx1_prepare_solved(const bcv1_scene* s,const bcv1_body* bodies,const bcv1_manifold* cache,double dt,bcx1_stage** out){return guarded([&]{need(s&&out&&bodies&&(!s->topology->P||cache)&&s->generation<UINT64_MAX&&std::isfinite(dt)&&dt>0&&dt<=.01);auto stage=std::make_unique<bcx1_stage>();stage->owner=s;stage->generation=s->generation;stage->payload=s->payload;auto& t=*s->topology;
 for(uint32_t e=0;e<t.E;e++){for(uint32_t b=0;b<t.B;b++){size_t index=e*t.B+b;const auto& before=s->payload.bodies[index];const auto& after=bodies[index];need(after.inverse_mass==before.inverse_mass&&std::equal(after.inverse_inertia,after.inverse_inertia+3,before.inverse_inertia));if(t.shapes[b].fixed)need(std::memcmp(after.state,before.state,sizeof(before.state))==0);stage->payload.bodies[index]=after;}
  for(uint32_t i=0;i<t.P;i++){const auto& m=cache[e*t.P+i];auto geometry=manifold(t,s->payload,e,i);need(m.count==geometry.count);need(std::memcmp(m.normal,geometry.normal,3*sizeof(float))==0&&std::memcmp(m.tangent1,geometry.tangent1,3*sizeof(float))==0&&std::memcmp(m.tangent2,geometry.tangent2,3*sizeof(float))==0);
   for(uint32_t p=0;p<m.count;p++){const auto& x=m.points[p];const auto& y=geometry.points[p];need(x.feature==y.feature&&x.depth==y.depth&&std::memcmp(x.point,y.point,3*sizeof(float))==0);float cap=s->payload.mu[e*t.P+i]*x.normal_impulse;need(finite(cap)&&std::hypot(x.tangent_impulse[0],x.tangent_impulse[1])<=cap+2e-6f*(1+cap));}
   stage->payload.cache[e*t.P+i]=m;}
  stage->payload.clocks[e]+=dt;
 }payload_valid(t,stage->payload);*out=stage.release();});}
bcv1_status bcx1_prepare_restore(const bcv1_scene* s,const bcv1_snapshot* snap,const uint8_t* mask,bcx1_stage** out){return guarded([&]{need(s&&snap&&out&&s->generation<UINT64_MAX);need(s->topology->key==snap->topology->key,BCV1_TOPOLOGY);auto& t=*s->topology;payload_valid(t,snap->payload);auto stage=std::make_unique<bcx1_stage>();stage->owner=s;stage->generation=s->generation;stage->payload=s->payload;auto& next=stage->payload;
 for(uint32_t e=0;e<t.E;e++)if(!mask||mask[e]){std::copy_n(snap->payload.bodies.data()+e*t.B,t.B,next.bodies.data()+e*t.B);if(t.P){std::copy_n(snap->payload.cache.data()+e*t.P,t.P,next.cache.data()+e*t.P);std::copy_n(snap->payload.mu.data()+e*t.P,t.P,next.mu.data()+e*t.P);}std::copy_n(snap->payload.gravity.data()+e*3,3,next.gravity.data()+e*3);next.clocks[e]=snap->payload.clocks[e];}*out=stage.release();});}
bcv1_status bcx1_stage_read(const bcx1_stage* s,bcv1_body* b,bcv1_manifold* m,double* c){return guarded([&]{need(s&&!s->consumed);auto& p=s->payload;if(b)std::copy(p.bodies.begin(),p.bodies.end(),b);if(m)std::copy(p.cache.begin(),p.cache.end(),m);if(c)std::copy(p.clocks.begin(),p.clocks.end(),c);});}
bcv1_status bcx1_stage_query(const bcx1_stage* s,bcv1_manifold* out){return guarded([&]{need(s&&!s->consumed&&(out||s->owner->topology->P==0));auto& t=*s->owner->topology;std::vector<bcv1_manifold> result;for(uint32_t e=0;e<t.E;e++)for(uint32_t p=0;p<t.P;p++)result.push_back(manifold(t,s->payload,e,p));std::copy(result.begin(),result.end(),out);});}
int bcx1_validate_commit(const bcv1_scene* s,const bcx1_stage* p){if(!s||!p)return BCV1_INVALID;return p->owner!=s||p->consumed||p->generation!=s->generation||s->generation==UINT64_MAX?BCX1_STALE:BCV1_OK;}
int bcx1_commit(bcv1_scene* s,bcx1_stage* p){int rc=bcx1_validate_commit(s,p);if(rc)return rc;std::swap(s->payload,p->payload);s->generation++;p->consumed=true;return BCV1_OK;}
void bcx1_stage_destroy(bcx1_stage* p){delete p;}
}
