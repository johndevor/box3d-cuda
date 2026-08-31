// SPDX-License-Identifier: MIT
// duck_world_v1: one articulated Open Duck (av2, glued exactly like idv1)
// plus a seeded grid of cube rigid bodies plus the static floor, batched over
// E environments. Static cubes are terrain; dynamic cubes are 6-dof free
// bodies (semi-implicit Euler, f64 state) with SAT manifolds, uniform-grid
// broadphase, contact-graph islands, sleeping, and per-island civ1 solves
// whose mass matrix is block-diagonal: duck NxN (from av2 PRE) + 6x6 per
// awake cube. Cube inertia is isotropic (m*s^2/6), so the world inertia is a
// constant diagonal and the gyroscopic term is exactly zero.
#include "duck_world_v1.h"
#include "contact_transaction_v1.h"
#include "dwv1_geometry.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
namespace {
struct Error { int status; };
void need(bool x,int code=DWV1_INVALID){if(!x)throw Error{code};}
template<class T> T desc(){T x{};x.struct_size=sizeof(T);x.version=AV2_ABI;return x;}
template<class F> int guard(F f){try{f();return 0;}catch(Error e){return e.status;}catch(const dwv1geo::Fail&){return DWV1_CONTACT;}catch(const std::bad_alloc&){return DWV1_ALLOCATION;}catch(...){return DWV1_INVALID;}}
using AP=std::unique_ptr<av2_pre,decltype(&av2_pre_destroy)>;
using AS=std::unique_ptr<av2_stage,decltype(&av2_stage_destroy)>;
using CS=std::unique_ptr<bcx1_stage,decltype(&bcx1_stage_destroy)>;
constexpr double SLEEP_LINEAR=.01,SLEEP_ANGULAR=.1,BROAD_MARGIN=.005;
constexpr uint32_t SLEEP_TICKS=50;
uint64_t mix(uint64_t x){x+=0x9E3779B97F4A7C15ull;x^=x>>30;x*=0xBF58476D1CE4E5B9ull;x^=x>>27;x*=0x94D049BB133111EBull;x^=x>>31;return x;}
double unit01(uint64_t x){return double(x>>11)*(1./9007199254740992.);}
double dot3(const float* a,const float* b){return double(a[0])*b[0]+double(a[1])*b[1]+double(a[2])*b[2];}
struct Saved {
 av2_snapshot d{};std::vector<double> q,v,w,t;std::vector<uint64_t> count;
 Saved(uint32_t E,uint32_t J):q(size_t(E)*(7+J)),v(size_t(E)*(6+J)),w(size_t(E)*3*J),t(E),count(E){d=desc<av2_snapshot>();d.environments=E;d.joints=J;bind();}
 void bind(){d.qpos=q.data();d.velocity=v.data();d.joint_warm_force=w.data();d.time=t.data();d.step_count=count.data();}
 Saved(const Saved& x):d(x.d),q(x.q),v(x.v),w(x.w),t(x.t),count(x.count){bind();}
};
struct Cubes {
 std::vector<double> pose,velocity; // [E,M,7] xyz+xyzw, [E,M,6] world lin+ang
 std::vector<uint8_t> awake;        // [E,M]
 std::vector<uint32_t> still;       // [E,M] consecutive low-motion ticks
 std::vector<std::map<uint64_t,bcv1_manifold>> cache; // [E], warm-start caches
 std::vector<uint8_t> foot;         // [E,F] foot contact flags, last accepted step
};
struct Rows {
 size_t N;std::vector<double> g,target,r,lo,hi,warm;std::vector<civ1_contact> contacts;
 explicit Rows(size_t n):N(n){}
 void add(const double* row,double tar,double reg,double lower,double upper,double w){g.insert(g.end(),row,row+N);target.push_back(tar);r.push_back(reg);lo.push_back(lower);hi.push_back(upper);warm.push_back(w);}
};
// key identifies a contact pair across steps for warm starting and queries
uint64_t pairkey(uint32_t ka,uint32_t ia,uint32_t kb,uint32_t ib){return (uint64_t(ka)<<56)|(uint64_t(ia&0xFFFFu)<<40)|(uint64_t(kb)<<32)|ib;}
struct CM { uint64_t key;uint32_t kind_a,index_a,kind_b,index_b;bcv1_manifold m; };
}
struct dwv1_scene {
 std::mutex lock;av2_scene* art=nullptr;bcv1_scene* contact=nullptr;
 uint32_t E{},B{},J{},N{},P{},M{},F{};dwv1_grid grid{};
 std::vector<av1_body> body;std::vector<bcv1_pair> pair;std::vector<float> mu;
 std::vector<uint32_t> foot;std::vector<dwv1geo::Hull> foot_hull;dwv1geo::Hull cube_hull;
 std::vector<double> gravity; // [E,3]
 Cubes cubes,initial_cubes;
 bcv1_snapshot* initial_contact=nullptr;
 ~dwv1_scene(){bcv1_snapshot_destroy(initial_contact);bcv1_destroy(contact);av2_destroy(art);}
};
struct dwv1_snapshot {
 const dwv1_scene* owner;Saved art;bcv1_snapshot* contact=nullptr;Cubes cubes;
 explicit dwv1_snapshot(const dwv1_scene& s):owner(&s),art(s.E,s.J){}
 ~dwv1_snapshot(){bcv1_snapshot_destroy(contact);}
};
namespace {
std::vector<bcv1_body> duck_bodies(const dwv1_scene& s,const av2_state_view& view){
 std::vector<bcv1_body> out(size_t(s.E)*s.B);
 for(uint32_t e=0;e<s.E;e++)for(uint32_t b=0;b<s.B;b++){
  size_t i=size_t(e)*s.B+b;auto& x=out[i];for(int k=0;k<7;k++)x.state[k]=float(view.body_pose[i*7+k]);for(int k=0;k<6;k++)x.state[7+k]=float(view.body_velocity[i*6+k]);
  if(b){x.inverse_mass=float(1/s.body[b].mass);for(int k=0;k<3;k++)x.inverse_inertia[k]=float(1/s.body[b].principal_inertia[k]);}
 }
 return out;
}
bcv1_body cube_geometry_body(const dwv1_scene& s,const double* pose){
 bcv1_body b{};for(int k=0;k<7;k++)b.state[k]=float(pose[k]);
 if(s.grid.dynamic){double m=s.grid.cube_mass,I=m*s.grid.cube_size*s.grid.cube_size/6;b.inverse_mass=float(1/m);for(int k=0;k<3;k++)b.inverse_inertia[k]=float(1/I);}
 return b;
}
void layout_cubes(const dwv1_grid& g,uint32_t E,uint32_t M,uint32_t F,Cubes& c){
 c.pose.assign(size_t(E)*M*7,0);c.velocity.assign(size_t(E)*M*6,0);
 c.awake.assign(size_t(E)*M,g.dynamic?1:0);c.still.assign(size_t(E)*M,0);
 c.cache.assign(E,{});c.foot.assign(size_t(E)*F,0);
 double pitch=g.spacing;
 for(uint32_t e=0;e<E;e++)for(uint32_t ix=0;ix<g.nx;ix++)for(uint32_t iz=0;iz<g.nz;iz++){
  uint32_t m=ix*g.nz+iz;double* p=&c.pose[(size_t(e)*M+m)*7];
  double jitter=g.height_jitter*unit01(mix(g.seed^((uint64_t(ix)<<32)|(iz+1))));
  p[0]=g.origin_x+(double(ix)-(double(g.nx)-1)*.5)*pitch;
  p[1]=g.origin_y+(double(iz)-(double(g.nz)-1)*.5)*pitch;
  p[2]=g.base_height+jitter+g.cube_size*.5;p[6]=1;
 }
}
void commit_pair(dwv1_scene& s,av2_stage* a,bcx1_stage* c){
 need(av2_validate_commit(s.art,a)==0&&bcx1_validate_commit(s.contact,c)==0,DWV1_TRANSACTION);
 // Both validated under the scene lock; both commits only swap preallocated
 // payloads. A post-validation failure is an invariant breach, never a
 // recoverable return claiming a half-committed scene is usable.
 if(av2_commit(s.art,a)!=0)std::terminate();if(bcx1_commit(s.contact,c)!=0)std::terminate();
}
void duck_side(const av2_pre_view& pre,uint32_t e,uint32_t body,double sign,const float* point,const float* direction,double* out){
 const size_t B=pre.bodies,N=pre.dofs;
 const double* p=pre.body_pose+(size_t(e)*B+body)*7;double arm[3]={point[0]-p[0],point[1]-p[1],point[2]-p[2]};
 double torque[3]={arm[1]*direction[2]-arm[2]*direction[1],arm[2]*direction[0]-arm[0]*direction[2],arm[0]*direction[1]-arm[1]*direction[0]};
 for(size_t n=0;n<N;n++)for(size_t k=0;k<3;k++)out[n]+=sign*(double(direction[k])*pre.spatial_jacobian[((size_t(e)*B+body)*6+k)*N+n]+torque[k]*pre.spatial_jacobian[((size_t(e)*B+body)*6+3+k)*N+n]);
}
void cube_side(const double* com,double sign,const float* point,const float* direction,double* out6){
 double arm[3]={point[0]-com[0],point[1]-com[1],point[2]-com[2]};
 for(int k=0;k<3;k++)out6[k]+=sign*direction[k];
 out6[3]+=sign*(arm[1]*direction[2]-arm[2]*direction[1]);
 out6[4]+=sign*(arm[2]*direction[0]-arm[0]*direction[2]);
 out6[5]+=sign*(arm[0]*direction[1]-arm[1]*direction[0]);
}
void warm_point(const bcv1_manifold& m,const bcv1_point& x,const bcv1_manifold* prev,double out[3]){
 out[0]=out[1]=out[2]=0;
 if(!prev||!prev->count||dot3(m.normal,prev->normal)<=.98)return;
 for(uint32_t q=0;q<prev->count;q++){const auto& y=prev->points[q];
  double dist=0;for(int a=0;a<3;a++)dist+=(double(x.point[a])-y.point[a])*(double(x.point[a])-y.point[a]);
  if(x.feature==y.feature&&dist<.0004){out[0]=y.normal_impulse;float t[3];for(int a=0;a<3;a++)t[a]=prev->tangent1[a]*y.tangent_impulse[0]+prev->tangent2[a]*y.tangent_impulse[1];out[1]=dot3(t,m.tangent1);out[2]=dot3(t,m.tangent2);break;}
 }
}
// 2D uniform grid over cube centers (world x,y); vertical stacks share a cell.
struct Broadphase {
 double h;std::unordered_map<uint64_t,std::vector<uint32_t>> cells;
 static uint64_t cellkey(int64_t x,int64_t y){return (uint64_t(uint32_t(int32_t(x)))<<32)|uint32_t(int32_t(y));}
 void build(const dwv1_scene& s,const double* pose,uint32_t M){
  h=std::max(s.grid.spacing,s.grid.cube_size);cells.clear();cells.reserve(2*M+1);
  for(uint32_t m=0;m<M;m++){const double* p=pose+size_t(m)*7;cells[cellkey(int64_t(std::floor(p[0]/h)),int64_t(std::floor(p[1]/h)))].push_back(m);}
 }
 void query(double minx,double miny,double maxx,double maxy,std::vector<uint32_t>& out)const{
  out.clear();
  int64_t x0=int64_t(std::floor(minx/h)),x1=int64_t(std::floor(maxx/h)),y0=int64_t(std::floor(miny/h)),y1=int64_t(std::floor(maxy/h));
  need(x1-x0<4096&&y1-y0<4096,DWV1_CAPACITY);
  for(int64_t x=x0;x<=x1;x++)for(int64_t y=y0;y<=y1;y++){auto it=cells.find(cellkey(x,y));if(it!=cells.end())out.insert(out.end(),it->second.begin(),it->second.end());}
  std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());
 }
};
struct Find {
 std::vector<uint32_t> parent;
 explicit Find(uint32_t n):parent(n){for(uint32_t i=0;i<n;i++)parent[i]=i;}
 uint32_t find(uint32_t x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;}
 void join(uint32_t a,uint32_t b){parent[find(a)]=find(b);}
};
void integrate_cube(double* pose,const double* v,double dt){
 for(int k=0;k<3;k++)pose[k]+=dt*v[k];
 double q[4]={pose[3],pose[4],pose[5],pose[6]},w[3]={v[3],v[4],v[5]};
 double dq[4]={w[0]*q[3]+w[1]*q[2]-w[2]*q[1],w[1]*q[3]+w[2]*q[0]-w[0]*q[2],w[2]*q[3]+w[0]*q[1]-w[1]*q[0],-w[0]*q[0]-w[1]*q[1]-w[2]*q[2]};
 for(int k=0;k<4;k++)q[k]+=.5*dt*dq[k];
 double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);need(std::isfinite(n)&&n>1e-12,DWV1_SOLVER);
 for(int k=0;k<4;k++)pose[3+k]=q[k]/n;
 for(int k=0;k<3;k++)need(std::isfinite(pose[k])&&std::fabs(pose[k])<=1e4,DWV1_SOLVER);
}
// geometry helpers shared by step and query -------------------------------
struct EnvGeometry {
 std::vector<bcv1_body> cube_body,foot_body;
 Broadphase broad;std::vector<std::vector<uint32_t>> foot_candidates;
 std::vector<std::pair<uint32_t,uint32_t>> cube_pairs; // dynamic only
 std::vector<uint32_t> floor_candidates;               // dynamic only
};
void gather_geometry(const dwv1_scene& s,uint32_t e,const double* foot_pose /*[F,7] source*/,EnvGeometry& g){
 const uint32_t M=s.M;const double* cpose=s.cubes.pose.data()+size_t(e)*M*7;
 g.cube_body.resize(M);for(uint32_t m=0;m<M;m++)g.cube_body[m]=cube_geometry_body(s,cpose+size_t(m)*7);
 g.broad.build(s,cpose,M);
 double half=s.grid.cube_size*.5,reach=half*1.7320508075688772+BROAD_MARGIN;
 g.foot_body.resize(s.F);g.foot_candidates.assign(s.F,{});
 for(uint32_t f=0;f<s.F;f++){
  bcv1_body fb{};for(int k=0;k<7;k++)fb.state[k]=float(foot_pose[size_t(f)*7+k]);g.foot_body[f]=fb;
  auto vs=dwv1geo::vertices(s.foot_hull[f],fb);
  float lo[3]={1e30f,1e30f,1e30f},hi[3]={-1e30f,-1e30f,-1e30f};
  for(auto v:vs)for(int k=0;k<3;k++){lo[k]=std::min(lo[k],v[k]);hi[k]=std::max(hi[k],v[k]);}
  std::vector<uint32_t> nearby;g.broad.query(lo[0]-reach,lo[1]-reach,hi[0]+reach,hi[1]+reach,nearby);
  for(uint32_t c:nearby){const double* p=cpose+size_t(c)*7;
   if(p[2]+reach<lo[2]-BROAD_MARGIN||p[2]-reach>hi[2]+BROAD_MARGIN)continue;
   g.foot_candidates[f].push_back(c);}
 }
 g.cube_pairs.clear();g.floor_candidates.clear();
 if(!s.grid.dynamic)return;
 double touch=s.grid.cube_size*1.7320508075688772+BROAD_MARGIN;std::vector<uint32_t> nearby;
 for(uint32_t i=0;i<M;i++){
  const double* pi=cpose+size_t(i)*7;
  if(pi[2]-half*1.7320508075688772<=BROAD_MARGIN)g.floor_candidates.push_back(i);
  g.broad.query(pi[0]-touch,pi[1]-touch,pi[0]+touch,pi[1]+touch,nearby);
  for(uint32_t j:nearby){if(j<=i)continue;const double* pj=cpose+size_t(j)*7;
   double d2=0;for(int k=0;k<3;k++)d2+=(pi[k]-pj[k])*(pi[k]-pj[k]);
   if(d2<=touch*touch)g.cube_pairs.push_back({i,j});}
 }
}
void reset_scene(dwv1_scene& s,const dwv1_snapshot* snap,const uint8_t* mask){
 if(snap)need(snap->owner==&s);
 av2_stage* a=nullptr;bcx1_stage* c=nullptr;
 AS astage(nullptr,av2_stage_destroy);CS cstage(nullptr,bcx1_stage_destroy);
 if(snap){
  Saved merged(s.E,s.J);need(av2_capture(s.art,&merged.d)==0,DWV1_ARTICULATED);need(merged.d.binding==snap->art.d.binding,DWV1_TRANSACTION);
  for(uint32_t e=0;e<s.E;e++)if(!mask||mask[e]){std::copy_n(snap->art.q.data()+size_t(e)*(7+s.J),7+s.J,merged.q.data()+size_t(e)*(7+s.J));std::copy_n(snap->art.v.data()+size_t(e)*s.N,s.N,merged.v.data()+size_t(e)*s.N);std::copy_n(snap->art.w.data()+size_t(e)*3*s.J,3*s.J,merged.w.data()+size_t(e)*3*s.J);merged.t[e]=snap->art.t[e];merged.count[e]=snap->art.count[e];}
  need(av2_prepare_restore(s.art,&merged.d,&a)==0,DWV1_ARTICULATED);astage.reset(a);
 }else{std::vector<uint8_t> normalized(s.E);for(uint32_t e=0;e<s.E;e++)normalized[e]=!mask||mask[e]?1:0;need(av2_prepare_reset(s.art,normalized.data(),s.E,&a)==0,DWV1_ARTICULATED);astage.reset(a);}
 need(bcx1_prepare_restore(s.contact,snap?snap->contact:s.initial_contact,mask,&c)==0,DWV1_CONTACT);cstage.reset(c);
 auto view=desc<av2_state_view>();need(av2_stage_read(a,&view)==0,DWV1_ARTICULATED);std::vector<double> clocks(s.E);need(bcx1_stage_read(c,nullptr,nullptr,clocks.data())==0,DWV1_CONTACT);
 for(uint32_t e=0;e<s.E;e++)need(clocks[e]==view.time[e],DWV1_TRANSACTION);
 // stage the cube payload fully before the two-owner commit
 Cubes staged=s.cubes;const Cubes& source=snap?snap->cubes:s.initial_cubes;
 for(uint32_t e=0;e<s.E;e++)if(!mask||mask[e]){
  std::copy_n(source.pose.data()+size_t(e)*s.M*7,size_t(s.M)*7,staged.pose.data()+size_t(e)*s.M*7);
  std::copy_n(source.velocity.data()+size_t(e)*s.M*6,size_t(s.M)*6,staged.velocity.data()+size_t(e)*s.M*6);
  std::copy_n(source.awake.data()+size_t(e)*s.M,s.M,staged.awake.data()+size_t(e)*s.M);
  std::copy_n(source.still.data()+size_t(e)*s.M,s.M,staged.still.data()+size_t(e)*s.M);
  std::copy_n(source.foot.data()+size_t(e)*s.F,s.F,staged.foot.data()+size_t(e)*s.F);
  staged.cache[e]=source.cache[e];
 }
 commit_pair(s,a,c);
 s.cubes=std::move(staged);
}
}
extern "C" {
int dwv1_create(const dwv1_registration* r,dwv1_scene** out){if(!out)return DWV1_INVALID;*out=nullptr;return guard([&]{
 need(r&&r->articulation&&r->articulation->model&&r->shapes&&r->reserved==0&&r->grid.reserved==0&&r->pairs<=16&&(!r->pairs||(r->contact_pairs&&r->friction)));
 auto s=std::make_unique<dwv1_scene>();const auto& a=*r->articulation;const auto& m=*a.model;
 need(m.bodies<=32&&m.joints<=26&&a.environments>0&&a.environments<=4096&&m.body&&a.gravity);
 s->E=a.environments;s->B=m.bodies;s->J=m.joints;s->N=6+m.joints;s->P=r->pairs;s->body.assign(m.body,m.body+m.bodies);
 if(s->P){s->pair.assign(r->contact_pairs,r->contact_pairs+s->P);s->mu.assign(r->friction,r->friction+size_t(s->E)*s->P);}
 for(uint32_t b=0;b<s->B;b++)need(r->shapes[b].fixed==(b==0));
 need(r->shapes[0].kind==BCV1_PLANE&&r->shapes[0].plane_normal[0]==0&&r->shapes[0].plane_normal[1]==0&&std::fabs(r->shapes[0].plane_normal[2]-1)<2e-5f&&r->shapes[0].plane_offset==0);
 for(uint32_t b=1;b<s->B;b++)if(r->shapes[b].kind==BCV1_CONVEX){s->foot.push_back(b);s->foot_hull.push_back(dwv1geo::build_hull(r->shapes[b].vertices,r->shapes[b].vertex_count));}
 s->F=uint32_t(s->foot.size());need(s->F>=1&&s->F<=4);
 const dwv1_grid& g=r->grid;s->grid=g;
 need(g.nx>=1&&g.nz>=1&&uint64_t(g.nx)*g.nz<=1024);s->M=g.nx*g.nz;need(uint64_t(s->E)*s->M<=262144,DWV1_CAPACITY);
 need(std::isfinite(g.cube_size)&&g.cube_size>=.01&&g.cube_size<=.5);
 if(s->grid.spacing==0)s->grid.spacing=g.cube_size;
 need(std::isfinite(s->grid.spacing)&&s->grid.spacing>0&&s->grid.spacing<=10);
 need(!g.dynamic||s->grid.spacing>=g.cube_size-1e-12);
 need(std::isfinite(g.base_height)&&g.base_height>=0&&g.base_height<=100);
 need(std::isfinite(g.height_jitter)&&g.height_jitter>=0&&g.height_jitter<=1);
 need(std::isfinite(g.origin_x)&&std::fabs(g.origin_x)<=1e3&&std::isfinite(g.origin_y)&&std::fabs(g.origin_y)<=1e3);
 need(g.dynamic<=1&&std::isfinite(g.friction)&&g.friction>=0&&g.friction<=4);
 if(g.dynamic)need(std::isfinite(g.cube_mass)&&g.cube_mass>0&&g.cube_mass<=100);
 float corner[8][3];double half=g.cube_size*.5;
 for(int i=0;i<8;i++)for(int k=0;k<3;k++)corner[i][k]=float(half*((i>>k&1)?1:-1));
 s->cube_hull=dwv1geo::build_hull(corner,8);
 s->gravity.assign(a.gravity,a.gravity+size_t(s->E)*3);
 layout_cubes(s->grid,s->E,s->M,s->F,s->initial_cubes);s->cubes=s->initial_cubes;
 need(av2_create(&a,&s->art)==0,DWV1_ARTICULATED);auto state=desc<av2_state_view>();need(av2_read(s->art,&state)==0,DWV1_ARTICULATED);auto bs=duck_bodies(*s,state);
 std::vector<float> gravity(size_t(s->E)*3);for(size_t i=0;i<gravity.size();i++)gravity[i]=float(a.gravity[i]);
 bcv1_registration cr{1,s->E,s->B,s->P,r->shapes,s->pair.data(),bs.data(),gravity.data(),s->mu.data()};
 need(bcv1_create(&cr,&s->contact)==0,DWV1_CONTACT);need(bcv1_capture(s->contact,&s->initial_contact)==0,DWV1_CONTACT);*out=s.release();
 });}
void dwv1_destroy(dwv1_scene* s){delete s;}
int dwv1_step(dwv1_scene* s,const av2_step* step,uint32_t max_iterations,double tolerance,dwv1_diagnostic* diagnostic){return guard([&]{
 need(s&&step&&diagnostic);std::lock_guard<std::mutex> lock(s->lock);std::fill_n(diagnostic,s->E,dwv1_diagnostic{});
 for(uint32_t e=0;e<s->E;e++){diagnostic[e].environment=e;diagnostic[e].phase=1;}
 av2_pre* rawpre=nullptr;int rc=av2_prepare(s->art,step,&rawpre);if(rc){diagnostic[0].native_status=rc;throw Error{DWV1_ARTICULATED};}
 AP pre(rawpre,av2_pre_destroy);auto p=desc<av2_pre_view>();need(av2_pre_read(pre.get(),&p)==0,DWV1_ARTICULATED);
 const size_t N=s->N,JRows=3*size_t(s->J);const uint32_t M=s->M;const double dt=step->dt;
 const double cmass=s->grid.dynamic?s->grid.cube_mass:0,cinertia=cmass*s->grid.cube_size*s->grid.cube_size/6;
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=2;
 std::vector<bcv1_manifold> manifolds(size_t(s->E)*s->P),old(size_t(s->E)*s->P);
 need(bcv1_query(s->contact,manifolds.data())==0&&bcv1_read(s->contact,nullptr,old.data(),nullptr)==0,DWV1_CONTACT);
 std::vector<double> v(size_t(s->E)*N),ji(size_t(s->E)*JRows),ci(size_t(s->E)*N,0);
 Cubes next=s->cubes; // staged; committed only after both foreign owners validate
 EnvGeometry geo;
 for(uint32_t e=0;e<s->E;e++){
  auto& d=diagnostic[e];d.phase=2;
  const double* cpose=s->cubes.pose.data()+size_t(e)*M*7;
  const double* cvel=s->cubes.velocity.data()+size_t(e)*M*6;
  uint8_t* awake=next.awake.data()+size_t(e)*M;uint32_t* still=next.still.data()+size_t(e)*M;
  std::vector<double> footpose(size_t(s->F)*7);
  for(uint32_t f=0;f<s->F;f++)std::copy_n(p.body_pose+(size_t(e)*s->B+s->foot[f])*7,7,footpose.data()+size_t(f)*7);
  gather_geometry(*s,e,footpose.data(),geo);
  std::vector<CM> cms;
  auto wake=[&](uint32_t c){if(!awake[c]){awake[c]=1;still[c]=0;}};
  for(uint32_t f=0;f<s->F;f++)for(uint32_t c:geo.foot_candidates[f]){
   auto m=dwv1geo::convex_contact(s->foot_hull[f],geo.foot_body[f],s->cube_hull,geo.cube_body[c]);
   if(!m.count)continue;
   cms.push_back({pairkey(DWV1_KIND_DUCK,s->foot[f],DWV1_KIND_CUBE,c),DWV1_KIND_DUCK,s->foot[f],DWV1_KIND_CUBE,c,m});
   if(s->grid.dynamic)wake(c);
  }
  if(s->grid.dynamic){
   std::vector<char> done(geo.cube_pairs.size(),0);bool changed=true;
   while(changed){changed=false;
    for(size_t i=0;i<geo.cube_pairs.size();i++){
     if(done[i])continue;uint32_t ca=geo.cube_pairs[i].first,cb=geo.cube_pairs[i].second;
     if(!awake[ca]&&!awake[cb])continue;done[i]=1;
     auto m=dwv1geo::convex_contact(s->cube_hull,geo.cube_body[ca],s->cube_hull,geo.cube_body[cb]);
     if(!m.count)continue;
     cms.push_back({pairkey(DWV1_KIND_CUBE,ca,DWV1_KIND_CUBE,cb),DWV1_KIND_CUBE,ca,DWV1_KIND_CUBE,cb,m});
     if(!awake[ca]){wake(ca);changed=true;}if(!awake[cb]){wake(cb);changed=true;}
    }
   }
   for(uint32_t c:geo.floor_candidates){
    if(!awake[c])continue;
    auto m=dwv1geo::plane_manifold(s->cube_hull,geo.cube_body[c],{0,0,1},0,false);
    if(m.count)cms.push_back({pairkey(DWV1_KIND_CUBE,c,DWV1_KIND_FLOOR,0),DWV1_KIND_CUBE,c,DWV1_KIND_FLOOR,0,m});
   }
  }
  std::sort(cms.begin(),cms.end(),[](const CM& a,const CM& b){return a.key<b.key;});
  // Duplicate-support suppression: two contact points with near-parallel
  // normals closer than 1mm whose rows touch identical dynamic columns (the
  // duck and/or the same awake cubes; only static-side partners differ, e.g.
  // a foot straddling the shared boundary of two coplanar static cubes) are
  // one constraint emitted twice. civ1's scalar sweep stalls on such
  // near-identical normal row pairs, so keep only the deeper point.
  {
   auto dynamic_cube=[&](uint32_t kind,uint32_t idx){return kind==DWV1_KIND_CUBE&&s->grid.dynamic&&awake[idx];};
   auto signature=[&](const CM& x){
    uint64_t duck=x.kind_a==DWV1_KIND_DUCK?1:0;
    uint64_t ca=dynamic_cube(x.kind_a,x.index_a)?x.index_a:0xFFFFu;
    uint64_t cb=dynamic_cube(x.kind_b,x.index_b)?x.index_b:0xFFFFu;
    if(ca>cb)std::swap(ca,cb);
    return (duck<<48)|(ca<<24)|cb;};
   struct Ref{size_t cm;uint32_t pt;uint64_t sig;};
   std::vector<Ref> kept;std::vector<std::vector<char>> drop(cms.size());
   for(size_t i=0;i<cms.size();i++){
    drop[i].assign(cms[i].m.count,0);uint64_t sig=signature(cms[i]);
    for(uint32_t k=0;k<cms[i].m.count;k++){
     int action=0;
     for(auto& ref:kept){
      if(ref.sig!=sig||drop[ref.cm][ref.pt])continue;
      const auto& a=cms[ref.cm].m;const auto& b=cms[i].m;
      if(dot3(a.normal,b.normal)<=.999)continue;
      double d2=0;for(int c=0;c<3;c++){double t=double(b.points[k].point[c])-a.points[ref.pt].point[c];d2+=t*t;}
      if(d2>=1e-6)continue;
      if(b.points[k].depth>a.points[ref.pt].depth){drop[ref.cm][ref.pt]=1;ref={i,k,sig};action=2;}
      else action=1;
      break;
     }
     if(action==0)kept.push_back({i,k,sig});
     else if(action==1)drop[i][k]=1;
    }
   }
   for(size_t i=0;i<cms.size();i++){
    auto& m=cms[i].m;uint32_t keep=0;
    for(uint32_t k=0;k<m.count;k++)if(!drop[i][k])m.points[keep++]=m.points[k];
    m.count=keep;
   }
   cms.erase(std::remove_if(cms.begin(),cms.end(),[](const CM& x){return x.m.count==0;}),cms.end());
  }
  d.phase=3;
  // island partition over awake cubes + the duck (node M)
  Find uf(M+1);
  auto cube_dynamic=[&](const CM& x,int side){uint32_t kind=side?x.kind_b:x.kind_a,idx=side?x.index_b:x.index_a;return kind==DWV1_KIND_CUBE&&s->grid.dynamic&&awake[idx];};
  for(const auto& x:cms){
   if(x.kind_a==DWV1_KIND_DUCK&&cube_dynamic(x,1))uf.join(M,x.index_b);
   if(x.kind_a==DWV1_KIND_CUBE&&x.kind_b==DWV1_KIND_CUBE&&cube_dynamic(x,0)&&cube_dynamic(x,1))uf.join(x.index_a,x.index_b);
  }
  std::vector<int32_t> island_of(M,-1);std::vector<std::vector<uint32_t>> island_cubes(1);
  uint32_t duck_root=uf.find(M);
  std::vector<int32_t> root_island; // lazy map root->island id
  root_island.assign(M+1,-1);root_island[duck_root]=0;
  uint32_t islands=1;
  for(uint32_t c=0;c<M;c++){if(!s->grid.dynamic||!awake[c])continue;uint32_t root=uf.find(c);
   if(root_island[root]<0){root_island[root]=int32_t(islands++);island_cubes.push_back({});}
   island_of[c]=root_island[root];island_cubes[size_t(root_island[root])].push_back(c);}
  std::vector<std::vector<size_t>> island_cms(islands);
  for(size_t i=0;i<cms.size();i++){const auto& x=cms[i];
   int32_t target=x.kind_a==DWV1_KIND_DUCK?0:cube_dynamic(x,0)?island_of[x.index_a]:island_of[x.index_b];
   need(target>=0,DWV1_TRANSACTION);island_cms[size_t(target)].push_back(i);}
  std::vector<size_t> cm_first_row(cms.size(),0);
  std::vector<double> impulse_delta(M,0);
  d.islands=islands;
  // solve each island against PRE state
  for(uint32_t is=0;is<islands;is++){
   const bool has_duck=is==0;const auto& members=island_cubes[is];
   const size_t K=members.size(),Ni=(has_duck?N:0)+6*K;
   std::vector<int32_t> offset(M,-1);
   for(size_t k=0;k<K;k++)offset[members[k]]=int32_t((has_duck?N:0)+6*k);
   d.max_island_dofs=std::max(d.max_island_dofs,uint32_t(Ni));
   if(has_duck)d.duck_island_cubes=uint32_t(K);
   std::vector<double> mass(Ni*Ni,0),smooth(Ni,0);
   if(has_duck){
    for(size_t i=0;i<N;i++)for(size_t j=0;j<N;j++)mass[i*Ni+j]=p.mass[(size_t(e)*N+i)*N+j];
    std::copy_n(p.smooth_velocity+size_t(e)*N,N,smooth.data());
   }
   for(size_t k=0;k<K;k++){
    size_t o=(has_duck?N:0)+6*k;const double* g=s->gravity.data()+size_t(e)*3;const double* cv=cvel+size_t(members[k])*6;
    for(int i=0;i<3;i++){mass[(o+i)*Ni+o+i]=cmass;mass[(o+3+i)*Ni+o+3+i]=cinertia;smooth[o+i]=cv[i]+dt*g[i];smooth[o+3+i]=cv[3+i];}
   }
   Rows rows(Ni);std::vector<double> g(Ni);
   if(has_duck){
    for(size_t r=0;r<JRows;r++){size_t k=size_t(e)*JRows+r;std::fill(g.begin(),g.end(),0.);std::copy_n(p.row_jacobian+k*N,N,g.data());
     rows.add(g.data(),p.row_target[k],p.row_regularizer[k],p.row_lower[k],p.row_upper[k],p.row_warm_impulse[k]);
     if(r>=s->J&&p.row_active[k])d.active_limits++;}
    for(uint32_t pair=0;pair<s->P;pair++){
     auto& m=manifolds[size_t(e)*s->P+pair];const auto& previous=old[size_t(e)*s->P+pair];double mu=s->mu[size_t(e)*s->P+pair];
     for(uint32_t k=0;k<m.count;k++){
      auto& x=m.points[k];double warm[3];warm_point(m,x,&previous,warm);
      rows.contacts.push_back({uint32_t(rows.target.size()),mu});
      const float* direction[3]={m.normal,m.tangent1,m.tangent2};
      for(int a=0;a<3;a++){std::fill(g.begin(),g.end(),0.);
       duck_side(p,e,s->pair[pair].body_a,-1,x.point,direction[a],g.data());
       duck_side(p,e,s->pair[pair].body_b,1,x.point,direction[a],g.data());
       double target=a==0?std::min(1.,.2*std::max(0.,double(x.depth)-2e-6)/dt):0;
       rows.add(g.data(),target,0,a==0?0:-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),warm[a]);}
      d.contact_points++;d.maximum_penetration=std::max(d.maximum_penetration,double(x.depth));
     }
    }
   }
   for(size_t ci_index:island_cms[is]){
    auto& x=cms[ci_index];auto& m=x.m;cm_first_row[ci_index]=rows.target.size();
    const bcv1_manifold* prev=nullptr;auto it=s->cubes.cache[e].find(x.key);if(it!=s->cubes.cache[e].end())prev=&it->second;
    for(uint32_t k=0;k<m.count;k++){
     auto& pt=m.points[k];double warm[3];warm_point(m,pt,prev,warm);
     rows.contacts.push_back({uint32_t(rows.target.size()),s->grid.friction});
     const float* direction[3]={m.normal,m.tangent1,m.tangent2};
     for(int a=0;a<3;a++){std::fill(g.begin(),g.end(),0.);
      if(x.kind_a==DWV1_KIND_DUCK)duck_side(p,e,x.index_a,-1,pt.point,direction[a],g.data());
      else if(cube_dynamic(x,0)&&offset[x.index_a]>=0)cube_side(cpose+size_t(x.index_a)*7,-1,pt.point,direction[a],g.data()+offset[x.index_a]);
      if(x.kind_b==DWV1_KIND_CUBE&&cube_dynamic(x,1)&&offset[x.index_b]>=0)cube_side(cpose+size_t(x.index_b)*7,1,pt.point,direction[a],g.data()+offset[x.index_b]);
      double target=a==0?std::min(1.,.2*std::max(0.,double(pt.depth)-2e-6)/dt):0;
      rows.add(g.data(),target,0,a==0?0:-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),warm[a]);}
     d.contact_points++;d.maximum_penetration=std::max(d.maximum_penetration,double(pt.depth));
    }
   }
   const size_t R=rows.target.size();
   need(Ni<=256&&R<=1536&&rows.contacts.size()<=512,DWV1_CAPACITY);
   std::vector<double> impulse(R),vel(Ni);
   if(R==0){std::copy(smooth.begin(),smooth.end(),vel.begin());}
   else{
    civ1_problem problem{uint32_t(Ni),uint32_t(R),uint32_t(rows.contacts.size()),max_iterations,tolerance,mass.data(),smooth.data(),rows.g.data(),rows.target.data(),rows.r.data(),rows.lo.data(),rows.hi.data(),rows.warm.data(),rows.contacts.data()};
    civ1_result result{};result.velocity=vel.data();result.impulse=impulse.data();rc=civ1_solve(&problem,&result);
    if(rc){d.native_status=rc;throw Error{DWV1_SOLVER};}
    d.iterations=std::max(d.iterations,result.iterations);
    d.joint_residual=std::max(d.joint_residual,result.joint_residual);
    d.normal_residual=std::max(d.normal_residual,result.normal_residual);
    d.tangent_residual=std::max(d.tangent_residual,result.tangent_residual);
    d.momentum_residual=std::max(d.momentum_residual,result.momentum_residual);
   }
   if(has_duck){
    std::copy_n(vel.data(),N,v.data()+size_t(e)*N);
    std::copy_n(impulse.data(),JRows,ji.data()+size_t(e)*JRows);
    for(size_t r=JRows;r<R;r++)for(size_t n=0;n<N;n++)ci[size_t(e)*N+n]+=rows.g[r*Ni+n]*impulse[r];
    size_t r=JRows;
    for(uint32_t pair=0;pair<s->P;pair++){auto& m=manifolds[size_t(e)*s->P+pair];for(uint32_t k=0;k<m.count;k++){auto& x=m.points[k];x.normal_impulse=float(impulse[r]);x.tangent_impulse[0]=float(impulse[r+1]);x.tangent_impulse[1]=float(impulse[r+2]);r+=3;d.maximum_normal_impulse=std::max(d.maximum_normal_impulse,double(x.normal_impulse));}}
   }
   for(size_t k=0;k<K;k++)std::copy_n(vel.data()+(has_duck?N:0)+6*k,6,next.velocity.data()+(size_t(e)*M+members[k])*6);
   for(size_t ci_index:island_cms[is]){
    auto& x=cms[ci_index];auto& m=x.m;size_t r=cm_first_row[ci_index];
    const bcv1_manifold* prev=nullptr;auto it=s->cubes.cache[e].find(x.key);if(it!=s->cubes.cache[e].end())prev=&it->second;
    for(uint32_t k=0;k<m.count;k++){
     auto& pt=m.points[k];double warm[3];warm_point(m,pt,prev,warm);
     pt.normal_impulse=float(impulse[r]);pt.tangent_impulse[0]=float(impulse[r+1]);pt.tangent_impulse[1]=float(impulse[r+2]);
     d.maximum_normal_impulse=std::max(d.maximum_normal_impulse,double(pt.normal_impulse));
     double delta=std::fabs(impulse[r]-warm[0]);
     if(x.kind_a==DWV1_KIND_CUBE)impulse_delta[x.index_a]=std::max(impulse_delta[x.index_a],delta);
     if(x.kind_b==DWV1_KIND_CUBE)impulse_delta[x.index_b]=std::max(impulse_delta[x.index_b],delta);
     r+=3;
    }
   }
  }
  // integrate awake cubes and update sleep bookkeeping (staged)
  if(s->grid.dynamic){
   const double impulse_gate=.1*cmass*9.81*dt;
   for(uint32_t c=0;c<M;c++){
    if(!awake[c])continue;
    double* pose=next.pose.data()+(size_t(e)*M+c)*7;const double* vn=next.velocity.data()+(size_t(e)*M+c)*6;
    integrate_cube(pose,vn,dt);
    double lin=std::sqrt(vn[0]*vn[0]+vn[1]*vn[1]+vn[2]*vn[2]),ang=std::sqrt(vn[3]*vn[3]+vn[4]*vn[4]+vn[5]*vn[5]);
    need(std::isfinite(lin)&&std::isfinite(ang)&&lin<=1e3&&ang<=1e3,DWV1_SOLVER);
    if(lin<SLEEP_LINEAR&&ang<SLEEP_ANGULAR&&impulse_delta[c]<=impulse_gate)still[c]++;else still[c]=0;
    if(still[c]>=SLEEP_TICKS){awake[c]=0;std::fill_n(next.velocity.data()+(size_t(e)*M+c)*6,6,0.);}
   }
  }
  uint32_t awake_count=0;for(uint32_t c=0;c<M;c++)awake_count+=awake[c]?1:0;d.awake_cubes=awake_count;
  // rebuild warm cache and foot flags for this environment
  auto& cache=next.cache[e];cache.clear();
  for(const auto& x:cms)if(x.m.count)cache[x.key]=x.m;
  for(uint32_t f=0;f<s->F;f++){
   uint8_t flag=0;
   for(uint32_t pair=0;pair<s->P;pair++){
    const auto& m=manifolds[size_t(e)*s->P+pair];if(!m.count)continue;
    uint32_t a=s->pair[pair].body_a,b=s->pair[pair].body_b;
    if((a==s->foot[f]&&b==0)||(b==s->foot[f]&&a==0))flag=1;
   }
   for(const auto& x:cms)if(x.m.count&&x.kind_a==DWV1_KIND_DUCK&&x.index_a==s->foot[f])flag=1;
   next.foot[size_t(e)*s->F+f]=flag;
  }
 }
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=4;
 auto solution=desc<av2_solution>();solution.velocity=v.data();solution.joint_impulse=ji.data();solution.contact_generalized_impulse=ci.data();
 av2_stage* araw=nullptr;rc=av2_complete(pre.get(),&solution,&araw);
 if(rc){diagnostic[0].native_status=rc;throw Error{DWV1_ARTICULATED};}
 AS astage(araw,av2_stage_destroy);auto state=desc<av2_state_view>();need(av2_stage_read(astage.get(),&state)==0,DWV1_ARTICULATED);auto bs=duck_bodies(*s,state);
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=5;
 bcx1_stage* craw=nullptr;rc=bcx1_prepare_solved(s->contact,bs.data(),manifolds.data(),step->dt,&craw);
 if(rc){diagnostic[0].native_status=rc;throw Error{DWV1_CONTACT};}
 CS cstage(craw,bcx1_stage_destroy);
 std::vector<double> clocks(s->E);need(bcx1_stage_read(cstage.get(),nullptr,nullptr,clocks.data())==0,DWV1_CONTACT);
 for(uint32_t e=0;e<s->E;e++)need(clocks[e]==state.time[e],DWV1_TRANSACTION);
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=6;
 commit_pair(*s,astage.get(),cstage.get());
 s->cubes=std::move(next);
 });}
int dwv1_read(dwv1_scene* s,double* q,double* v,double* w,double* t,uint64_t* count,bcv1_body* b,bcv1_manifold* cache,bcv1_manifold* geometry,double* cube_pose,double* cube_velocity,uint8_t* cube_awake,uint8_t* foot_contact){return guard([&]{
 need(s);std::lock_guard<std::mutex> lock(s->lock);auto view=desc<av2_state_view>();need(av2_read(s->art,&view)==0,DWV1_ARTICULATED);
 std::vector<bcv1_manifold> ms(size_t(s->E)*s->P);if(geometry)need(bcv1_query(s->contact,ms.data())==0,DWV1_CONTACT);
 if(q)std::copy_n(view.qpos,size_t(s->E)*(7+s->J),q);if(v)std::copy_n(view.velocity,size_t(s->E)*s->N,v);if(w)std::copy_n(view.joint_warm_force,size_t(s->E)*3*s->J,w);if(t)std::copy_n(view.time,s->E,t);if(count)std::copy_n(view.step_count,s->E,count);
 need(bcv1_read(s->contact,b,cache,nullptr)==0,DWV1_CONTACT);if(geometry)std::copy(ms.begin(),ms.end(),geometry);
 if(cube_pose)std::copy(s->cubes.pose.begin(),s->cubes.pose.end(),cube_pose);
 if(cube_velocity)std::copy(s->cubes.velocity.begin(),s->cubes.velocity.end(),cube_velocity);
 if(cube_awake)std::copy(s->cubes.awake.begin(),s->cubes.awake.end(),cube_awake);
 if(foot_contact)std::copy(s->cubes.foot.begin(),s->cubes.foot.end(),foot_contact);
 });}
int dwv1_query(dwv1_scene* s,uint32_t environment,dwv1_contact* output,uint32_t capacity,uint32_t* count){return guard([&]{
 need(s&&count&&environment<s->E);std::lock_guard<std::mutex> lock(s->lock);
 auto view=desc<av2_state_view>();need(av2_read(s->art,&view)==0,DWV1_ARTICULATED);
 std::vector<double> footpose(size_t(s->F)*7);
 for(uint32_t f=0;f<s->F;f++)std::copy_n(view.body_pose+(size_t(environment)*s->B+s->foot[f])*7,7,footpose.data()+size_t(f)*7);
 EnvGeometry geo;gather_geometry(*s,environment,footpose.data(),geo);
 std::vector<dwv1_contact> found;
 for(uint32_t f=0;f<s->F;f++)for(uint32_t c:geo.foot_candidates[f]){
  auto m=dwv1geo::convex_contact(s->foot_hull[f],geo.foot_body[f],s->cube_hull,geo.cube_body[c]);
  if(m.count)found.push_back({DWV1_KIND_DUCK,s->foot[f],DWV1_KIND_CUBE,c,m});
 }
 for(auto pr:geo.cube_pairs){
  auto m=dwv1geo::convex_contact(s->cube_hull,geo.cube_body[pr.first],s->cube_hull,geo.cube_body[pr.second]);
  if(m.count)found.push_back({DWV1_KIND_CUBE,pr.first,DWV1_KIND_CUBE,pr.second,m});
 }
 for(uint32_t c:geo.floor_candidates){
  auto m=dwv1geo::plane_manifold(s->cube_hull,geo.cube_body[c],{0,0,1},0,false);
  if(m.count)found.push_back({DWV1_KIND_CUBE,c,DWV1_KIND_FLOOR,0,m});
 }
 *count=uint32_t(found.size());
 if(output){need(capacity>=found.size(),DWV1_CAPACITY);std::copy(found.begin(),found.end(),output);}
 });}
int dwv1_override_cube(dwv1_scene* s,uint32_t environment,uint32_t cube,const double* pose,const double* velocity){return guard([&]{
 need(s&&pose&&velocity);std::lock_guard<std::mutex> lock(s->lock);
 need(s->grid.dynamic&&environment<s->E&&cube<s->M);
 for(int k=0;k<7;k++)need(std::isfinite(pose[k]));
 for(int k=0;k<3;k++)need(std::fabs(pose[k])<=1e4);
 double n=std::sqrt(pose[3]*pose[3]+pose[4]*pose[4]+pose[5]*pose[5]+pose[6]*pose[6]);need(std::fabs(n-1)<1e-6);
 for(int k=0;k<6;k++)need(std::isfinite(velocity[k])&&std::fabs(velocity[k])<=1e3);
 std::copy_n(pose,7,s->cubes.pose.data()+(size_t(environment)*s->M+cube)*7);
 std::copy_n(velocity,6,s->cubes.velocity.data()+(size_t(environment)*s->M+cube)*6);
 s->cubes.awake[size_t(environment)*s->M+cube]=1;s->cubes.still[size_t(environment)*s->M+cube]=0;
 auto& cache=s->cubes.cache[environment];
 for(auto it=cache.begin();it!=cache.end();){
  uint64_t key=it->first;uint32_t ka=uint32_t(key>>56),ia=uint32_t((key>>40)&0xFFFFu),kb=uint32_t((key>>32)&0xFFu),ib=uint32_t(key&0xFFFFFFFFu);
  bool hit=(ka==DWV1_KIND_CUBE&&ia==cube)||(kb==DWV1_KIND_CUBE&&ib==cube);
  if(hit)it=cache.erase(it);else ++it;
 }
 });}
int dwv1_capture(dwv1_scene* s,dwv1_snapshot** out){if(!out)return DWV1_INVALID;return guard([&]{
 need(s);std::lock_guard<std::mutex> lock(s->lock);auto snap=std::make_unique<dwv1_snapshot>(*s);
 need(av2_capture(s->art,&snap->art.d)==0,DWV1_ARTICULATED);need(bcv1_capture(s->contact,&snap->contact)==0,DWV1_CONTACT);
 snap->cubes=s->cubes;*out=snap.release();});}
void dwv1_snapshot_destroy(dwv1_snapshot* s){delete s;}
int dwv1_restore(dwv1_scene* s,const dwv1_snapshot* snap,const uint8_t* mask){return guard([&]{need(s&&snap);std::lock_guard<std::mutex> lock(s->lock);reset_scene(*s,snap,mask);});}
int dwv1_reset(dwv1_scene* s,const uint8_t* mask){return guard([&]{need(s);std::lock_guard<std::mutex> lock(s->lock);reset_scene(*s,nullptr,mask);});}
}
