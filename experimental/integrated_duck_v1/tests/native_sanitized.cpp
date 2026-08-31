// SPDX-License-Identifier: MIT
#include "integrated_duck_v1.h"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
int main(){
 av1_body body[2]={{0,{0,0,0}},{1,{.02,.03,.04}}};
 double reference[7]={0,0,.1,0,0,0,1};
 av1_model model{};model.struct_size=sizeof(model);model.version=AV1_ABI;model.bodies=2;model.body=body;model.root_source_to_principal[6]=1;model.reference_qpos=reference;
 double q[14];std::memcpy(q,reference,sizeof(reference));std::memcpy(q+7,reference,sizeof(reference));q[7]=2;
 double v[12]={0,0,-1,0,0,0,0,0,-1,0,0,0};double gravity[6]{};
 av2_registration art{};art.struct_size=sizeof(art);art.version=AV2_ABI;art.environments=2;art.model=&model;art.initial_qpos=q;art.initial_velocity=v;art.gravity=gravity;
 bcv1_shape shapes[2]{};shapes[0].caller_id=0;shapes[0].kind=2;shapes[0].fixed=1;shapes[0].plane_normal[2]=1;
 shapes[1].caller_id=1;shapes[1].kind=1;shapes[1].vertex_count=8;for(int i=0;i<8;i++)for(int k=0;k<3;k++)shapes[1].vertices[i][k]=(i&(1<<k))?.1f:-.1f;
 bcv1_pair pair{1,0,1};float friction[2]={.6f,.6f};idv1_registration reg{&art,1,0,shapes,&pair,friction};idv1_scene* s=nullptr;
 assert(idv1_create(&reg,&s)==0);idv1_snapshot* initial=nullptr;assert(idv1_capture(s,&initial)==0);
 av2_step step{};step.struct_size=sizeof(step);step.version=AV2_ABI;step.dt=.002;step.momentum_tolerance=step.joint_impulse_tolerance=1e-8;idv1_diagnostic diagnostics[2];
 for(int i=0;i<3;i++)assert(idv1_step(s,&step,4096,1e-8,diagnostics)==0);
 std::array<double,14> beforeq{},afterq{};double clock[2];uint64_t count[2];std::array<bcv1_manifold,2> cache{};
 assert(idv1_read(s,beforeq.data(),nullptr,nullptr,clock,count,nullptr,cache.data(),nullptr)==0);assert(count[0]==3&&clock[0]==.006);
 uint8_t mask[2]={255,0};assert(idv1_restore(s,initial,mask)==0);assert(idv1_read(s,afterq.data(),nullptr,nullptr,clock,count,nullptr,nullptr,nullptr)==0);
 assert(std::memcmp(afterq.data(),q,7*sizeof(double))==0);assert(std::memcmp(afterq.data()+7,beforeq.data()+7,7*sizeof(double))==0);assert(count[0]==0&&count[1]==3);
 assert(idv1_reset(s,nullptr)==0);assert(idv1_step(s,&step,4096,1e-8,diagnostics)==0);
 idv1_snapshot_destroy(initial);idv1_destroy(s);std::cout<<"combined ASAN/UBSAN lifecycle passed\n";
}
