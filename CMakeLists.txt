cmake_minimum_required(VERSION 3.24)
project(box3d_cuda VERSION 0.1.0 LANGUAGES C CXX CUDA)

add_library(box3d_cuda_cabi SHARED
  csrc/native_api.cu
  csrc/native_scene_v2.cu)
target_include_directories(box3d_cuda_cabi
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(box3d_cuda_cabi PRIVATE BOX3D_CUDA_BUILD_SHARED=1)
set_target_properties(box3d_cuda_cabi PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  OUTPUT_NAME box3d_cuda)

include(GNUInstallDirs)
install(TARGETS box3d_cuda_cabi
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES proposals/box3d_cuda_v2.h
  proposals/box3d_cuda_machine_coupling_v1.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/box3d_cuda)

add_executable(box3d_cuda_native_smoke examples/native_smoke.cu)
target_link_libraries(box3d_cuda_native_smoke PRIVATE box3d_cuda_cabi)

add_executable(box3d_cuda_topology_digest_smoke examples/topology_digest_smoke.cpp)
target_compile_features(box3d_cuda_topology_digest_smoke PRIVATE cxx_std_17)

add_executable(box3d_cuda_native_scene_v2_smoke examples/native_scene_v2_smoke.cu)
target_link_libraries(box3d_cuda_native_scene_v2_smoke PRIVATE box3d_cuda_cabi)
set_target_properties(box3d_cuda_native_scene_v2_smoke PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES)

add_executable(box3d_cuda_native_machine_coupling_v1_smoke
  examples/native_machine_coupling_v1_smoke.cu)
target_link_libraries(box3d_cuda_native_machine_coupling_v1_smoke
  PRIVATE box3d_cuda_cabi)
set_target_properties(box3d_cuda_native_machine_coupling_v1_smoke PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES)
