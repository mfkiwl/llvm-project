//===----- data_sharing.cu - OpenMP GPU data sharing ------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of data sharing environments
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/omptarget.h"
#include "target/shuffle.h"
#include "target_impl.h"

////////////////////////////////////////////////////////////////////////////////
// Runtime functions for trunk data sharing scheme.
////////////////////////////////////////////////////////////////////////////////

static constexpr unsigned MinBytes = 8;

static constexpr unsigned Alignment = 8;

/// External symbol to access dynamic shared memory.
extern unsigned char DynamicSharedBuffer[] __attribute__((aligned(Alignment)));
#pragma omp allocate(DynamicSharedBuffer) allocator(omp_pteam_mem_alloc)

EXTERN void *__kmpc_get_dynamic_shared() { return DynamicSharedBuffer; }

EXTERN void *llvm_omp_get_dynamic_shared() {
  return __kmpc_get_dynamic_shared();
}

template <unsigned BPerThread, unsigned NThreads = MAX_THREADS_PER_TEAM>
struct alignas(32) ThreadStackTy {
  static constexpr unsigned BytesPerThread = BPerThread;
  static constexpr unsigned NumThreads = NThreads;
  static constexpr unsigned NumWarps = (NThreads + WARPSIZE - 1) / WARPSIZE;

  unsigned char Data[NumThreads][BytesPerThread];
  unsigned char Usage[NumThreads];
};

[[clang::loader_uninitialized]] ThreadStackTy<MinBytes * 8, 1> MainSharedStack;
#pragma omp allocate(MainSharedStack) allocator(omp_pteam_mem_alloc)

[[clang::loader_uninitialized]] ThreadStackTy<MinBytes,
                                              MAX_THREADS_PER_TEAM / 4>
    WorkerSharedStack;
#pragma omp allocate(WorkerSharedStack) allocator(omp_pteam_mem_alloc)

EXTERN void *__kmpc_alloc_shared(size_t Bytes) {
  size_t AlignedBytes = Bytes + (Bytes % MinBytes);
  int TID = __kmpc_get_hardware_thread_id_in_block();
  if (__kmpc_is_generic_main_thread(TID)) {
    // Main thread alone, use shared memory if space is available.
    if (MainSharedStack.Usage[0] + AlignedBytes <= MainSharedStack.BytesPerThread) {
      void *Ptr = &MainSharedStack.Data[0][MainSharedStack.Usage[0]];
      MainSharedStack.Usage[0] += AlignedBytes;
      return Ptr;
    }
  } else if (TID < WorkerSharedStack.NumThreads) {
    if (WorkerSharedStack.Usage[TID] + AlignedBytes <= WorkerSharedStack.BytesPerThread) {
      void *Ptr = &WorkerSharedStack.Data[TID][WorkerSharedStack.Usage[TID]];
      WorkerSharedStack.Usage[TID] += AlignedBytes;
      return Ptr;
    }
  }
  // Fallback to malloc
  return SafeMalloc(Bytes, "AllocGlobalFallback");
}

EXTERN void __kmpc_free_shared(void *Ptr, size_t Bytes) {
  size_t AlignedBytes = Bytes + (Bytes % MinBytes);
  int TID = __kmpc_get_hardware_thread_id_in_block();
  if (__kmpc_is_generic_main_thread(TID)) {
    if (Ptr >= &MainSharedStack.Data[0][0] &&
        Ptr < &MainSharedStack.Data[MainSharedStack.NumThreads][0]) {
      MainSharedStack.Usage[0] -= AlignedBytes;
      return;
    }
  } else if (TID < WorkerSharedStack.NumThreads) {
    if (Ptr >= &WorkerSharedStack.Data[0][0] &&
        Ptr < &WorkerSharedStack.Data[WorkerSharedStack.NumThreads][0]) {
      int TID = __kmpc_get_hardware_thread_id_in_block();
      WorkerSharedStack.Usage[TID] -= AlignedBytes;
      return;
    }
  }
  SafeFree(Ptr, "FreeGlobalFallback");
}

EXTERN void *__kmpc_alloc_aggregate_arg(void *LocalPtr, void *GlobalPtr) {
  return (__kmpc_is_spmd_exec_mode() ? LocalPtr : GlobalPtr);
}

EXTERN void __kmpc_data_sharing_init_stack() {
  for (unsigned i = 0; i < MainSharedStack.NumWarps; ++i)
    MainSharedStack.Usage[i] = 0;
  for (unsigned i = 0; i < WorkerSharedStack.NumThreads; ++i)
    WorkerSharedStack.Usage[i] = 0;
}

/// Allocate storage in shared memory to communicate arguments, stored in the
/// aggregating struct, from the main thread to the workers in generic mode.
[[clang::loader_uninitialized]] static void *SharedMemAggregatePtr[1];
#pragma omp allocate(SharedMemAggregatePtr) \
    allocator(omp_pteam_mem_alloc)

// This function will return the pointer to the struct that aggregates shared
// variables to pass to the outlined parallel function.
// Called by all workers.
EXTERN void __kmpc_get_shared_variables_aggregate(void **GlobalArgs) {
  *GlobalArgs = SharedMemAggregatePtr[0];
}

// This function sets the aggregate struct of shared variables for the team,
// by storing the pointer of the aggregate, passed as an argument from the main
// thread, to the shared memory storage.
EXTERN void __kmpc_set_shared_variables_aggregate(void *args) {
  SharedMemAggregatePtr[0] = args;
}

// This function is used to init static memory manager. This manager is used to
// manage statically allocated global memory. This memory is allocated by the
// compiler and used to correctly implement globalization of the variables in
// target, teams and distribute regions.
EXTERN void __kmpc_get_team_static_memory(int16_t isSPMDExecutionMode,
                                          const void *buf, size_t size,
                                          int16_t is_shared,
                                          const void **frame) {
  if (is_shared) {
    *frame = buf;
    return;
  }
  if (isSPMDExecutionMode) {
    if (__kmpc_get_hardware_thread_id_in_block() == 0) {
      *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
    }
    __kmpc_impl_syncthreads();
    return;
  }
  ASSERT0(LT_FUSSY,
          __kmpc_get_hardware_thread_id_in_block() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  *frame = omptarget_nvptx_simpleMemoryManager.Acquire(buf, size);
  __kmpc_impl_threadfence();
}

EXTERN void __kmpc_restore_team_static_memory(int16_t isSPMDExecutionMode,
                                              int16_t is_shared) {
  if (is_shared)
    return;
  if (isSPMDExecutionMode) {
    __kmpc_impl_syncthreads();
    if (__kmpc_get_hardware_thread_id_in_block() == 0) {
      omptarget_nvptx_simpleMemoryManager.Release();
    }
    return;
  }
  __kmpc_impl_threadfence();
  ASSERT0(LT_FUSSY,
          __kmpc_get_hardware_thread_id_in_block() == GetMasterThreadID(),
          "Must be called only in the target master thread.");
  omptarget_nvptx_simpleMemoryManager.Release();
}

#pragma omp end declare target
