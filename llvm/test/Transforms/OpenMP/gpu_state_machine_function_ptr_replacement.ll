; RUN: opt -S -passes=openmp-opt -openmp-ir-builder-optimistic-attributes -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s
; RUN: opt -S -passes=openmp-opt -pass-remarks=openmp-opt -openmp-print-gpu-kernels < %s | FileCheck %s

; C input used for this test:

; void bar(void) {
;     #pragma omp parallel
;     { }
; }
; void foo(void) {
;   #pragma omp target teams
;   {
;     #pragma omp parallel
;     {}
;     bar();
;     unknown();
;     #pragma omp parallel
;     {}
;   }
; }

; Verify we replace the function pointer uses for the first and last outlined
; region (1 and 3) but not for the middle one (2) because it could be called from
; another kernel.

; CHECK-DAG: @__omp_outlined__1_wrapper.ID = private constant i8 undef
; CHECK-DAG: @__omp_outlined__2_wrapper.ID = private constant i8 undef

; CHECK-DAG:   icmp eq void (i16, i32)* %worker.work_fn.addr_cast, bitcast (i8* @__omp_outlined__1_wrapper.ID to void (i16, i32)*)
; CHECK-DAG:   icmp eq void (i16, i32)* %worker.work_fn.addr_cast, bitcast (i8* @__omp_outlined__2_wrapper.ID to void (i16, i32)*)


; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* noundef @1, i32 %{{.*}}, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*, %struct.anon{{[\.0-9]*}}*)* @__omp_outlined__1 to i8*), i8* noundef @__omp_outlined__1_wrapper.ID, i8* noundef null)
; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* noundef @1, i32 %{{.*}}, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*, %struct.anon{{[\.0-9]*}}*)* @__omp_outlined__2 to i8*), i8* noundef @__omp_outlined__2_wrapper.ID, i8* noundef null)
; CHECK-DAG:   call void @__kmpc_parallel_51(%struct.ident_t* noundef @2, i32 %{{.*}}, i32 noundef 1, i32 noundef -1, i32 noundef -1, i8* noundef bitcast (void (i32*, i32*, %struct.anon{{[\.0-9]*}}*)* @__omp_outlined__3 to i8*), i8* noundef bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*), i8* noundef null)

%struct.ident_t = type { i32, i32, i32, i32, i8* }
%struct.anon = type {}
%struct.anon.0 = type {}
%struct.anon.1 = type {}
%struct.anon.2 = type {}

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@__omp_offloading_4d_953a6a3c_foo_l6_exec_mode = weak constant i8 1
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 2, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@llvm.compiler.used = appending global [1 x i8*] [i8* @__omp_offloading_4d_953a6a3c_foo_l6_exec_mode], section "llvm.metadata"

define weak void @__omp_offloading_4d_953a6a3c_foo_l6() #0 {
entry:
  %omp.outlined.arg.agg. = alloca %struct.anon, align 1
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* @1, i8 1, i1 true, i1 true)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @1)
  store i32 0, i32* %.zero.addr, align 4
  store i32 %1, i32* %.threadid_temp., align 4
  call void @__omp_outlined__(i32* %.threadid_temp., i32* %.zero.addr, %struct.anon* %omp.outlined.arg.agg.) #4
  call void @__kmpc_target_deinit(%struct.ident_t* @1, i8 1, i1 true)
  ret void

worker.exit:                                      ; preds = %entry
  ret void
}

declare i32 @__kmpc_target_init(%struct.ident_t*, i8, i1, i1)

define internal void @__omp_outlined__(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon* noalias %__context) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon*, align 8
  %omp.outlined.arg.agg. = alloca %struct.anon.0, align 1
  %omp.outlined.arg.agg.1 = alloca %struct.anon.1, align 1
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon* %__context, %struct.anon** %__context.addr, align 8
  %0 = load %struct.anon*, %struct.anon** %__context.addr, align 8
  %1 = load i32*, i32** %.global_tid..addr, align 8
  %2 = load i32, i32* %1, align 4
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %2, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*, %struct.anon.0*)* @__omp_outlined__1 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__1_wrapper to i8*), i8* null)
  call void @bar() #5
  %call = call i32 bitcast (i32 (...)* @unknown to i32 ()*)() #5
  call void @__kmpc_parallel_51(%struct.ident_t* @1, i32 %2, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*, %struct.anon.1*)* @__omp_outlined__2 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8* null)
  ret void
}

define internal void @__omp_outlined__1(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon.0* noalias %__context) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon.0*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon.0* %__context, %struct.anon.0** %__context.addr, align 8
  %0 = load %struct.anon.0*, %struct.anon.0** %__context.addr, align 8
  ret void
}

define internal void @__omp_outlined__1_wrapper(i16 zeroext %0, i32 %1) #0 {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8*, align 8
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  store i32 0, i32* %.zero.addr, align 4
  call void @__kmpc_get_shared_variables_aggregate(i8** %global_args)
  call void @__omp_outlined__1(i32* %.addr1, i32* %.zero.addr, %struct.anon.0* null) #4
  ret void
}

declare void @__kmpc_get_shared_variables_aggregate(i8**)

declare void @__kmpc_parallel_51(%struct.ident_t*, i32, i32, i32, i32, i8*, i8*, i8*) #1

define hidden void @bar() #2 {
entry:
  %omp.outlined.arg.agg. = alloca %struct.anon.2, align 1
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @2)
  call void @__kmpc_parallel_51(%struct.ident_t* @2, i32 %0, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*, %struct.anon.2*)* @__omp_outlined__3 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__3_wrapper to i8*), i8* null)
  ret void
}

declare i32 @unknown(...) #3

define internal void @__omp_outlined__2(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon.1* noalias %__context) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon.1*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon.1* %__context, %struct.anon.1** %__context.addr, align 8
  %0 = load %struct.anon.1*, %struct.anon.1** %__context.addr, align 8
  ret void
}

define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1) #0 {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8*, align 8
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  store i32 0, i32* %.zero.addr, align 4
  call void @__kmpc_get_shared_variables_aggregate(i8** %global_args)
  call void @__omp_outlined__2(i32* %.addr1, i32* %.zero.addr, %struct.anon.1* null) #4
  ret void
}

declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #4

declare void @__kmpc_target_deinit(%struct.ident_t*, i8, i1)

define internal void @__omp_outlined__3(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon.2* noalias %__context) #0 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon.2*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon.2* %__context, %struct.anon.2** %__context.addr, align 8
  %0 = load %struct.anon.2*, %struct.anon.2** %__context.addr, align 8
  ret void
}

define internal void @__omp_outlined__3_wrapper(i16 zeroext %0, i32 %1) #0 {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8*, align 8
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  store i32 0, i32* %.zero.addr, align 4
  call void @__kmpc_get_shared_variables_aggregate(i8** %global_args)
  call void @__omp_outlined__3(i32* %.addr1, i32* %.zero.addr, %struct.anon.2* null) #4
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" }
attributes #1 = { alwaysinline }
attributes #2 = { convergent noinline nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" }
attributes #3 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" }
attributes #4 = { nounwind }
attributes #5 = { convergent }

!omp_offload.info = !{!0}
!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3, !4}

!0 = !{i32 0, i32 77, i32 -1791333828, !"foo", i32 6, i32 0}
!1 = !{void ()* @__omp_offloading_4d_953a6a3c_foo_l6, !"kernel", i32 1}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"openmp", i32 50}
!4 = !{i32 7, !"openmp-device", i32 50}

