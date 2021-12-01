; RUN: opt -passes=openmp-opt -pass-remarks=openmp-opt -pass-remarks-missed=openmp-opt -pass-remarks-analysis=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
target triple = "nvptx64"

; CHECK: remark: llvm/test/Transforms/OpenMP/custom_state_machines_remarks.c:10:1: Generic-mode kernel is executed with a customized state machine that requires a fallback.
; CHECK: remark: llvm/test/Transforms/OpenMP/custom_state_machines_remarks.c:12:5: Call may contain unknown parallel regions. Use `__attribute__((assume("omp_no_parallelism")))` to override.
; CHECK: remark: llvm/test/Transforms/OpenMP/custom_state_machines_remarks.c:14:5: Call may contain unknown parallel regions. Use `__attribute__((assume("omp_no_parallelism")))` to override.
; CHECK: remark: llvm/test/Transforms/OpenMP/custom_state_machines_remarks.c:20:1: Rewriting generic-mode kernel with a customized state machine.


;; void unknown(void);
;; void known(void) {
;;   #pragma omp parallel
;;   {
;;     unknown();
;;   }
;; }
;;
;; void test_fallback(void) {
;;   #pragma omp target teams
;;   {
;;     unknown();
;;     known();
;;     unknown();
;;   }
;; }
;;
;; void no_openmp(void) __attribute__((assume("omp_no_openmp")));
;; void test_no_fallback(void) {
;;   #pragma omp target teams
;;   {
;;     known();
;;     known();
;;     known();
;;     no_openmp();     // make it non-spmd
;;   }
;; }

%struct.ident_t = type { i32, i32, i32, i32, i8* }
%struct.anon = type {}
%struct.anon.1 = type {}
%struct.anon.0 = type {}

@0 = private unnamed_addr constant [87 x i8] c";custom_state_machines_remarks.c;__omp_offloading_4d_155eab7e_test_fallback_l10;10;1;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([87 x i8], [87 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant [54 x i8] c";custom_state_machines_remarks.c;test_fallback;10;1;;\00", align 1
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @2, i32 0, i32 0) }, align 8
@4 = private unnamed_addr constant [88 x i8] c";custom_state_machines_remarks.c;__omp_offloading_4d_155eab7e_test_fallback_l10;10;25;;\00", align 1
@5 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([88 x i8], [88 x i8]* @4, i32 0, i32 0) }, align 8
@__omp_offloading_4d_155eab7e_test_fallback_l10_exec_mode = weak constant i8 1
@6 = private unnamed_addr constant [90 x i8] c";custom_state_machines_remarks.c;__omp_offloading_4d_155eab7e_test_no_fallback_l20;20;1;;\00", align 1
@7 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([90 x i8], [90 x i8]* @6, i32 0, i32 0) }, align 8
@8 = private unnamed_addr constant [57 x i8] c";custom_state_machines_remarks.c;test_no_fallback;20;1;;\00", align 1
@9 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([57 x i8], [57 x i8]* @8, i32 0, i32 0) }, align 8
@10 = private unnamed_addr constant [91 x i8] c";custom_state_machines_remarks.c;__omp_offloading_4d_155eab7e_test_no_fallback_l20;20;25;;\00", align 1
@11 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([91 x i8], [91 x i8]* @10, i32 0, i32 0) }, align 8
@__omp_offloading_4d_155eab7e_test_no_fallback_l20_exec_mode = weak constant i8 1
@12 = private unnamed_addr constant [45 x i8] c";custom_state_machines_remarks.c;known;3;1;;\00", align 1
@13 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 2, i32 0, i8* getelementptr inbounds ([45 x i8], [45 x i8]* @12, i32 0, i32 0) }, align 8
@llvm.compiler.used = appending global [2 x i8*] [i8* @__omp_offloading_4d_155eab7e_test_fallback_l10_exec_mode, i8* @__omp_offloading_4d_155eab7e_test_no_fallback_l20_exec_mode], section "llvm.metadata"

; Function Attrs: convergent noinline norecurse nounwind
define weak void @__omp_offloading_4d_155eab7e_test_fallback_l10() #0 !dbg !14 {
entry:
  %omp.outlined.arg.agg. = alloca %struct.anon, align 1
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* @1, i8 1, i1 true, i1 true), !dbg !17
  %exec_user_code = icmp eq i32 %0, -1, !dbg !17
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit, !dbg !17

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @3)
  store i32 0, i32* %.zero.addr, align 4, !dbg !17
  store i32 %1, i32* %.threadid_temp., align 4, !dbg !17
  call void @__omp_outlined__(i32* %.threadid_temp., i32* %.zero.addr, %struct.anon* %omp.outlined.arg.agg.) #3, !dbg !17
  call void @__kmpc_target_deinit(%struct.ident_t* @5, i8 1, i1 true), !dbg !18
  ret void, !dbg !19

worker.exit:                                      ; preds = %entry
  ret void, !dbg !17
}

declare i32 @__kmpc_target_init(%struct.ident_t*, i8, i1, i1)

; Function Attrs: convergent noinline norecurse nounwind
define internal void @__omp_outlined__(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon* noalias %__context) #0 !dbg !20 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon* %__context, %struct.anon** %__context.addr, align 8
  %0 = load %struct.anon*, %struct.anon** %__context.addr, align 8, !dbg !21
  call void @unknown() #6, !dbg !22
  call void @known() #6, !dbg !23
  call void @unknown() #6, !dbg !24
  ret void, !dbg !25
}

; Function Attrs: convergent
declare void @unknown() #1

; Function Attrs: convergent noinline nounwind
define hidden void @known() #2 !dbg !26 {
entry:
  %omp.outlined.arg.agg. = alloca %struct.anon.1, align 1
  %0 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @13)
  call void @__kmpc_parallel_51(%struct.ident_t* @13, i32 %0, i32 1, i32 -1, i32 -1, i8* bitcast (void (i32*, i32*, %struct.anon.1*)* @__omp_outlined__2 to i8*), i8* bitcast (void (i16, i32)* @__omp_outlined__2_wrapper to i8*), i8* null), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%struct.ident_t*) #3

declare void @__kmpc_target_deinit(%struct.ident_t*, i8, i1)

; Function Attrs: convergent noinline norecurse nounwind
define weak void @__omp_offloading_4d_155eab7e_test_no_fallback_l20() #0 !dbg !29 {
entry:
  %omp.outlined.arg.agg. = alloca %struct.anon.0, align 1
  %.zero.addr = alloca i32, align 4
  %.threadid_temp. = alloca i32, align 4
  %0 = call i32 @__kmpc_target_init(%struct.ident_t* @7, i8 1, i1 true, i1 true), !dbg !30
  %exec_user_code = icmp eq i32 %0, -1, !dbg !30
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit, !dbg !30

user_code.entry:                                  ; preds = %entry
  %1 = call i32 @__kmpc_global_thread_num(%struct.ident_t* @9)
  store i32 0, i32* %.zero.addr, align 4, !dbg !30
  store i32 %1, i32* %.threadid_temp., align 4, !dbg !30
  call void @__omp_outlined__1(i32* %.threadid_temp., i32* %.zero.addr, %struct.anon.0* %omp.outlined.arg.agg.) #3, !dbg !30
  call void @__kmpc_target_deinit(%struct.ident_t* @11, i8 1, i1 true), !dbg !31
  ret void, !dbg !32

worker.exit:                                      ; preds = %entry
  ret void, !dbg !30
}

; Function Attrs: convergent noinline norecurse nounwind
define internal void @__omp_outlined__1(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon.0* noalias %__context) #0 !dbg !33 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon.0*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon.0* %__context, %struct.anon.0** %__context.addr, align 8
  %0 = load %struct.anon.0*, %struct.anon.0** %__context.addr, align 8, !dbg !34
  call void @known() #6, !dbg !35
  call void @known() #6, !dbg !36
  call void @known() #6, !dbg !37
  call void @no_openmp() #7, !dbg !38
  ret void, !dbg !39
}

; Function Attrs: convergent
declare void @no_openmp() #4

; Function Attrs: convergent noinline norecurse nounwind
define internal void @__omp_outlined__2(i32* noalias %.global_tid., i32* noalias %.bound_tid., %struct.anon.1* noalias %__context) #0 !dbg !40 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %__context.addr = alloca %struct.anon.1*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  store %struct.anon.1* %__context, %struct.anon.1** %__context.addr, align 8
  %0 = load %struct.anon.1*, %struct.anon.1** %__context.addr, align 8, !dbg !41
  call void @unknown() #6, !dbg !42
  ret void, !dbg !43
}

; Function Attrs: convergent noinline norecurse nounwind
define internal void @__omp_outlined__2_wrapper(i16 zeroext %0, i32 %1) #0 !dbg !44 {
entry:
  %.addr = alloca i16, align 2
  %.addr1 = alloca i32, align 4
  %.zero.addr = alloca i32, align 4
  %global_args = alloca i8*, align 8
  store i16 %0, i16* %.addr, align 2
  store i32 %1, i32* %.addr1, align 4
  store i32 0, i32* %.zero.addr, align 4, !dbg !45
  call void @__kmpc_get_shared_variables_aggregate(i8** %global_args), !dbg !45
  call void @__omp_outlined__2(i32* %.addr1, i32* %.zero.addr, %struct.anon.1* null) #3, !dbg !45
  ret void, !dbg !45
}

declare void @__kmpc_get_shared_variables_aggregate(i8**)

; Function Attrs: alwaysinline
declare void @__kmpc_parallel_51(%struct.ident_t*, i32, i32, i32, i32, i8*, i8*, i8*) #5

attributes #0 = { convergent noinline norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx32,+sm_70" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx32,+sm_70" }
attributes #2 = { convergent noinline nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx32,+sm_70" }
attributes #3 = { nounwind }
attributes #4 = { convergent "frame-pointer"="all" "llvm.assume"="omp_no_openmp" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx32,+sm_70" }
attributes #5 = { alwaysinline }
attributes #6 = { convergent }
attributes #7 = { convergent "llvm.assume"="omp_no_openmp" }

!llvm.dbg.cu = !{!0}
!omp_offload.info = !{!2, !3}
!nvvm.annotations = !{!4, !5}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "llvm/test/Transforms/OpenMP/custom_state_machines_remarks.c", directory: "llvm-project/llvm/test/Transforms/OpenMP")
!2 = !{i32 0, i32 77, i32 358525822, !"test_no_fallback", i32 20, i32 1}
!3 = !{i32 0, i32 77, i32 358525822, !"test_fallback", i32 10, i32 0}
!4 = !{void ()* @__omp_offloading_4d_155eab7e_test_fallback_l10, !"kernel", i32 1}
!5 = !{void ()* @__omp_offloading_4d_155eab7e_test_no_fallback_l20, !"kernel", i32 1}
!6 = !{i32 7, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"openmp", i32 50}
!10 = !{i32 7, !"openmp-device", i32 50}
!11 = !{i32 7, !"PIC Level", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 14.0.0 (git@github.com:ggeorgakoudis/llvm-project.git 8625a61ab71af1f605ce972359656eefab8ffbb8)"}
!14 = distinct !DISubprogram(name: "__omp_offloading_4d_155eab7e_test_fallback_l10", scope: !1, file: !1, line: 10, type: !15, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !16)
!15 = !DISubroutineType(types: !16)
!16 = !{}
!17 = !DILocation(line: 10, column: 1, scope: !14)
!18 = !DILocation(line: 10, column: 25, scope: !14)
!19 = !DILocation(line: 15, column: 3, scope: !14)
!20 = distinct !DISubprogram(name: "__omp_outlined__", scope: !1, file: !1, line: 10, type: !15, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !16)
!21 = !DILocation(line: 10, column: 1, scope: !20)
!22 = !DILocation(line: 12, column: 5, scope: !20)
!23 = !DILocation(line: 13, column: 5, scope: !20)
!24 = !DILocation(line: 14, column: 5, scope: !20)
!25 = !DILocation(line: 15, column: 3, scope: !20)
!26 = distinct !DISubprogram(name: "known", scope: !1, file: !1, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!27 = !DILocation(line: 3, column: 1, scope: !26)
!28 = !DILocation(line: 7, column: 1, scope: !26)
!29 = distinct !DISubprogram(name: "__omp_offloading_4d_155eab7e_test_no_fallback_l20", scope: !1, file: !1, line: 20, type: !15, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !16)
!30 = !DILocation(line: 20, column: 1, scope: !29)
!31 = !DILocation(line: 20, column: 25, scope: !29)
!32 = !DILocation(line: 26, column: 3, scope: !29)
!33 = distinct !DISubprogram(name: "__omp_outlined__1", scope: !1, file: !1, line: 20, type: !15, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !16)
!34 = !DILocation(line: 20, column: 1, scope: !33)
!35 = !DILocation(line: 22, column: 5, scope: !33)
!36 = !DILocation(line: 23, column: 5, scope: !33)
!37 = !DILocation(line: 24, column: 5, scope: !33)
!38 = !DILocation(line: 25, column: 5, scope: !33)
!39 = !DILocation(line: 26, column: 3, scope: !33)
!40 = distinct !DISubprogram(name: "__omp_outlined__2", scope: !1, file: !1, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !16)
!41 = !DILocation(line: 3, column: 1, scope: !40)
!42 = !DILocation(line: 5, column: 5, scope: !40)
!43 = !DILocation(line: 6, column: 3, scope: !40)
!44 = distinct !DISubprogram(linkageName: "__omp_outlined__2_wrapper", scope: !1, file: !1, line: 3, type: !15, scopeLine: 3, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !16)
!45 = !DILocation(line: 3, column: 1, scope: !44)
