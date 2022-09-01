//===- IntrinsicsOpenMP.cpp - Codegen OpenMP from IR intrinsics ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code generation for OpenMP from intrinsics embedded in
// the IR, using the OpenMPIRBuilder
//
//===-------------------------------------------------------------------------===//

#include "llvm-c/Transforms/IntrinsicsOpenMP.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IntrinsicsOpenMP/IntrinsicsOpenMP.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

  // TODO: explose clauses through namespace omp?
  enum DSAType {
    DSA_PRIVATE,
    DSA_FIRSTPRIVATE,
    DSA_SHARED,
    DSA_REDUCTION_ADD
  };

  static const DenseMap<StringRef, Directive> StringToDir = {
      {"DIR.OMP.PARALLEL", OMPD_parallel},
      {"DIR.OMP.SINGLE", OMPD_single},
      {"DIR.OMP.CRITICAL", OMPD_critical},
      {"DIR.OMP.BARRIER", OMPD_barrier},
      {"DIR.OMP.LOOP", OMPD_for},
      {"DIR.OMP.PARALLEL.LOOP", OMPD_parallel_for},
      {"DIR.OMP.TASK", OMPD_task},
      {"DIR.OMP.TASKWAIT", OMPD_taskwait},
      {"DIR.OMP.TARGET", OMPD_target}
  };

  // TODO: add more reduction operators.
  static const DenseMap<StringRef, DSAType> StringToDSA = {
      {"QUAL.OMP.PRIVATE", DSA_PRIVATE},
      {"QUAL.OMP.FIRSTPRIVATE", DSA_FIRSTPRIVATE},
      {"QUAL.OMP.SHARED", DSA_SHARED},
      {"QUAL.OMP.REDUCTION.ADD", DSA_REDUCTION_ADD},
  };

  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  using BodyGenCallbackTy = OpenMPIRBuilder::BodyGenCallbackTy;
  using FinalizeCallbackTy = OpenMPIRBuilder::FinalizeCallbackTy;

  struct CGReduction {
    static OpenMPIRBuilder::InsertPointTy
    sumReduction(OpenMPIRBuilder::InsertPointTy IP, Value *LHS, Value *RHS,
                 Value *&Result) {
      IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
      Type *VTy = RHS->getType();
      if (VTy->isIntegerTy())
        Result = Builder.CreateAdd(LHS, RHS, "red.add");
      else if (VTy->isFloatTy() || VTy->isDoubleTy())
        Result = Builder.CreateFAdd(LHS, RHS, "red.add");
      else
        assert(false && "Unsupported type for sumReduction");
      return Builder.saveIP();
    }

    static OpenMPIRBuilder::InsertPointTy
    sumAtomicReduction(OpenMPIRBuilder::InsertPointTy IP, Value *LHS,
                       Value *RHS) {
      IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
      Type *VTy = RHS->getType()->getPointerElementType();
      Value *Partial = Builder.CreateLoad(VTy, RHS, "red.partial");
      if (VTy->isIntegerTy())
        Builder.CreateAtomicRMW(AtomicRMWInst::Add, LHS, Partial, None,
                                AtomicOrdering::Monotonic);
      else if (VTy->isFloatTy() || VTy->isDoubleTy())
        Builder.CreateAtomicRMW(AtomicRMWInst::FAdd, LHS, Partial, None,
                                AtomicOrdering::Monotonic);
      else
        assert(false && "Unsupported type for sumAtomicReduction");
      return Builder.saveIP();
    }
  };

struct Outliner {
      static Function*
      createOutlinedFunction(Module &M, OpenMPIRBuilder &OMPBuilder,
                             MapVector<Value *, DSAType> &DSAValueMap,
                             Function *OuterFn, BasicBlock *BBEntry, BasicBlock *StartBB,
                             BasicBlock *EndBB, BasicBlock *AfterBB,
                             SmallVectorImpl<llvm::Value *> &CapturedVars) {
        SmallVector<Value *, 16> Privates;
        SmallVector<Value *, 16> CapturedShared;
        SmallVector<Value *, 16> CapturedFirstprivate;
        for (auto &It : DSAValueMap) {
          Value *V = It.first;
          auto DSA = It.second;

          if (DSA_PRIVATE == DSA)
            Privates.push_back(V);
          else if (DSA_FIRSTPRIVATE == DSA)
            CapturedFirstprivate.push_back(V);
          else if (DSA_SHARED == DSA)
            CapturedShared.push_back(V);
          else
            assert(false && "Unsupported DSA type");
        }

        OpenMPIRBuilder::OutlineInfo OI;
        OI.EntryBB = StartBB;
        OI.ExitBB = EndBB;
        SmallPtrSet<BasicBlock *, 8> BlockSet;
        SmallVector<BasicBlock *, 8> BlockVector;
        OI.collectBlocks(BlockSet, BlockVector);

        SmallVector<Type *, 16> Params;
        // tid
        Params.push_back(OMPBuilder.Int32Ptr);
        // bound_tid
        Params.push_back(OMPBuilder.Int32Ptr);
        for(auto *V : CapturedShared)
          Params.push_back(V->getType());
        for(auto *V : CapturedFirstprivate)
          Params.push_back(V->getType());

        FunctionType *OutlinedFnTy =
            FunctionType::get(OMPBuilder.Void, Params, /* isVarArgs */ false);
        Function *OutlinedFn =
            Function::Create(OutlinedFnTy, GlobalValue::InternalLinkage,
                             OuterFn->getName() + ".omp_outlined", M);

        // Name the parameters.
        OutlinedFn->arg_begin()->setName("global_tid");
        std::next(OutlinedFn->arg_begin())->setName("bound_tid");
        Function::arg_iterator AI = std::next(OutlinedFn->arg_begin(), 2);
        int num_arg = 2;
        for(auto *V : CapturedShared) {
          AI->setName(V->getName() + ".shared");
          OutlinedFn->addParamAttr(num_arg, Attribute::NonNull);
          OutlinedFn->addParamAttr(num_arg, Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
          ++AI;
          ++num_arg;
        }
        for (auto *V : CapturedFirstprivate) {
          AI->setName(V->getName() + ".firstprivate");
          OutlinedFn->addParamAttr(num_arg, Attribute::NonNull);
          OutlinedFn->addParamAttr(num_arg, Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
          ++AI;
        }

        BasicBlock *OutlinedEntryBB =
            BasicBlock::Create(M.getContext(), ".outlined.entry", OutlinedFn);
        BasicBlock *OutlinedExitBB = BasicBlock::Create(M.getContext(), ".outlined.exit", OutlinedFn);
        OMPBuilder.Builder.SetInsertPoint(OutlinedEntryBB);

        OutlinedFn->addParamAttr(0, Attribute::NoAlias);
        OutlinedFn->addParamAttr(1, Attribute::NoAlias);
        OutlinedFn->addFnAttr(Attribute::NoUnwind);
        OutlinedFn->addFnAttr(Attribute::NoRecurse);

        auto CollectUses = [&BlockSet](Value *V, SetVector<Use *> &Uses) {
          for (Use &U : V->uses())
            if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
              if (BlockSet.count(UserI->getParent()))
                Uses.insert(&U);
        };

        auto ReplaceUses = [](SetVector<Use *> &Uses, Value *ReplacementValue) {
          for (Use *UPtr : Uses)
            UPtr->set(ReplacementValue);
        };

        for (auto *V : Privates) {
          SetVector<Use *> Uses;
          CollectUses(V, Uses);

          Type *VTy = V->getType()->getPointerElementType();
          Value *ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, nullptr, V->getName() + ".private");

          ReplaceUses(Uses, ReplacementValue);
        }

        AI = std::next(OutlinedFn->arg_begin(), 2);
        for (auto *V : CapturedShared) {
          SetVector<Use *> Uses;
          CollectUses(V, Uses);

          Value *ReplacementValue = AI;

          ReplaceUses(Uses, ReplacementValue);
          ++AI;
        }

        for (auto *V : CapturedFirstprivate) {
          SetVector<Use *> Uses;
          CollectUses(V, Uses);

          Type *VTy = V->getType()->getPointerElementType();
          Value *ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, nullptr, V->getName() + ".copy");
          Value *Load = OMPBuilder.Builder.CreateLoad(VTy, AI, V->getName() + ".reload");
          OMPBuilder.Builder.CreateStore(Load, ReplacementValue);

          ReplaceUses(Uses, ReplacementValue);

          ++AI;
        }

        OMPBuilder.Builder.CreateBr(StartBB);

        EndBB->getTerminator()->setSuccessor(0, OutlinedExitBB);
        OMPBuilder.Builder.SetInsertPoint(OutlinedExitBB);
        OMPBuilder.Builder.CreateRetVoid();

        for(auto *BB : BlockSet)
          BB->moveAfter(&OutlinedFn->getEntryBlock());

        dbgs() << "=== Dump OutlinedFn\n" << *OutlinedFn << "=== End of Dump OutlinedFn\n";

        /*
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
        BBEntry->getTerminator()->eraseFromParent();
        OpenMPIRBuilder::LocationDescription Loc(
            InsertPointTy(BBEntry, BBEntry->end()), DL);


        dbgs() << "=== BEFORE Dump OuterFn\n" << *OuterFn << "=== End of Dump OuterFn\n";

        Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
        OMPBuilder.Builder.restoreIP(Loc.IP);
        OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

        Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr);
        //Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

        auto *OutlinedFnCast = OMPBuilder.Builder.CreateBitCast(
            OutlinedFn, OMPBuilder.ParallelTaskPtr);
        FunctionCallee ForkCall = OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_fork_call);
        SmallVector<Value *, 16> ForkArgs;
        ForkArgs.append(
            {Ident,
             OMPBuilder.Builder.getInt32(CapturedShared.size() +
                                         CapturedFirstprivate.size()),
             OutlinedFnCast});
        ForkArgs.append(CapturedShared);
        ForkArgs.append(CapturedFirstprivate);

        OMPBuilder.Builder.CreateCall(ForkCall, ForkArgs);
        OMPBuilder.Builder.CreateBr(AfterBB);

        dbgs() << "=== Dump OuterFn\n" << *OuterFn << "=== End of Dump OuterFn\n";
        */

        if (verifyFunction(*OutlinedFn, &errs()))
          report_fatal_error("Verification of OutlinedFn failed!");

        CapturedVars.append(CapturedShared);
        CapturedVars.append(CapturedFirstprivate);
        return OutlinedFn;
      }

    };

  struct IntrinsicsOpenMP: public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    IntrinsicsOpenMP() : ModulePass(ID) {}

    static void emitOMPParallel(OpenMPIRBuilder &OMPBuilder,
                                MapVector<Value *, DSAType> &DSAValueMap,
                                const DebugLoc &DL, Function *Fn,
                                BasicBlock *BBEntry, BasicBlock *StartBB,
                                BasicBlock *EndBB, BasicBlock *AfterBB,
                                FinalizeCallbackTy FiniCB, Value *IfCondition,
                                Value *NumThreads) {
      InsertPointTy BodyIP, BodyAllocaIP;
      SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

      auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                        Value &Orig, Value &Inner,
                        Value *&ReplacementValue) -> InsertPointTy {
        auto It = DSAValueMap.find(&Orig);
        LLVM_DEBUG(dbgs() << "DSAValueMap for Orig " << Orig << " Inner " << Inner);
        if (It != DSAValueMap.end())
          LLVM_DEBUG(dbgs() << It->second);
        else
          LLVM_DEBUG(dbgs() << " (null)!");
        LLVM_DEBUG(dbgs() << "\n ");

        assert(It != DSAValueMap.end() && "Expected Value in DSAValueMap");

        DSAType DSA = It->second;

        if (DSA == DSA_PRIVATE) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, /*ArraySize */ nullptr, Inner.getName());
          LLVM_DEBUG(dbgs() << "Privatizing Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n");
        } else if (DSA == DSA_FIRSTPRIVATE) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          Value *V = OMPBuilder.Builder.CreateLoad(VTy, &Inner,
                                                   Orig.getName() + ".reload");
          ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, /*ArraySize */ nullptr, Orig.getName() + ".copy");
          OMPBuilder.Builder.restoreIP(CodeGenIP);
          OMPBuilder.Builder.CreateStore(V, ReplacementValue);
          LLVM_DEBUG(dbgs() << "Firstprivatizing Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n");
        } else if (DSA == DSA_REDUCTION_ADD) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          Value *V = OMPBuilder.Builder.CreateAlloca(
              VTy, /* ArraySize */ nullptr, Orig.getName() + ".red.priv");
          ReplacementValue = V;

          OMPBuilder.Builder.restoreIP(CodeGenIP);
          // Store idempotent value based on operation and type.
          // TODO: create templated emitInitAndAppendInfo in CGReduction
          if (VTy->isIntegerTy())
            OMPBuilder.Builder.CreateStore(ConstantInt::get(VTy, 0), V);
          else if (VTy->isFloatTy() || VTy->isDoubleTy())
            OMPBuilder.Builder.CreateStore(ConstantFP::get(VTy, 0.0), V);
          else
            assert(false && "Unsupported type to init with idempotent reduction value");

          ReductionInfos.push_back({&Orig, V, CGReduction::sumReduction,
                                    CGReduction::sumAtomicReduction});

          return OMPBuilder.Builder.saveIP();
        } else {
          ReplacementValue = &Inner;
          LLVM_DEBUG(dbgs() << "Shared Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n");
        }

        return CodeGenIP;
      };

      auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                           BasicBlock &ContinuationIP) {
        BasicBlock *CGStartBB = CodeGenIP.getBlock();
        BasicBlock *CGEndBB = SplitBlock(CGStartBB, &*CodeGenIP.getPoint());
        assert(StartBB != nullptr && "StartBB should not be null");
        CGStartBB->getTerminator()->setSuccessor(0, StartBB);
        assert(EndBB != nullptr && "EndBB should not be null");
        EndBB->getTerminator()->setSuccessor(0, CGEndBB);

        BodyIP = InsertPointTy(CGEndBB, CGEndBB->getFirstInsertionPt());
        BodyAllocaIP = AllocaIP;
      };

      IRBuilder<>::InsertPoint AllocaIP(
          &Fn->getEntryBlock(), Fn->getEntryBlock().getFirstInsertionPt());

      // Set the insertion location at the end of the BBEntry.
      BBEntry->getTerminator()->eraseFromParent();

      Value *IfConditionCast = nullptr;
      if (IfCondition) {
        OMPBuilder.Builder.SetInsertPoint(BBEntry);
        IfConditionCast = OMPBuilder.Builder.CreateIntCast(
            IfCondition, OMPBuilder.Builder.getInt1Ty(), /* isSigned */ false);
      }

      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(BBEntry, BBEntry->end()), DL);

      // TODO: support cancellable, binding.
      InsertPointTy AfterIP = OMPBuilder.createParallel(
          Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
          /* IfCondition */ IfConditionCast, /* NumThreads */ NumThreads,
          OMP_PROC_BIND_default, /* IsCancellable */ false);

      if (!ReductionInfos.empty())
        OMPBuilder.createReductions(BodyIP, BodyAllocaIP, ReductionInfos);

      BranchInst::Create(AfterBB, AfterIP.getBlock());
    }

    static void emitOMPParallelDevice(Module &M, OpenMPIRBuilder &OMPBuilder,
                                      MapVector<Value *, DSAType> &DSAValueMap,
                                      const DebugLoc &DL, Function *OuterFn,
                                      BasicBlock *BBEntry, BasicBlock *StartBB,
                                      BasicBlock *EndBB, BasicBlock *AfterBB,
                                      Value *IfCondition, Value *NumThreads) {
      // Extract parallel region
#if 0
      CodeExtractorAnalysisCache CEAC(Fn);
      CodeExtractor Extractor(Blocks, /* DominatorTree */ nullptr,
                              /* AggregateArgs */ false,
                              /* BlockFrequencyInfo */ nullptr,
                              /* BranchProbabilityInfo */ nullptr,
                              /* AssumptionCache */ nullptr,
                              /* AllowVarArgs */ true,
                              /* AllowAlloca */ true,
                              /* Suffix */ ".omp_par_device");
#endif
      SmallVector<Value *, 16> CapturedVars;
      Function *OutlinedFn = Outliner::createOutlinedFunction(
          M, OMPBuilder, DSAValueMap, OuterFn, BBEntry, StartBB, EndBB, AfterBB,
          CapturedVars);

      // Create wrapper for worker threads
      SmallVector<Type *, 2> Params;
      // parallelism level, unused?
      Params.push_back(OMPBuilder.Int16);
      // tid
      Params.push_back(OMPBuilder.Int32);

      FunctionType *OutlinedWrapperFnTy =
          FunctionType::get(OMPBuilder.Void, Params, /* isVarArgs */ false);
      Function *OutlinedWrapperFn =
          Function::Create(OutlinedWrapperFnTy, GlobalValue::InternalLinkage,
                           OutlinedFn->getName() + ".wrapper", M);
      BasicBlock *OutlinedWrapperEntryBB =
          BasicBlock::Create(M.getContext(), "entry", OutlinedWrapperFn);

      // Code generation for the outlined wrapper function.
      OMPBuilder.Builder.SetInsertPoint(OutlinedWrapperEntryBB);

      constexpr const int TIDArgNo = 1;
      AllocaInst *TIDAddr = OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".tid.addr");
      AllocaInst *ZeroAddr = OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, "zero.addr");
      AllocaInst *GlobalArgs = OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int8PtrPtr, nullptr, "global_args");

      OMPBuilder.Builder.CreateStore(OutlinedWrapperFn->getArg(TIDArgNo), TIDAddr);
      OMPBuilder.Builder.CreateStore(Constant::getNullValue(OMPBuilder.Int32), ZeroAddr);
      FunctionCallee KmpcGetSharedVariables =
          OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_get_shared_variables);
      OMPBuilder.Builder.CreateCall(KmpcGetSharedVariables, {GlobalArgs});

      SmallVector<Value *, 16> OutlinedFnArgs;
      OutlinedFnArgs.push_back(TIDAddr);
      OutlinedFnArgs.push_back(ZeroAddr);
      #if 0
      for (size_t ArgNo = 2, GlobalArgNo = 0; ArgNo < OutlinedFn->arg_size();
           ++ArgNo, ++GlobalArgNo) {
        Value *LoadGlobalArgs =
            OMPBuilder.Builder.CreateLoad(OMPBuilder.Int8PtrPtr, GlobalArgs);
        Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP1_64(
            OMPBuilder.Int8Ptr, LoadGlobalArgs, GlobalArgNo);
        Value *Bitcast = OMPBuilder.Builder.CreateBitCast(
            GEP, OutlinedFn->getArg(ArgNo)->getType()->getPointerTo());
        Value *Load = OMPBuilder.Builder.CreateLoad(
            OutlinedFn->getArg(ArgNo)->getType(), Bitcast);
        OutlinedFnArgs.push_back(Load);
      }
      #endif
      #if 1
      for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
        dbgs() << "CapturedVar " << Idx << " " << *CapturedVars[Idx] << "\n";
        dbgs() << "OutlinedArg " << *OutlinedFn->getArg(Idx+2) << "\n";
        Value *LoadGlobalArgs =
            OMPBuilder.Builder.CreateLoad(OMPBuilder.Int8PtrPtr, GlobalArgs);
        Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP1_64(
            OMPBuilder.Int8Ptr, LoadGlobalArgs, Idx);
        Value *Bitcast = OMPBuilder.Builder.CreateBitCast(
            GEP, CapturedVars[Idx]->getType()->getPointerTo());
        Value *Load = OMPBuilder.Builder.CreateLoad(
            CapturedVars[Idx]->getType(), Bitcast);
        OutlinedFnArgs.push_back(Load);
      }
      #endif

      OMPBuilder.Builder.CreateCall(OutlinedFn->getFunctionType(), OutlinedFn,
                                    OutlinedFnArgs);
      OMPBuilder.Builder.CreateRetVoid();

      if (verifyFunction(*OutlinedWrapperFn, &errs()))
        report_fatal_error("Verification of OutlinedWrapperFn failed!");

      LLVM_DEBUG(dbgs() << "=== Dump OutlinedWrapper\n"
                        << *OutlinedWrapperFn
                        << "=== End of Dump OutlinedWrapper\n");

      // Setup the call to kmpc_parallel_51
      BBEntry->getTerminator()->eraseFromParent();
      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(BBEntry, BBEntry->end()), DL);

      Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
      OMPBuilder.Builder.restoreIP(Loc.IP);
      OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

      // Create the address table of the global data.
      // The number of outlined arguments without global_tid, bound_tid.
      Value *NumCapturedArgs = ConstantInt::get(OMPBuilder.SizeTy, CapturedVars.size());
      Type *CapturedVarsAddrsTy = ArrayType::get(OMPBuilder.Int8Ptr, CapturedVars.size());
      // TODO: move alloca to the entry block of the outer function.
      Value *CapturedVarsAddrs = OMPBuilder.Builder.CreateAlloca(
          CapturedVarsAddrsTy, nullptr, ".captured_var_addrs");
      for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
        LLVM_DEBUG(dbgs() << "CapturedVar " << Idx << " " << *CapturedVars[Idx] << "\n");
        Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP2_64(CapturedVarsAddrsTy, CapturedVarsAddrs, 0, Idx);
        Value *Bitcast = OMPBuilder.Builder.CreateBitCast(CapturedVars[Idx], OMPBuilder.Int8Ptr);
        OMPBuilder.Builder.CreateStore(Bitcast, GEP);
      }

      Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr);
      Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

      if (!IfCondition)
        IfCondition = Constant::getNullValue(OMPBuilder.Int32);

      if (!NumThreads)
        NumThreads = Constant::getNullValue(OMPBuilder.Int32);

      FunctionCallee KmpcParallel51 =
          OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_parallel_51);

      // Set proc_bind to -1 by default as it is unused.
      assert(Ident && "Expected non-null Ident");
      assert(ThreadID && "Expected non-null ThreadID");
      assert(IfCondition && "Expected non-null IfCondition");
      assert(NumThreads && "Expected non-null NumThreads");
      assert(OutlinedWrapperFn && "Expected non-null OutlinedWrapperFn");
      assert(CapturedVarsAddrs && "Expected non-null CapturedVarsAddrs");
      assert(NumCapturedArgs && "Expected non-null NumCapturedArgs");

      Value *ProcBind = OMPBuilder.Builder.getInt32(-1);
      Value *OutlinedFnBitcast = OMPBuilder.Builder.CreateBitCast(OutlinedFn, OMPBuilder.VoidPtr);
      Value *OutlinedWrapperFnBitcast = OMPBuilder.Builder.CreateBitCast(OutlinedWrapperFn, OMPBuilder.VoidPtr);
      Value *CapturedVarAddrsBitcast = OMPBuilder.Builder.CreateBitCast(CapturedVarsAddrs, OMPBuilder.VoidPtrPtr);
      OMPBuilder.Builder.CreateCall(
          KmpcParallel51, {Ident, ThreadID, IfCondition, NumThreads, ProcBind,
                           OutlinedFnBitcast, OutlinedWrapperFnBitcast,
                           CapturedVarAddrsBitcast, NumCapturedArgs});
      OMPBuilder.Builder.CreateBr(AfterBB);

      LLVM_DEBUG(dbgs() << "=== Dump OuterFn\n" << *OuterFn << "=== End of Dump OuterFn\n");

      if (verifyFunction(*OuterFn, &errs()))
        report_fatal_error("Verification of OuterFn failed!");
    }

    static void emitOMPFor(Module &M, OpenMPIRBuilder &OMPBuilder,
                           MapVector<Value *, DSAType> &DSAValueMap, Value *IV,
                           Value *UB, BasicBlock *PreHeader, BasicBlock *Exit,
                           OMPScheduleType Sched, Value *Chunk,
                           bool IsStandalone) {
      Type *IVTy = IV->getType()->getPointerElementType();
      SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

      auto GetKmpcForStaticInit = [&]() -> FunctionCallee {
        LLVM_DEBUG(dbgs() << "Type " << *IVTy << "\n");
        unsigned Bitwidth = IVTy->getIntegerBitWidth();
        LLVM_DEBUG(dbgs() << "Bitwidth " << Bitwidth << "\n");
        if (Bitwidth == 32)
          return OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_for_static_init_4u);
        if (Bitwidth == 64)
          return OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_for_static_init_8u);
        llvm_unreachable("unknown OpenMP loop iterator bitwidth");
      };

      FunctionCallee KmpcForStaticInit = GetKmpcForStaticInit();
      FunctionCallee KmpcForStaticFini = OMPBuilder.getOrCreateRuntimeFunction(
          M, OMPRTL___kmpc_for_static_fini);

      const DebugLoc DL = PreHeader->getTerminator()->getDebugLoc();
      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(PreHeader, PreHeader->getTerminator()->getIterator()),
          DL);
      Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
      Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr);

      // Create allocas for static init values.
      InsertPointTy AllocaIP(PreHeader, PreHeader->getFirstInsertionPt());
      Type *I32Type = Type::getInt32Ty(M.getContext());
      OMPBuilder.Builder.restoreIP(AllocaIP);
      Value *PLastIter =
          OMPBuilder.Builder.CreateAlloca(I32Type, nullptr, "omp_lastiter");
      Value *PLowerBound =
          OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp_lb");
      Value *PStride =
          OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp_stride");
      Value *PUpperBound =
          OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp_ub");

      OpenMPIRBuilder::OutlineInfo OI;
      OI.EntryBB = PreHeader;
      OI.ExitBB = Exit;
      SmallPtrSet<BasicBlock *, 8> BlockSet;
      SmallVector<BasicBlock *, 8> BlockVector;
      OI.collectBlocks(BlockSet, BlockVector);

      // Do privatization if standalone.
      // TODO: create PrivCBHelper and re-use PrivCB from emitOMPParallel.
      if (IsStandalone)
        for (auto &It : DSAValueMap) {
          Value *Orig = It.first;
          DSAType DSA = It.second;
          Value *ReplacementValue = nullptr;
          Type *VTy = Orig->getType()->getPointerElementType();

          if (DSA == DSA_SHARED)
            continue;

          // Store previous uses to set them to the ReplacementValue after
          // privatization codegen.
          SetVector<Use *> Uses;
          for (Use &U : Orig->uses())
            if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
              if (BlockSet.count(UserI->getParent()))
                Uses.insert(&U);

          OMPBuilder.Builder.restoreIP(AllocaIP);
          if (DSA == DSA_PRIVATE) {
            ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                VTy, /*ArraySize */ nullptr, Orig->getName() + ".for.priv");
          } else if (DSA == DSA_FIRSTPRIVATE) {
            Value *V = OMPBuilder.Builder.CreateLoad(
                VTy, Orig, Orig->getName() + ".for.firstpriv.reload");
            ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                VTy, /*ArraySize */ nullptr,
                Orig->getName() + ".for.firstpriv.copy");
            OMPBuilder.Builder.CreateStore(V, ReplacementValue);
            // ReplacementValue = Orig;
          } else if (DSA == DSA_REDUCTION_ADD) {
            ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                VTy, /* ArraySize */ nullptr, Orig->getName() + ".red.priv");

            // Store idempotent value based on operation and type.
            // TODO: create templated emitInitAndAppendInfo in CGReduction
            if (VTy->isIntegerTy())
              OMPBuilder.Builder.CreateStore(ConstantInt::get(VTy, 0),
                                             ReplacementValue);
            else if (VTy->isFloatTy() || VTy->isDoubleTy())
              OMPBuilder.Builder.CreateStore(ConstantFP::get(VTy, 0.0),
                                             ReplacementValue);
            else
              assert(
                  false &&
                  "Unsupported type to init with idempotent reduction value");

            ReductionInfos.push_back({Orig, ReplacementValue,
                                      CGReduction::sumReduction,
                                      CGReduction::sumAtomicReduction});
          } else
            assert(false && "Unsupported privatization");

          assert(ReplacementValue && "Expected non-null ReplacementValue");

          for (Use *UPtr : Uses)
            UPtr->set(ReplacementValue);
        }

      OMPBuilder.Builder.SetInsertPoint(PreHeader->getTerminator());

      // Store the initial normalized upper bound to PUpperBound.
      Value *LoadUB = OMPBuilder.Builder.CreateLoad(
          UB->getType()->getPointerElementType(), UB);
      OMPBuilder.Builder.CreateStore(LoadUB, PUpperBound);

      Constant *Zero = ConstantInt::get(IVTy, 0);
      Constant *One = ConstantInt::get(IVTy, 1);
      OMPBuilder.Builder.CreateStore(Zero, PLowerBound);
      OMPBuilder.Builder.CreateStore(One, PStride);

      // If Chunk is not specified (nullptr), default to one, complying with the
      // OpenMP specification.
      if (!Chunk)
        Chunk = One;
      Value *ChunkCast = OMPBuilder.Builder.CreateIntCast(Chunk, IVTy, /*isSigned*/ false);

      Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);

      // TODO: add more scheduling types.
      Constant *SchedulingType =
          ConstantInt::get(I32Type, static_cast<int>(Sched));

      LLVM_DEBUG(dbgs() << "=== SchedulingType " << *SchedulingType << "\n");
      LLVM_DEBUG(dbgs() << "=== PLowerBound " << *PLowerBound << "\n");
      LLVM_DEBUG(dbgs() << "=== PUpperBound " << *PUpperBound << "\n");
      LLVM_DEBUG(dbgs() << "=== PStride " << *PStride << "\n");
      LLVM_DEBUG(dbgs() << "=== Incr " << *One << "\n");
      LLVM_DEBUG(dbgs() << "=== ChunkCast " << *ChunkCast << "\n");
      OMPBuilder.Builder.CreateCall(
          KmpcForStaticInit, {SrcLoc, ThreadNum, SchedulingType, PLastIter,
                              PLowerBound, PUpperBound, PStride, One, ChunkCast});
      // Load returned upper bound to UB.
      Value *LoadPUpperBound = OMPBuilder.Builder.CreateLoad(
          PUpperBound->getType()->getPointerElementType(), PUpperBound);
      OMPBuilder.Builder.CreateStore(LoadPUpperBound, UB);
      // Add lower bound to IV.
      Value *LowerBound = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
      Value *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, IV);
      Value *UpdateIV = OMPBuilder.Builder.CreateAdd(LoadIV, LowerBound);
      OMPBuilder.Builder.CreateStore(UpdateIV, IV);

      // Add fini call, reductions, and barrier after the loop exit block.
      BasicBlock *FiniBB = SplitBlock(Exit, &*Exit->getFirstInsertionPt());
      BasicBlock *NextFiniBB = SplitBlock(FiniBB, &*FiniBB->getFirstInsertionPt());
      OMPBuilder.Builder.SetInsertPoint(FiniBB, FiniBB->getFirstInsertionPt());
      OMPBuilder.Builder.CreateCall(KmpcForStaticFini, {SrcLoc, ThreadNum});

      // Emit reductions, barrier if standalone.
      if (IsStandalone) {
        if (!ReductionInfos.empty())
          OMPBuilder.createReductions(OMPBuilder.Builder.saveIP(), AllocaIP,
                                      ReductionInfos);

        OMPBuilder.Builder.SetInsertPoint(NextFiniBB->getTerminator());
        OMPBuilder.createBarrier(OpenMPIRBuilder::LocationDescription(
                                     OMPBuilder.Builder.saveIP(), Loc.DL),
                                 omp::Directive::OMPD_for,
                                 /* ForceSimpleCall */ false,
                                 /* CheckCancelFlag */ false);
      }
    }

    static void emitOMPTask(Module &M, OpenMPIRBuilder &OMPBuilder,
                            MapVector<Value *, DSAType> &DSAValueMap,
                            Function *Fn, BasicBlock *BBEntry,
                            BasicBlock *StartBB, BasicBlock *EndBB,
                            BasicBlock *AfterBB) {
      // Define types.
      // ************** START TYPE DEFINITION ************** //
      enum {
        TiedFlag = 0x1,
        FinalFlag = 0x2,
        DestructorsFlag = 0x8,
        PriorityFlag = 0x20,
        DetachableFlag = 0x40,
      };

      // This is a union for priority/firstprivate destructors, use the
      // routine entry pointer to allocate space since it is larger than
      // Int32Ty for priority, see kmp.h. Unused for now.
      StructType *KmpCmplrdataTy =
          StructType::create({OMPBuilder.TaskRoutineEntryPtr});
      StructType *KmpTaskTTy = StructType::create(
          {OMPBuilder.VoidPtr, OMPBuilder.TaskRoutineEntryPtr, OMPBuilder.Int32,
           KmpCmplrdataTy, KmpCmplrdataTy},
          "struct.kmp_task_t");
      Type *KmpTaskTPtrTy = KmpTaskTTy->getPointerTo();

      FunctionCallee KmpcOmpTaskAlloc = OMPBuilder.getOrCreateRuntimeFunction(
          M, OMPRTL___kmpc_omp_task_alloc);
      SmallVector<Type *, 8> SharedsTy;
      SmallVector<Type *, 8> PrivatesTy;
      for (auto &It : DSAValueMap) {
        Value *OriginalValue = It.first;
        if (It.second == DSA_SHARED)
          SharedsTy.push_back(OriginalValue->getType());
        else if (It.second == DSA_PRIVATE || It.second == DSA_FIRSTPRIVATE) {
          assert(isa<PointerType>(OriginalValue->getType()) &&
                 "Expected private, firstprivate value with pointer type");
          // Store a copy of the value, thus get the pointer element type.
          PrivatesTy.push_back(
              OriginalValue->getType()->getPointerElementType());
        } else
          assert(false && "Unknown DSA type");
      }

      StructType *KmpSharedsTTy = nullptr;
      if (SharedsTy.empty())
          KmpSharedsTTy = StructType::create(M.getContext(), "struct.kmp_shareds");
      else
          KmpSharedsTTy = StructType::create(SharedsTy, "struct.kmp_shareds");
      assert(KmpSharedsTTy && "Expected non-null KmpSharedsTTy");
      Type *KmpSharedsTPtrTy = KmpSharedsTTy->getPointerTo();
      StructType *KmpPrivatesTTy =
          StructType::create(PrivatesTy, "struct.kmp_privates");
      Type *KmpPrivatesTPtrTy = KmpPrivatesTTy->getPointerTo();
      StructType *KmpTaskTWithPrivatesTy = StructType::create(
          {KmpTaskTTy, KmpPrivatesTTy}, "struct.kmp_task_t_with_privates");
      Type *KmpTaskTWithPrivatesPtrTy = KmpTaskTWithPrivatesTy->getPointerTo();

      // Declare the task entry function.
      Function *TaskEntryFn = Function::Create(
          OMPBuilder.TaskRoutineEntry, GlobalValue::InternalLinkage,
          Fn->getAddressSpace(), Fn->getName() + ".omp_task_entry", &M);
      // Name arguments.
      TaskEntryFn->getArg(0)->setName(".global_tid");
      TaskEntryFn->getArg(1)->setName(".task_t_with_privates");

      // Declare the task outlined function.
      FunctionType *TaskOutlinedFnTy = FunctionType::get(
          OMPBuilder.Void,
          {OMPBuilder.Int32, OMPBuilder.Int32Ptr, OMPBuilder.VoidPtr,
           KmpTaskTPtrTy, KmpSharedsTPtrTy},
          /*isVarArg=*/false);
      Function *TaskOutlinedFn = Function::Create(
          TaskOutlinedFnTy, GlobalValue::InternalLinkage, Fn->getAddressSpace(),
          Fn->getName() + ".omp_task_outlined", &M);
      TaskOutlinedFn->getArg(0)->setName(".global_tid");
      TaskOutlinedFn->getArg(1)->setName(".part_id");
      TaskOutlinedFn->getArg(2)->setName(".privates");
      TaskOutlinedFn->getArg(3)->setName(".task.data");
      TaskOutlinedFn->getArg(4)->setName(".shareds");

      // ************** END TYPE DEFINITION ************** //

      // Emit kmpc_omp_task_alloc, kmpc_omp_task
      {
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
        OpenMPIRBuilder::LocationDescription Loc(
            InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
            DL);
        Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
        Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr);
        // TODO: parse clauses, for now fix flags to tied
        unsigned TaskFlags = TiedFlag;
        Value *SizeofShareds = nullptr;
        if (KmpSharedsTTy->isEmptyTy())
          SizeofShareds = OMPBuilder.Builder.getInt64(0);
        else
          SizeofShareds = OMPBuilder.Builder.getInt64(
              M.getDataLayout().getTypeAllocSize(KmpSharedsTTy));
        Value *SizeofKmpTaskTWithPrivates = OMPBuilder.Builder.getInt64(
            M.getDataLayout().getTypeAllocSize(KmpTaskTWithPrivatesTy));
        OMPBuilder.Builder.SetInsertPoint(BBEntry,
                                          BBEntry->getFirstInsertionPt());
        Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);
        Value *KmpTaskTWithPrivatesVoidPtr = OMPBuilder.Builder.CreateCall(
            KmpcOmpTaskAlloc,
            {SrcLoc, ThreadNum, OMPBuilder.Builder.getInt32(TaskFlags),
             SizeofKmpTaskTWithPrivates, SizeofShareds, TaskEntryFn},
            ".task.data");
        Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
            KmpTaskTWithPrivatesVoidPtr, KmpTaskTWithPrivatesPtrTy);

        const unsigned KmpTaskTIdx = 0;
        const unsigned KmpSharedsIdx = 0;
        Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpTaskTIdx);
        Value *KmpSharedsGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTTy, KmpTaskT, KmpSharedsIdx);
        Value *KmpSharedsVoidPtr =
            OMPBuilder.Builder.CreateLoad(OMPBuilder.VoidPtr, KmpSharedsGEP);
        Value *KmpShareds = OMPBuilder.Builder.CreateBitCast(KmpSharedsVoidPtr,
                                                             KmpSharedsTPtrTy);
        const unsigned KmpPrivatesIdx = 1;
        Value *KmpPrivates = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpPrivatesIdx);

        // Store shareds by reference, firstprivates by value, in task data storage.
        unsigned SharedsGEPIdx = 0;
        unsigned PrivatesGEPIdx = 0;
        for (auto &It : DSAValueMap) {
          Value *OriginalValue = It.first;
          if (It.second == DSA_SHARED) {
            Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpSharedsTTy, KmpShareds, SharedsGEPIdx,
                OriginalValue->getName() + ".task.shared");
            OMPBuilder.Builder.CreateStore(OriginalValue, SharedGEP);
            ++SharedsGEPIdx;
          } else if (It.second == DSA_FIRSTPRIVATE) {
            Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpPrivatesTTy, KmpPrivates, PrivatesGEPIdx,
                OriginalValue->getName() + ".task.firstprivate");
            Value *Load = OMPBuilder.Builder.CreateLoad(
                OriginalValue->getType()->getPointerElementType(),
                OriginalValue);
            OMPBuilder.Builder.CreateStore(Load, FirstprivateGEP);
            ++PrivatesGEPIdx;
          } else if (It.second == DSA_PRIVATE)
            ++PrivatesGEPIdx;
        }

        FunctionCallee KmpcOmpTask =
            OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_task);
        OMPBuilder.Builder.CreateCall(
            KmpcOmpTask, {SrcLoc, ThreadNum, KmpTaskTWithPrivatesVoidPtr});
      }

      // Emit task entry function.
      {
        BasicBlock *TaskEntryBB =
            BasicBlock::Create(M.getContext(), "entry", TaskEntryFn);
        OMPBuilder.Builder.SetInsertPoint(TaskEntryBB);
        const unsigned TaskTIdx= 0;
        const unsigned PrivatesIdx = 1;
        const unsigned SharedsIdx = 0;
        Value *GTId = TaskEntryFn->getArg(0);
        Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
            TaskEntryFn->getArg(1), KmpTaskTWithPrivatesPtrTy);
        Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, TaskTIdx,
            ".task.data");
        Value *SharedsGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTTy, KmpTaskT, SharedsIdx, ".shareds.gep");
        Value *SharedsVoidPtr = OMPBuilder.Builder.CreateLoad(
            OMPBuilder.VoidPtr, SharedsGEP, ".shareds.void.ptr");
        Value *Shareds = OMPBuilder.Builder.CreateBitCast(
            SharedsVoidPtr, KmpSharedsTPtrTy, ".shareds");

        Value *Privates = nullptr;
        if (PrivatesTy.empty()) {
          Privates = Constant::getNullValue(OMPBuilder.VoidPtr);
        } else {
          Value *PrivatesTyped = OMPBuilder.Builder.CreateStructGEP(
              KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, PrivatesIdx,
              ".privates");
          Privates = OMPBuilder.Builder.CreateBitCast(
              PrivatesTyped, OMPBuilder.VoidPtr, ".privates.void.ptr");
        }
        assert(Privates && "Expected non-null privates");

        const unsigned PartIdIdx = 2;
        Value *PartId = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTTy, KmpTaskT, PartIdIdx, ".part_id");
        OMPBuilder.Builder.CreateCall(
            TaskOutlinedFnTy, TaskOutlinedFn,
            {GTId, PartId, Privates, KmpTaskT, Shareds});
        OMPBuilder.Builder.CreateRet(OMPBuilder.Builder.getInt32(0));
      }

      // Emit TaskOutlinedFn code.
      {
        OpenMPIRBuilder::OutlineInfo OI;
        OI.EntryBB = StartBB;
        OI.ExitBB = EndBB;
        SmallPtrSet<BasicBlock *, 8> OutlinedBlockSet;
        SmallVector<BasicBlock *, 8> OutlinedBlockVector;
        OI.collectBlocks(OutlinedBlockSet, OutlinedBlockVector);
        BasicBlock *TaskOutlinedEntryBB =
            BasicBlock::Create(M.getContext(), "entry", TaskOutlinedFn);
        BasicBlock *TaskOutlinedExitBB =
            BasicBlock::Create(M.getContext(), "exit", TaskOutlinedFn);
        for (BasicBlock *BB : OutlinedBlockVector)
          BB->moveBefore(TaskOutlinedExitBB);
        // Explicitly move EndBB to the outlined functions, since OutlineInfo
        // does not contain it in the OutlinedBlockVector.
        EndBB->moveBefore(TaskOutlinedExitBB);
        EndBB->getTerminator()->setSuccessor(0, TaskOutlinedExitBB);

        OMPBuilder.Builder.SetInsertPoint(TaskOutlinedEntryBB);
        const unsigned KmpPrivatesArgNo = 2;
        const unsigned KmpSharedsArgNo = 4;
        Value *KmpPrivatesArgVoidPtr = TaskOutlinedFn->getArg(KmpPrivatesArgNo);
        Value *KmpPrivatesArg = OMPBuilder.Builder.CreateBitCast(
            KmpPrivatesArgVoidPtr, KmpPrivatesTPtrTy);
        Value *KmpSharedsArg = TaskOutlinedFn->getArg(KmpSharedsArgNo);

        // Replace shareds, privates, firstprivates to refer to task data
        // storage.
        unsigned SharedsGEPIdx = 0;
        unsigned PrivatesGEPIdx = 0;
        for (auto &It : DSAValueMap) {
          Value *OriginalValue = It.first;
          Value *ReplacementValue = nullptr;
          if (It.second == DSA_SHARED) {
            Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpSharedsTTy, KmpSharedsArg, SharedsGEPIdx,
                OriginalValue->getName() + ".task.shared.gep");
            ReplacementValue = OMPBuilder.Builder.CreateLoad(
                OriginalValue->getType(), SharedGEP,
                OriginalValue->getName() + ".task.shared");
            ++SharedsGEPIdx;
          } else if (It.second == DSA_PRIVATE) {
            Value *PrivateGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
                OriginalValue->getName() + ".task.private.gep");
            ReplacementValue = PrivateGEP;
            ++PrivatesGEPIdx;
          } else if (It.second == DSA_FIRSTPRIVATE) {
            Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
                OriginalValue->getName() + ".task.firstprivate.gep");
            ReplacementValue = FirstprivateGEP;
            ++PrivatesGEPIdx;
          }
          else
            assert(false && "Unknown DSA type");

          assert(ReplacementValue && "Expected non-null ReplacementValue");
          SmallVector<User *, 8> Users(OriginalValue->users());
          for (User *U : Users)
            if (Instruction *I = dyn_cast<Instruction>(U))
              if (OutlinedBlockSet.contains(I->getParent()))
                I->replaceUsesOfWith(OriginalValue, ReplacementValue);
        }

        OMPBuilder.Builder.CreateBr(StartBB);
        OMPBuilder.Builder.SetInsertPoint(TaskOutlinedExitBB);
        OMPBuilder.Builder.CreateRetVoid();
        BBEntry->getTerminator()->setSuccessor(0, AfterBB);
      }
    }

    bool runOnModule(Module &M) override {
      LLVM_DEBUG(dbgs() << "=== Start IntrinsicsOpenMPPass v4\n");

      Function *RegionEntryF = M.getFunction("llvm.directive.region.entry");

      // Return early for lack of directive intrinsics.
      if (!RegionEntryF) {
        LLVM_DEBUG(dbgs() << "No intrinsics directives, exiting...\n");
        return false;
      }

      OpenMPIRBuilder OMPBuilder(M);
      OMPBuilder.initialize();
      Triple TargetTriple(M.getTargetTriple());
      bool IsOpenMPDevice = false;
      switch (TargetTriple.getArch()) {
      case llvm::Triple::nvptx:
      case llvm::Triple::nvptx64:
      case llvm::Triple::amdgcn:
        LLVM_DEBUG(dbgs() << "Codegen for OpenMP Device runtime\n");
        IsOpenMPDevice = true;
        break;
      default:
        LLVM_DEBUG(dbgs() << "Codegen for OpenMP CPU runtime\n");
      }

      LLVM_DEBUG(dbgs() << "=== Dump module\n" << M << "=== End of Dump module\n");

      // Iterate over all calls to directive intrinsics and transform code
      // using OpenMPIRBuilder for lowering.
      SmallVector<User *, 4> RegionEntryUsers(RegionEntryF->users());
      for (User *Usr : RegionEntryUsers) {
        LLVM_DEBUG(dbgs() << "Found Usr " << *Usr << "\n");
        CallBase *CBEntry = dyn_cast<CallBase>(Usr);
        assert(CBEntry && "Expected call to region entry intrinsic");

        // Extract the directive kind and data sharing attributes of values
        // from the operand bundles of the intrinsic call.
        Directive Dir = OMPD_unknown;
        SmallVector<OperandBundleDef, 16> OpBundles;
        MapVector<Value *, DSAType> DSAValueMap;

        struct {
          Value *IV = nullptr;
          Value *UB = nullptr;
          // Implementation defined: set default schedule to static.
          OMPScheduleType Sched = OMPScheduleType::Static;
          Value *Chunk = nullptr;
        } OMPLoopInfo;

        struct {
          Value *NumThreads = nullptr;
          Value *IfCondition = nullptr;
        } ParRegionInfo;

        CBEntry->getOperandBundlesAsDefs(OpBundles);
        // TODO: parse clauses.
        for (OperandBundleDef &O : OpBundles) {
          StringRef Tag = O.getTag();
          LLVM_DEBUG(dbgs() << "OPB " << Tag << "\n");

          // TODO: check for conflicting DSA, for example reduction variables
          // cannot be set private. Should be done in Numba.
          if (Tag.startswith("DIR")) {
            auto It = StringToDir.find(Tag);
            assert(It != StringToDir.end() && "Directive is not supported!");
            Dir = It->second;
          } else if (Tag.startswith("QUAL")) {
            for (auto I = O.input_begin(), E = O.input_end(); I != E; ++I) {
              Value *V = dyn_cast<Value>(*I);
              assert(V && "Expected Value");

              if (Tag.startswith("QUAL.OMP.NORMALIZED.IV")) {
                assert(O.input_size() == 1 && "Expected single IV value");
                OMPLoopInfo.IV = V;
              } else if (Tag.startswith("QUAL.OMP.NORMALIZED.UB")) {
                assert(O.input_size() == 1 && "Expected single UB value");
                OMPLoopInfo.UB = V;
              } else if (Tag.startswith("QUAL.OMP.NUM_THREADS")) {
                assert(O.input_size() == 1 &&
                       "Expected single NumThreads value");
                ParRegionInfo.NumThreads = V;
              } else if (Tag.startswith("QUAL.OMP.SCHEDULE")) {
                assert(O.input_size() == 1 &&
                       "Expected single chunking scheduling value");
                Constant *Zero = ConstantInt::get(V->getType(), 0);
                OMPLoopInfo.Chunk = V;

                if (Tag == "QUAL.OMP.SCHEDULE.STATIC") {
                  assert(V == Zero && "Chunking is not yet supported, requires "
                                      "the use of omp_stride (static_chunked)");
                  if (V == Zero)
                    OMPLoopInfo.Sched = OMPScheduleType::Static;
                  else
                    OMPLoopInfo.Sched = OMPScheduleType::StaticChunked;
                } else
                  assert(false && "Unsupported scheduling type");
              } else if (Tag.startswith("QUAL.OMP.IF")) {
                assert(O.input_size() == 1 &&
                       "Expected single if condition value");
                ParRegionInfo.IfCondition = V;
              } else if (Tag.startswith("QUAL.OMP.REDUCTION.ADD")) {
                DSAValueMap[V] = DSA_REDUCTION_ADD;
              } else /* DSA Qualifiers */ {
                auto It = StringToDSA.find(Tag);
                assert(It != StringToDSA.end() && "DSA type not found in map");
                DSAValueMap[V] = It->second;
              }
            }
          }
        }

        assert(Dir != OMPD_unknown && "Expected valid OMP directive");

        assert(CBEntry->getNumUses() == 1 &&
               "Expected single use of the directive entry CB");
        Use &U = *CBEntry->use_begin();
        CallBase *CBExit = dyn_cast<CallBase>(U.getUser());
        assert(CBExit && "Expected call to region exit intrinsic");
        LLVM_DEBUG(dbgs() << "Found Use of " << *CBEntry << "\n-> AT ->\n"
               << *CBExit << "\n");

        // Gather info.
        BasicBlock *BBEntry = CBEntry->getParent();
        Function *Fn = BBEntry->getParent();
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();

        // Create the basic block structure to isolate the outlined region.
        BasicBlock *StartBB = SplitBlock(BBEntry, CBEntry);
        assert(BBEntry->getUniqueSuccessor() == StartBB &&
               "Expected unique successor at region start BB");

        BasicBlock *BBExit = CBExit->getParent();
        BasicBlock *EndBB = SplitBlock(BBExit, CBExit->getNextNode());
        assert(BBExit->getUniqueSuccessor() == EndBB &&
               "Expected unique successor at region end BB");
        BasicBlock *AfterBB = SplitBlock(EndBB, &*EndBB->getFirstInsertionPt());

        // Define the default BodyGenCB lambda.
        auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                             BasicBlock &ContinuationIP) {
          BasicBlock *CGStartBB = CodeGenIP.getBlock();
          BasicBlock *CGEndBB = SplitBlock(CGStartBB, &*CodeGenIP.getPoint());
          assert(StartBB != nullptr && "StartBB should not be null");
          CGStartBB->getTerminator()->setSuccessor(0, StartBB);
          assert(EndBB != nullptr && "EndBB should not be null");
          EndBB->getTerminator()->setSuccessor(0, CGEndBB);
        };

        // Define the default FiniCB lambda.
        auto FiniCB = [&](InsertPointTy CodeGenIP) {};

        // Remove intrinsics of OpenMP tags, first CBExit to also remove use
        // of CBEntry, then CBEntry.
        CBExit->eraseFromParent();
        CBEntry->eraseFromParent();

        if (Dir == OMPD_parallel) {
          if (IsOpenMPDevice)
            emitOMPParallelDevice(M, OMPBuilder, DSAValueMap, DL, Fn, BBEntry,
                                  StartBB, EndBB, AfterBB, ParRegionInfo.IfCondition,
                                  ParRegionInfo.NumThreads);
          else
            emitOMPParallel(OMPBuilder, DSAValueMap, DL, Fn, BBEntry, StartBB,
                            EndBB, AfterBB, FiniCB, ParRegionInfo.IfCondition,
                            ParRegionInfo.NumThreads);

          LLVM_DEBUG(dbgs() << "=== Before Fn\n" << *Fn << "=== End of Before Fn\n");
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
          LLVM_DEBUG(dbgs() << "=== Finalize Fn\n" << *Fn << "=== End of Finalize Fn\n");
        } else if (Dir == OMPD_single) {
          // Set the insertion location at the end of the BBEntry.
          BBEntry->getTerminator()->eraseFromParent();
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->end()), DL);

          InsertPointTy AfterIP = OMPBuilder.createSingle(
              Loc, BodyGenCB, FiniCB, /*DidIt*/ nullptr);
          BranchInst::Create(AfterBB, AfterIP.getBlock());
          LLVM_DEBUG(dbgs() << "=== Single Fn\n" << *Fn << "=== End of Single Fn\n");
        } else if (Dir == OMPD_critical) {
          // Set the insertion location at the end of the BBEntry.
          BBEntry->getTerminator()->eraseFromParent();
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->end()), DL);

          InsertPointTy AfterIP =
              OMPBuilder.createCritical(Loc, BodyGenCB, FiniCB, "",
                                        /*HintInst*/ nullptr);
          BranchInst::Create(AfterBB, AfterIP.getBlock());
          LLVM_DEBUG(dbgs() << "=== Critical Fn\n" << *Fn << "=== End of Critical Fn\n");
        } else if (Dir == OMPD_barrier) {
          // Set the insertion location at the end of the BBEntry.
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
              DL);

          // TODO: check ForceSimpleCall usage.
          OMPBuilder.createBarrier(Loc, OMPD_barrier,
                                   /*ForceSimpleCall*/ false,
                                   /*CheckCancelFlag*/ true);
          LLVM_DEBUG(dbgs() << "=== Barrier Fn\n" << *Fn << "=== End of Barrier Fn\n");
        } else if (Dir == OMPD_for) {
          LLVM_DEBUG(dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n");
          LLVM_DEBUG(dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n");
          assert(OMPLoopInfo.IV && "Expected non-null IV");
          assert(OMPLoopInfo.UB && "Expected non-null UB");

          BasicBlock *PreHeader = StartBB;
          BasicBlock *Header = PreHeader->getUniqueSuccessor();
          BasicBlock *Exit = BBExit;
          assert(Header &&
                 "Expected unique successor from PreHeader to Header");
          LLVM_DEBUG(dbgs() << "=== PreHeader\n" << *PreHeader << "=== End of PreHeader\n");
          LLVM_DEBUG(dbgs() << "=== Header\n" << *Header << "=== End of Header\n");
          LLVM_DEBUG(dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n");

          emitOMPFor(M, OMPBuilder, DSAValueMap, OMPLoopInfo.IV, OMPLoopInfo.UB,
                     PreHeader, Exit, OMPLoopInfo.Sched, OMPLoopInfo.Chunk,
                     /* IsStandalone */ true);
          LLVM_DEBUG(dbgs() << "=== For Fn\n" << *Fn << "=== End of For Fn\n");
        } else if (Dir == OMPD_parallel_for) {
          // TODO: Verify the DSA for IV, UB since they are implicit in the
          // combined directive entry.
          assert(OMPLoopInfo.IV && "Expected non-null IV");
          assert(OMPLoopInfo.UB && "Expected non-null UB");

          // TODO: Check setting DSA for IV, UB for correctness.
          DSAValueMap[OMPLoopInfo.IV] = DSA_PRIVATE;
          DSAValueMap[OMPLoopInfo.UB] = DSA_FIRSTPRIVATE;

          BasicBlock *PreHeader = StartBB;
          BasicBlock *Header = PreHeader->getUniqueSuccessor();
          BasicBlock *Exit = BBExit;
          assert(Header &&
                 "Expected unique successor from PreHeader to Header");
          LLVM_DEBUG(dbgs() << "=== PreHeader\n" << *PreHeader << "=== End of PreHeader\n");
          LLVM_DEBUG(dbgs() << "=== Header\n" << *Header << "=== End of Header\n");
          LLVM_DEBUG(dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n");
          emitOMPFor(M, OMPBuilder, DSAValueMap, OMPLoopInfo.IV, OMPLoopInfo.UB, PreHeader,
                     Exit, OMPLoopInfo.Sched, OMPLoopInfo.Chunk, /* IsStandalone */ false);
          emitOMPParallel(OMPBuilder, DSAValueMap, DL, Fn, BBEntry, StartBB,
                          EndBB, AfterBB, FiniCB,
                          ParRegionInfo.IfCondition, ParRegionInfo.NumThreads);
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
        } else if (Dir == OMPD_task) {
          emitOMPTask(M, OMPBuilder, DSAValueMap, Fn, BBEntry, StartBB, EndBB,
                      AfterBB);
        } else if (Dir == OMPD_taskwait) {
          // Set the insertion location at the end of the BBEntry.
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
              DL);

          OMPBuilder.createTaskwait(Loc);
        } else if (Dir == OMPD_target) {
          LLVM_DEBUG(dbgs() << "TODO target\n");
        } else {
          LLVM_DEBUG(dbgs() << "Unknown directive " << *CBEntry << "\n");
          assert(false && "Unknown directive");
        }
      }

      LLVM_DEBUG(dbgs() << "=== Dump Lowered Module\n"
             << M << "=== End of Dump Lowered Module\n");

      LLVM_DEBUG(dbgs() << "=== End of IntrinsicsOpenMP pass\n");
      return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
    }
  };

}

PreservedAnalyses IntrinsicsOpenMPPass::run(Module &M, ModuleAnalysisManager &AM) {
    IntrinsicsOpenMP IOMP;
    bool Changed = IOMP.runOnModule(M);

    if(Changed)
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
}

char IntrinsicsOpenMP::ID = 0;
static RegisterPass<IntrinsicsOpenMP> X("intrinsics-openmp", "IntrinsicsOpenMP Pass");

// TODO: Explicitly add the pass to the builder to make sure it runs before any
// optimization?
//static RegisterStandardPasses Y(PassManagerBuilder::EP_ModuleOptimizerEarly,
//                                [](const PassManagerBuilder &Builder,
//                                   legacy::PassManagerBase &PM) {
//                                  PM.add(new IntrinsicsOpenMP());
//                                });

//static RegisterStandardPasses Z(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                                [](const PassManagerBuilder &Builder,
//                                   legacy::PassManagerBase &PM) {
//                                  PM.add(new IntrinsicsOpenMP());
//                                });
ModulePass *llvm::createIntrinsicsOpenMPPass() { return new IntrinsicsOpenMP(); }

void LLVMAddIntrinsicsOpenMPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIntrinsicsOpenMPPass());
}
