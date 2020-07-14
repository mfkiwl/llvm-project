//===- Apollo.cpp - Apollo instrumentation pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Apollo instrumentation and tuning for OpenMP parallel
// regions
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "apollo"

STATISTIC(ApolloOpenMPRegionsInstrumented,
          "Number of OpenMP parallel regions instrumented for Apollo");

cl::list<unsigned> NumThreadsList("apollo-omp-numthreads", cl::CommaSeparated,
                             cl::desc("Num threads for OpenMP tuning"),
                             cl::value_desc("1, 2, 3, ..."), cl::OneOrMore);

namespace {
struct Apollo : public ModulePass {
  static char ID;
  bool Changed = false;
  static bool HasRun;
  static unsigned RegionID;
  Apollo() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    if (HasRun)
      return false;

    HasRun = true;

    IRBuilder<> IRB(M.getContext());
    OpenMPIRBuilder OMPIRB(M);
    OMPIRB.initialize();

    FunctionCallee ApolloRegionCreate = M.getOrInsertFunction(
        "__apollo_region_create", IRB.getInt8PtrTy(), IRB.getInt32Ty(),
        IRB.getInt8PtrTy(), IRB.getInt32Ty());
    FunctionCallee ApolloRegionBegin = M.getOrInsertFunction(
        "__apollo_region_begin", IRB.getVoidTy(), IRB.getInt8PtrTy());
    FunctionCallee ApolloRegionSetFeature =
        M.getOrInsertFunction("__apollo_region_set_feature", IRB.getVoidTy(),
                              IRB.getInt8PtrTy(), IRB.getFloatTy());
    FunctionCallee ApolloGetPolicy = M.getOrInsertFunction(
        "__apollo_region_get_policy", IRB.getInt32Ty(), IRB.getInt8PtrTy());
    FunctionCallee ApolloRegionEnd = M.getOrInsertFunction(
        "__apollo_region_end", IRB.getVoidTy(), IRB.getInt8PtrTy());

    DenseMap<CallInst *, SmallVector<SmallVector<CallInst *, 1>, 1>>
        ForkCItoLoopInitCI;

    auto OMPLoopIDs = {
        OMPRTL___kmpc_for_static_init_4,
        OMPRTL___kmpc_for_static_init_4u,
        OMPRTL___kmpc_for_static_init_8,
        OMPRTL___kmpc_for_static_init_8u,
    };

    // Find all calls parallel for init populate a map from
    // the fork call to the callpath of the parallel for init
    for (auto &LID : OMPLoopIDs) {
      Function *F = OMPIRB.getOrCreateRuntimeFunctionPtr(LID);
      if (!F)
        continue;

      errs() << "Found FUNCTION " << F->getName() << "\n";
      for (User *U : F->users()) {
        CallInst *LoopInitCI = dyn_cast<CallInst>(U);
        if (!LoopInitCI)
          continue;

        Function *Caller = LoopInitCI->getFunction();
        errs() << "Caller " << Caller->getName() << "\n";

        // Find the AbstractCallSite fork call user and the callpath leading
        // to the parallel for init call
        CallInst *ForkCI = nullptr;
        SmallVector<CallInst *, 1> Callpath;
        Callpath.clear();
        Callpath.push_back(LoopInitCI);
        auto FindFork = [&](Function *F, auto &self) -> void {
          for (Use &U : F->uses()) {
            errs() << "User U " << *U.getUser() << "\n";
            CallInst *CI = dyn_cast<CallInst>(U.getUser());
            // If found a direct call, recurse to find the fork callback.
            if (CI) {
              Callpath.push_back(CI);
              self(CI->getFunction(), self);
            }
            AbstractCallSite ACS(&U);
            if (!ACS || !ACS.isCallbackCall())
              continue;

            ForkCI = dyn_cast<CallInst>(ACS.getInstruction());
            errs() << "ForkCI callee" << ForkCI->getCalledFunction()->getName()
                   << "\n";
            assert(ForkCI->getCalledFunction() ==
                       OMPIRB.getOrCreateRuntimeFunctionPtr(
                           OMPRTL___kmpc_fork_call) &&
                   "Must call OpenMP fork call");
          }
        };

        FindFork(Caller, FindFork);
        // Ignore dangling parallel for calls
        if (!ForkCI)
          continue;

        assert(ForkCI != nullptr);
        ForkCItoLoopInitCI[ForkCI].push_back(Callpath);
      }
    }

    // Instrument fork call and set features using the iteration count of each
    // parallel for loop in the callpath
    for (auto &Iter : ForkCItoLoopInitCI) {
      const unsigned CallbackCalleeOperand = 2;
      const unsigned CallbackFirstArgOperand = 3;
      const unsigned CalleeFirstArgNo = 2;

      const unsigned UpperBoundArgNo = 5;

      Changed = true;

      CallInst *ForkCI = Iter.first;
      SmallVector<SmallVector<CallInst *, 1>, 1> &LoopInitCIVector =
          Iter.second;

      // Inject __apollo_region_create using a GV to store the handle if
      // it does not exist
      IRB.SetInsertPoint(ForkCI);
      Constant *SrcLocStr = nullptr;
      // TODO: Use debug information when available for naming?
      /*
      IRB.SetCurrentDebugLocation(ForkCI->getDebugLoc());
      DILocation *DI = ForkCI->getDebugLoc().get();
      errs() << "DI " << *DI << "\n";
      if(DI)
        SrcLocStr = OMPIRB.getOrCreateSrcLocStr(IRB);
      else
      */
     // Get last 32 chars of Module name
      std::string ModuleSubstr = M.getName().str().substr(
          (M.getName().size() > 32) ? M.getName().size() - 32 : 0);
      SrcLocStr = IRB.CreateGlobalStringPtr(
          /*demangle(ForkCI->getFunction()->getName().str()) + */
          ModuleSubstr + ".apollo.region." + Twine(RegionID).str());

      assert(SrcLocStr != nullptr);
      errs() << "SrcLocStr " << *SrcLocStr << "\n";
      GlobalVariable *GV =
          new GlobalVariable(M, IRB.getInt8PtrTy(), /*isConstant=*/false,
                             GlobalValue::PrivateLinkage,
                             ConstantPointerNull::get(IRB.getInt8PtrTy()),
                             ".apollo.region.handle." + Twine(RegionID));
      GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      GV->setAlignment(Align(8));

      Value *Cond = IRB.CreateICmpEQ(
          IRB.CreateLoad(GV), ConstantPointerNull::get(IRB.getInt8PtrTy()));
      Instruction *ThenTI =
          SplitBlockAndInsertIfThen(Cond, ForkCI, /*Unreachable=*/false);
      IRB.SetInsertPoint(ThenTI);
      CallInst *ApolloRegionCreateCI = IRB.CreateCall(
          ApolloRegionCreate, {IRB.getInt32(LoopInitCIVector.size()), SrcLocStr,
                               // +1 to include the default case 0
                               IRB.getInt32(NumThreadsList.size() + 1)});
      IRB.CreateStore(ApolloRegionCreateCI, GV);

      // Instrument with __apollo_region_begin.
      IRB.SetInsertPoint(ForkCI);
      IRB.CreateCall(ApolloRegionBegin, {IRB.CreateLoad(GV)});
      errs() << "ForkCI " << *ForkCI << "\n";

      // Move instruction slices before fork call to calculate number of
      // iterations of each loop as features.
      for (SmallVector<CallInst *, 1> Callpath : LoopInitCIVector) {
        // Set of visited instructions for fast lookup.
        SmallPtrSet<Instruction *, 8> InstructionSet;
        // Slice of instructions in reverse order.
        SmallVector<Instruction *, 8> InstructionSliceReverse;
        Value *UB = nullptr;

        // Unpeel the callpath and slice instructions.
        for (CallInst *CI : Callpath) {
          errs() << "CI " << *CI << "\n";

          DominatorTree DT(*CI->getFunction());
          int count = 1;

          // Slice instructions based on Value V up to Insturction LimitI.
          auto Slice = [&](Value *V, Instruction *LimitI, auto &self) -> void {
            Instruction *I = dyn_cast<Instruction>(V);
            if (!I)
              return;

            // Instruction has been added, but we need to bump it up
            // the slice since it may be an operand of another 
            // instruction
            if (InstructionSet.count(I)) {
              for(auto it=InstructionSliceReverse.begin(); it!=InstructionSliceReverse.end(); it++) {
                if(*it != I)
                  continue;
                InstructionSliceReverse.erase(it);
                InstructionSliceReverse.push_back(I);
                errs() << "Bump I " << *I << "\n";
                break;
              }
              return;
            }

            errs() << "Follow I " << *I << "\n";
            InstructionSet.insert(I);

            for (User *U : V->users()) {
              Instruction *II = dyn_cast<Instruction>(U);
              if (!II)
                continue;
              if (InstructionSet.count(II))
                continue;
              if (!isa<StoreInst>(II))
                continue;
              if (!DT.dominates(II, CI))
                continue;

              errs() << count << " >>>>>>>>>>>>>>>>>>>>>>>>>>>> RECURSE I "
                     << *II << "\n";
              count++;
              self(II, LimitI, self);
              count--;
              errs() << count << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
            }

            InstructionSliceReverse.push_back(I);
            errs() << "Added Value I " << *I << "\n";
            errs() << count << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
            errs() << count << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";

            for (Value *V : I->operand_values()) {
              errs() << "Found OpI " << *V << "\n";
              self(V, LimitI, self);
            }
          };

          // LoopInitCI is by construction at the front of the Callpath, so get
          // the upper bound number of iterations to use as feature and slice
          // based to that.
          if (CI == Callpath.front()) {
            UB = CI->getArgOperand(UpperBoundArgNo);
            errs() << "UB " << *UB << "\n";
            Slice(UB, CI, Slice);
          } else {
            for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
              Value *V = CI->getArgOperand(i);
              errs() << "Arg " << *V << "\n";
              Slice(V, CI, Slice);
            }
          }
        }

        assert(UB != nullptr);
        errs() << "====================== RESULT ==========================\n";

        llvm::ValueToValueMapTy VMap;

        // Map arguments between the fork call callback and the outlined called
        // to be used by sliced instruction cloning.
        Function *Callee = dyn_cast<Function>(
            ForkCI->getArgOperand(CallbackCalleeOperand)->stripPointerCasts());
        errs() << "Callee " << Callee->getName() << "\n";
        errs() << "Forwarding " << *ForkCI << " -> " << Callee->getName()
               << "\n";
        // TODO: generalize for callbacks using metadata for argument list
        // Set the mappings between arguments in the outlined to the callback.
        for (unsigned u = CallbackFirstArgOperand, uu = CalleeFirstArgNo,
                      e = ForkCI->getNumArgOperands();
             u < e; ++u, ++uu) {
          errs() << "Mapping arg " << *Callee->getArg(uu) << " -> "
                 << *ForkCI->getArgOperand(u) << "\n";
          VMap[Callee->getArg(uu)] = ForkCI->getArgOperand(u);
          errs() << "Mapping arg " << *Callee->getArg(uu) << " -> "
                 << *VMap[Callee->getArg(uu)] << "\n";
        }

        // Map arguments in the Callpath used by sliced instruction cloning.
        for (CallInst *CI : reverse(Callpath)) {
          Function *Callee = CI->getCalledFunction();
          errs() << "Forwarding " << *CI << " -> " << Callee->getName() << "\n";
          for (unsigned i = 0; i < Callee->arg_size(); i++) {
            // Forward VMap if found due to nested Callpath.
            if (VMap.count(CI->getArgOperand(i)))
              VMap[Callee->getArg(i)] = VMap[CI->getArgOperand(i)];
            else
              VMap[Callee->getArg(i)] = CI->getArgOperand(i);
            errs() << "Mapping arg " << *Callee->getArg(i) << " -> "
                   << *VMap[Callee->getArg(i)] << "\n";
          }
        }

        // Clone instructions and remap them. Note inserting clones and
        // remapping them is correct, because instructions insert in program
        // order
        Value *UBClone = nullptr;
        for (auto It = InstructionSliceReverse.rbegin();
             It < InstructionSliceReverse.rend(); ++It) {
          Instruction *I = *It;
          errs() << "Slice I " << *I << "\n";
          Instruction *CloneI = I->clone();
          VMap[I] = CloneI;
          // Update value map for nested instructions clones in the Callpath.
          for (auto It : VMap) {
            const Value *Key = It.first;
            Value *Val = It.second;
            if (Val == I)
              VMap[Key] = CloneI;
          }
          Instruction *UBI = dyn_cast<Instruction>(UB);

          // Store the upper bound clone to create the feature
          if (UBI == I)
            UBClone = CloneI;

          // TODO: Ignore missing locals or not?
          // TODO: No module level changes?
          RemapInstruction(
              CloneI, VMap
              /*, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals*/);
          IRB.Insert(CloneI);
          errs() << "CloneI " << *CloneI << "\n\n";
        }

        assert(UBClone != nullptr);

        errs() << "UBClone I " << *UBClone << "\n";
        PointerType *UBPointerType = dyn_cast<PointerType>(UBClone->getType());
        assert(UBPointerType);
        Type *UBType = UBPointerType->getElementType();
        errs() << "UBClone Type " << *UBType << "\n";
        LoadInst *LoadI = IRB.CreateLoad(UBType, UBClone);
        Value *IntNumIters = IRB.CreateAdd(LoadI, ConstantInt::get(UBType, 1));
        Value *FloatNumIters = IRB.CreateUIToFP(IntNumIters, IRB.getFloatTy());
        IRB.CreateCall(ApolloRegionSetFeature,
                       {IRB.CreateLoad(GV), FloatNumIters});
      }
      errs() << "ForkCI " << *ForkCI << "\n";

      CallInst *ApolloGetPolicyCI =
          IRB.CreateCall(ApolloGetPolicy, {IRB.CreateLoad(GV)});

      BasicBlock *ForkBB = SplitBlock(ForkCI->getParent(), ForkCI);
      IRB.SetInsertPoint(ApolloGetPolicyCI->getParent()->getTerminator());
      OMPIRB.updateToLocation(IRB);
      SrcLocStr = OMPIRB.getOrCreateSrcLocStr(IRB);
      Value *Ident = OMPIRB.getOrCreateIdent(SrcLocStr);
      Value *ThreadID = OMPIRB.getOrCreateThreadID(Ident);
      SwitchInst *SwitchI = IRB.CreateSwitch(ApolloGetPolicyCI, ForkBB);
      ApolloGetPolicyCI->getParent()->getTerminator()->eraseFromParent();

      auto CreateCase = [&](int CaseNo, int NumThreads) {
        BasicBlock *Case =
            BasicBlock::Create(M.getContext(), ".apollo.case." + Twine(CaseNo),
                               ForkBB->getParent(), ForkBB);
        IRB.SetInsertPoint(Case);
        OMPIRB.updateToLocation(IRB);
        // Build call __kmpc_push_num_threads(&Ident, global_tid, num_threads).
        Value *Args[] = {
            Ident,
            ThreadID,
            IRB.getInt32(NumThreads),
        };
        IRB.CreateCall(OMPIRB.getOrCreateRuntimeFunctionPtr(
                           OMPRTL___kmpc_push_num_threads),
                       Args);
        IRB.CreateBr(ForkBB);
        SwitchI->addCase(IRB.getInt32(CaseNo), Case);
      };

      // Create different number of threads cases.
      // CaseNo starts from 1, 0 is the default case
      int CaseNo = 1;
      for (unsigned NumThreads : NumThreadsList) {
        CreateCase(CaseNo, NumThreads);
        CaseNo++;
      }

      OMPIRB.finalize();

      IRB.SetInsertPoint(ForkCI->getNextNode());
      IRB.CreateCall(ApolloRegionEnd, {IRB.CreateLoad(GV)});

      RegionID++;
      ApolloOpenMPRegionsInstrumented++;
    }

    return Changed;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (!Changed)
      AU.setPreservesAll();
  }
};
} // namespace

bool Apollo::HasRun = false;
unsigned Apollo::RegionID = 1;
char Apollo::ID = 0;
static RegisterPass<Apollo> X("apollo", "Apollo instrumentation module pass");

static RegisterStandardPasses Y(PassManagerBuilder::EP_ModuleOptimizerEarly,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Apollo());
                                });

static RegisterStandardPasses Z(PassManagerBuilder::EP_EnabledOnOptLevel0,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Apollo());
                                });
