

### AnghaBench-ll-O2/darwin-xnu/bsd/net/pktsched/extr_pktsched_qfq.c_qfq_calc_index/predict.ll

Error 1:
llc: error: llc: predict.ll:31:25: error: '%14' defined with type 'i32' but expected 'i64'
  %16 = shl nsw i64 -1, %14


Error2 :
llc: error: llc: predict_fix.ll:57:3: error: instruction expected to be numbered '%34'
  %35 = getelementptr inbounds %struct.qfq_if, ptr %0, i64 0, i32 1
Label error: labels should be numbered correctly

After fix the previous errors, there are still errors:

PHINode should have one entry for each predecessor of its parent basic block!
  %29 = phi i32 [ %22, %25 ], [ 0, %8 ]
Instruction does not dominate all uses!
  %29 = phi i32 [ %22, %25 ], [ 0, %8 ]
  ret i32 %29
Instruction does not dominate all uses!
  %26 = load i32, ptr @LOG_DEBUG, align 4, !tbaa !5
  %43 = tail call i32 @log(i32 noundef %26, ptr noundef nonnull @.str, i32 noundef %37, i32 noundef %39, i32 noundef %40, i32 noundef %29, i32 noundef %42) #2
llc: error: 'predict_fix.ll': input module cannot be verified


### AnghaBench-ll-O2/darwin-xnu/osfmk/kern/extr_coalition.c_i_coal_jetsam_remove_task

Error 1:
llc: error: llc: predict.ll:54:41: error: '%30' defined with type 'i64' but expected 'ptr'
  %32 = getelementptr inbounds i32, ptr %30, i64 %31
Fix: change the define of %30 from int to ptr

Error 2:
Did not see access type in access path!
  %30 = load ptr, ptr %1, align 8, !tbaa !18
!18 = !{!14, !10, i64 0}
Did not see access type in access path!
  %38 = load ptr, ptr %1, align 8, !tbaa !18
!18 = !{!14, !10, i64 0}
Did not see access type in access path!
  %42 = load ptr, ptr %1, align 8, !tbaa !18
!18 = !{!14, !10, i64 0}
llc: error: 'predict_fix.ll': input module cannot be verified



### AnghaBench-ll-O2/darwin-xnu/osfmk/vm/extr_vm_pageout.c_vector_upl_subupl_byoffset

Error 1:
llc: error: llc: predict.ll:58:51: error: '%20' defined with type 'ptr' but expected 'i32'
  %36 = tail call i32 @llvm.smin.i32(i32 %21, i32 %20)

Fix: change %20 to %21
Fix the labels follow the numbers
Compile Pass!
It seems write!



### AnghaBench-ll-O2/fastsocket/kernel/arch/ia64/kernel/extr_smpboot.c_get_delta
Error 1:
llc: error: llc: predict_fix.ll:38:3: error: instruction forward referenced with type 'i64'
  %24 = getelementptr inbounds i64, ptr %22, i64 %23

Label/Type totally wrong, cannot fix by human!



### AnghaBench-ll-O2/fastsocket/kernel/drivers/infiniband/hw/qib/extr_qib_verbs.c_wait_kmem
Only one function call wrong!

Predict:
```c
tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %18, i64 8, i1 false), !tbaa.struct !15
```
Actual:
```c
%25 = tail call i32 @mod_timer(ptr noundef nonnull %22, i64 noundef %24) #2
```


### AnghaBench-ll-O2/fastsocket/kernel/drivers/net/benet/extr_be_main.c_be_set_vf_tx_rate/predict_fix.ll
Function call wrong!
Like `c_wait_kmem`
Others are right!


### AnghaBench-ll-O2/fastsocket/kernel/drivers/net/wan/extr_sdla.c_sdla_deassoc
confusion between label and variable

llc: error: llc: predict.ll:77:3: error: instruction forward referenced with type 'label'
  %43 = tail call i32 @sdla_cmd(ptr noundef %0, i32 noundef %42, ptr noundef null, ptr noundef null, i32 noundef 0, i32 noundef 2, ptr noundef null, ptr noundef null) #3
Control graph wrong.
Can not fix by human.


### AnghaBench-ll-O2/fastsocket/kernel/drivers/net/wireless/brcm80211/brcmsmac/extr_main.c_brcms_b_write_template_ram
llc: error: llc: predict_fix.ll:56:3: error: instruction forward referenced with type 'label'
  %36 = icmp sgt i32 %18, 4

Fixed by change forward label from `%36` to `%37`
Looks right!


### AnghaBench-ll-O2/fastsocket/kernel/drivers/net/wireless/iwlwifi/dvm/extr_devices.c_iwl6000_nic_config
Compile pass but
Misses a function call, wrong!

### AnghaBench-ll-O2/fastsocket/kernel/drivers/net/wireless/libertas/extr_assoc.c_lbs_try_associate
Looks right


### AnghaBench-ll-O2/fastsocket/kernel/drivers/net/wireless/brcm80211/brcmsmac/extr_main.c_brcms_b_write_template_ram
llc: error: llc: predict.ll:56:3: error: instruction forward referenced with type 'label'
  %36 = icmp sgt i32 %18, 4
  ^

After fix label, looks right.

### AnghaBench-ll-O2/fastsocket/kernel/drivers/s390/scsi/extr_zfcp_fsf.c_zfcp_fsf_set_data_dir

llc: error: llc: predict.ll:70:3: error: instruction forward referenced with type 'label'
  %21 = load i32, ptr @FSF_DATADIR_DIF_WRITE_CONVERT, align 4, !tbaa !10
  ^

After fix label, can compile,
but the switch conditions are not right.


### AnghaBench-ll-O2/fastsocket/kernel/net/ipv4/extr_tcp_input.c_tcp_urg
llc: error: llc: predict.ll:68:20: error: '%40' is not a basic block
  br i1 %43, label %40, label %44

Can not fix, basic block didn't match, thus we can not find the corresponding label for `40`.


### AnghaBench-ll-O2/fastsocket/kernel/sound/core/seq/extr_seq_dummy.c_dummy_unuse
llc: error: llc: predict.ll:22:55: error: invalid getelementptr indices
  %8 = getelementptr inbounds %struct.snd_seq_client, ptr %0, i64 0, i32 1, i32 1
                                                      ^

Get struct member wrong.


### AnghaBench-ll-O2/fastsocket/kernel/sound/core/seq/extr_seq_dummy.c_dummy_unuse
It's right!


### AnghaBench-ll-O2/freebsd/contrib/blacklist/bin/extr_blacklistd.c_rules_flush
llc: error: llc: predict.ll:50:9: error: '%24' defined with type 'i64' but expected 'i1'
  br i1 %24, label %18, label %25, !llvm.loop !16
        ^
  
Does not generate the `cmp` instruction:
Actual:
```c
  %23 = add nuw i64 %19, 1
  %24 = load i64, ptr @lconf, align 8, !tbaa !11
  br i1 %24, label %18, label %25, !llvm.loop !16
``

Expected:
```C
  %25 = add nuw i64 %21, 1
  %26 = load i64, ptr @lconf, align 8, !tbaa !13
  %27 = icmp ult i64 %25, %26
  br i1 %27, label %20, label %17, !llvm.loop !19
```

### AnghaBench-ll-O2/freebsd/contrib/file/src/extr_funcs.c_file_replace
llc: error: llc: predict.ll:62:3: error: instruction forward referenced with type 'i32'
  %35 = call i64 @file_regexec(ptr noundef nonnull %4, ptr noundef %34, i32 noundef 1, ptr noundef nonnull %5, ptr noundef null) #3
  ^

fixed to right lable. Looks right!


### AnghaBench-ll-O2/freebsd/contrib/gcc/extr_cgraphunit.c_cgraph_varpool_remove_unreferenced_decls

llc: error: llc: predict.ll:46:3: error: instruction forward referenced with type 'label'
  %25 = icmp eq i64 %24, 0
  ^

Wrong label and wrong function call. Can not fix.

### AnghaBench-ll-O2/freebsd/contrib/gcc/extr_gcse.c_load_killed_in_block_p

llc: error: llc: predict.ll:55:3: error: instruction forward referenced with type 'label'
  %31 = tail call i64 @XEXP(i64 noundef %16, i32 noundef 1) #2
  ^
Wrong program structure, cannot fix.

