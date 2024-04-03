

### AnghaBench-ll-O2/Craft/deps/sqlite/extr_sqlite3.c_sqlite3MemSize
Semantic right!

### AnghaBench-ll-O2/darwin-xnu/bsd/kern/extr_kpi_socket.c_sock_iskerne
Semantic right!

### AnghaBench-ll-O2/darwin-xnu/bsd/net/extr_kpi_interface.c_ifnet_unit
Semantic right!

### AnghaBench-ll-O2/darwin-xnu/osfmk/arm/extr_loose_ends.c_fls
Wrong instructions:

Predicted:
```C
3:                                                ; preds = %1
  %4 = tail call i32 @llvm.ctlz.i32(i32 %0, i1 true), !range !5
  %5 = add nuw nsw i32 %4, 32
  %6 = add nsw i32 %5, -33
  br label %7
```
Groud truth:
```C
3:                                                ; preds = %1
  %4 = tail call i32 @llvm.ctlz.i32(i32 %0, i1 true), !range !5
  %5 = sub nuw nsw i32 32, %4
  br label %6
```

### AnghaBench-ll-O2/esp-idf/components/freertos/test/extr_test_task_priorities.c_counter_task

Perfect decompilation

### AnghaBench-ll-O2/esp-idf/components/perfmon/test/extr_test_perfmon_ansi.c_exec_callback
Changed loop condition.
Semantic right!


### AnghaBench-ll-O2/esp-idf/components/perfmon/test/extr_test_perfmon_ansi.c_test_ca
Semantic right!
Changed loop condition.

### AnghaBench-ll-O2/esp-idf/examples/system/perfmon/main/extr_perfmon_example_main.c_exec_test_function
Semantic right!

### AnghaBench-ll-O2/fastsocket/kernel/arch/s390/lib/extr_ucmpdi2.c___ucmpdi2
Semantic right!

### AnghaBench-ll-O2/fastsocket/kernel/arch/x86/kernel/extr_amd_iommu.c_check_device
Semantic right!



### AnghaBench-ll-O2/fastsocket/kernel/drivers/hwmon/extr_g760a.c_rpm_from_cnt
Maybe wrong!

### AnghaBench-ll-O2/fastsocket/kernel/drivers/infiniband/hw/qib/extr_qib_pcie.c_fld2va
Perfect decompilation!

### AnghaBench-ll-O2/fastsocket/kernel/drivers/md/extr_dm.c_dm_get_mapinfo

Semantic right

### AnghaBench-ll-O2/fastsocket/kernel/drivers/target/extr_target_core_spc.c_spc_modesense_dpofua

Semantic similar but wrong type!
Predict:
```C
  %5 = load i32, ptr %0, align 4, !tbaa !5
  %6 = or i32 %5, 16
  store i32 %6, ptr %0, align 4, !tbaa !5
  br label %7
```
Ground truth:
```C
  %5 = load i8, ptr %0, align 1, !tbaa !5
  %6 = or i8 %5, 16
  store i8 %6, ptr %0, align 1, !tbaa !5
  br label %7
```


### AnghaBench-ll-O2/fastsocket/kernel/net/netfilter/ipvs/extr_ip_vs_wrr.c_gcd

GCD different solution.

Semantic right.


### freebsd/contrib/binutils/ld/extr_ldexp.c_align_n
Semantic right.


### /data0/xiachunwei/Projects/alpaca-lora/freebsd/contrib/gcc/config/arm/extr_arm.c_number_of_first_bit_set
llvm-diff report diff but actually equal.
Perfect decompilation!


### freebsd/contrib/gdb/gdb/extr_ui-out.c_uo_flush
Perfect decompilation


### freebsd/contrib/ntp/ntpd/extr_ntp_prio_q.c_empty

Note: llvm-diff can not compare the phi node
Perfect decompilation


### freebsd/contrib/subversion/subversion/libsvn_fs_fs/extr_index.c_decode_int
Semantic right!
