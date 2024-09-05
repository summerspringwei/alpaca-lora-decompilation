

import utils.preprocessing_assembly

assembly_str = '\t.text\n\t.file\t\"exebench_lscat-ACT41_2318101tct6vlp_.c\"\n\t.globl\tinit_friends_data               # -- Begin function init_friends_data\n\t.p2align\t4, 0x90\n\t.type\tinit_friends_data,@function\ninit_friends_data:                      # @init_friends_data\n\t.cfi_startproc\n# %bb.0:                                # %entry\n\tmovl\tfriends_replay_logevent(%rip), %eax\n\tmovl\t%eax, replay_logevent(%rip)\n\txorl\t%eax, %eax\n\tretq\n.Lfunc_end0:\n\t.size\tinit_friends_data, .Lfunc_end0-init_friends_data\n\t.cfi_endproc\n                                        # -- End function\n\t.ident\t\"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 88bf774c565080e30e0a073676c316ab175303af)\"\n\t.section\t\".note.GNU-stack\",\"\",@progbits\n\n'

output = utils.preprocessing_assembly.preprocessing_assembly(assembly_str)

print(output)
