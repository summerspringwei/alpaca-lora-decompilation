import torch
from torch.profiler import profile, record_function, ProfilerActivity
torch.cuda.memory._record_memory_history()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_modules=True, with_stack=True) as prof:
    # Your code here
    pass
prof.export_chrome_trace("trace.json")
torch.cuda.memory._dump_snapshot(f"snapshot-backward.pickle")
