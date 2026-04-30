import torch
a = torch.tensor([1.,2.,3.])

print(torch.square(a))
print(a**2)
print(a*a)

def time_pytorch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm up
    for _ in range(10):
        func(input)
    start.record()
    func(input)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000,10000).cuda()

def square_func2(x):
    return x * x

def square_3(x):
    return x ** 2

time_pytorch_function(torch.square, b)
time_pytorch_function(square_func2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")


# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_func2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))