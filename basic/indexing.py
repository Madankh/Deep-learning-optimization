import  torch
import time

M = K = N = 1024
a = torch.randn(M, K, device='cuda')
b = torch.randn(K, N, device='cuda')
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    c = a@b

end = time.time()
print("Pytorch avg ms:", (end - start) * 1000 / 100)
