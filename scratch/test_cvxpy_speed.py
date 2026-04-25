import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import time

def test():
    T = 100
    K = 10
    r = 2
    lam = 1.0
    B = 32

    print("Building CVXPY Problem...")
    y = cp.Variable(T)
    s = cp.Parameter(T)
    objective = cp.Minimize(lam * cp.sum_squares(y) - s @ y)
    constraints = [
        cp.sum(y) <= K,
        y[0] == 0,
        y >= 0,
        y <= 1
    ]
    for t in range(T):
        for u in range(t + 1, min(t + r + 1, T)):
            constraints.append(y[t] + y[u] <= 1)
    
    prob = cp.Problem(objective, constraints)
    print("Initializing CvxpyLayer...")
    layer = CvxpyLayer(prob, parameters=[s], variables=[y])

    # Fake input
    saliency = torch.randn(B, T, requires_grad=True)
    
    print("Forward Pass...")
    start = time.time()
    y_star, = layer(saliency, solver_args={"max_iters": 2000})
    print(f"Forward took {time.time() - start:.4f}s")
    
    print("Backward Pass...")
    loss = y_star.sum()
    start = time.time()
    loss.backward()
    print(f"Backward took {time.time() - start:.4f}s")
    print(f"Gradient norm: {saliency.grad.norm().item():.4f}")

if __name__ == "__main__":
    test()
