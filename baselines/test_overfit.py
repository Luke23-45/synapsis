import torch
import torch.nn as nn
from src.core.config import ExperimentConfig
from src.planner.planner_recent import PlannerRecent

def test_overfit():
    config = ExperimentConfig.for_dataset("pusht", "A1_recent")
    model = PlannerRecent(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Dummy data
    B, W, D = 2, config.data.history_window, config.data.proprio_dim
    K, A = config.data.action_chunk_size, config.data.action_dim
    
    batch = {
        "proprio": torch.randn(B, D).cuda(),
        "proprio_history": torch.randn(B, W, D).cuda(),
        "action_chunk": torch.randn(B, K, A).cuda()
    }

    print("Starting overfit test...")
    for i in range(50):
        optimizer.zero_grad()
        pred = model(batch)
        loss = nn.functional.mse_loss(pred, batch["action_chunk"])
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Step {i}: loss = {loss.item():.6f}")

if __name__ == "__main__":
    test_overfit()
