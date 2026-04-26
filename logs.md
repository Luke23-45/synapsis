2026-04-26 16:45:38,720 [INFO] __main__: Loaded config: configs/experiment/full.yaml
2026-04-26 16:45:38,724 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-26 16:45:38,724 [INFO] __main__: SYNAPSE Phase 4 Experiment: 3 datasets × 3 conditions
2026-04-26 16:45:38,724 [INFO] __main__:   Datasets: ['pusht', 'aloha_transfer', 'xarm_lift']
2026-04-26 16:45:38,724 [INFO] __main__:   Conditions: ['A1_recent', 'A2_uniform', 'B_synapse']
2026-04-26 16:45:38,724 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-26 16:45:38,724 [INFO] __main__: 
▓▓▓ DATASET: PUSHT ▓▓▓

2026-04-26 16:45:38,724 [INFO] __main__: Loading dataset: pusht (source=lerobot)
2026-04-26 16:45:38,766 [INFO] src.data.adapters.lerobot_adapter: Loading from local Parquet: /content/synapsis/baselines/data/datasets/pusht/train.parquet
2026-04-26 16:45:38,851 [INFO] numexpr.utils: NumExpr defaulting to 2 threads.
2026-04-26 16:45:39,014 [INFO] src.data.adapters.lerobot_adapter: Loaded 25650 rows from Parquet
2026-04-26 16:45:39,090 [INFO] src.data.adapters.lerobot_adapter: LeRobotAdapter[pusht]: loaded 206 episodes (proprio_dim=2, action_dim=2, phases=3)
2026-04-26 16:45:39,092 [INFO] __main__: Loaded 206 episodes from 'pusht' (proprio_dim=2, action_dim=2)
2026-04-26 16:45:39,093 [INFO] src.data.dataset: Split 206 episodes: train=164, val=20, test=22
2026-04-26 16:45:39,094 [INFO] src.core.normalization: Computing normalization stats from 20719 timesteps × 2 dimensions
2026-04-26 16:45:39,095 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=2, mean_range=[229.3067, 293.6066], std_range=[96.2621, 101.0113])
2026-04-26 16:45:39,097 [INFO] src.core.normalization: Saved normalization stats to output/20260426_164538_phase4_cross_domain/dataset_pusht/normalization_stats.pt
2026-04-26 16:45:39,100 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 164 episodes → output/20260426_164538_phase4_cross_domain/dataset_pusht/synapse_features/train.pt
2026-04-26 16:45:39,108 [INFO] src.synapse.synapse_cache: Cached 164 episodes (0 errors). Anchors: [164, 10, 5], Topo: [164, 8]
2026-04-26 16:45:39,109 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_164538_phase4_cross_domain/dataset_pusht/synapse_features/train.pt: 164 episodes
2026-04-26 16:45:39,111 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 20 episodes → output/20260426_164538_phase4_cross_domain/dataset_pusht/synapse_features/val.pt
2026-04-26 16:45:39,113 [INFO] src.synapse.synapse_cache: Cached 20 episodes (0 errors). Anchors: [20, 10, 5], Topo: [20, 8]
2026-04-26 16:45:39,113 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_164538_phase4_cross_domain/dataset_pusht/synapse_features/val.pt: 20 episodes
2026-04-26 16:45:39,113 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 22 episodes → output/20260426_164538_phase4_cross_domain/dataset_pusht/synapse_features/test.pt
2026-04-26 16:45:39,115 [INFO] src.synapse.synapse_cache: Cached 22 episodes (0 errors). Anchors: [22, 10, 5], Topo: [22, 8]
2026-04-26 16:45:39,116 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_164538_phase4_cross_domain/dataset_pusht/synapse_features/test.pt: 22 episodes
2026-04-26 16:45:39,116 [INFO] __main__: ═══ PUSHT × Recent Window ═══
2026-04-26 16:45:39,118 [INFO] src.data.dataset: RoboticsDataset(train): 164 episodes, 19407 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A1_recent
2026-04-26 16:45:39,118 [INFO] src.data.dataset: RoboticsDataset(val): 20 episodes, 2242 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A1_recent
2026-04-26 16:45:39,119 [INFO] src.data.dataset: RoboticsDataset(test): 22 episodes, 2353 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A1_recent
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
2026-04-26 16:45:39,241 [INFO] src.engine.train: Training device: cuda
2026-04-26 16:45:39,488 [INFO] src.engine.train: Created planner for condition=A1_recent, params=3178000
2026-04-26 16:45:40,452 [INFO] src.engine.train: Starting training: condition=A1_recent, max_epochs=200, patience=10
Epoch 1/200:   0% 0/151 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
2026-04-26 16:45:53,419 [INFO] src.engine.train: Epoch   0 | train_mse=1.007182 | val_mse=1.136714 | lr=3.02e-05 | 1610.3 samples/s
2026-04-26 16:45:53,419 [INFO] src.engine.train:   → New best val_mse=1.136714 (checkpoint saving disabled)
2026-04-26 16:46:06,333 [INFO] src.engine.train: Epoch   1 | train_mse=1.011982 | val_mse=1.136800 | lr=6.04e-05 | 1608.3 samples/s
2026-04-26 16:46:19,160 [INFO] src.engine.train: Epoch   2 | train_mse=1.023379 | val_mse=1.136663 | lr=9.06e-05 | 1620.2 samples/s
2026-04-26 16:46:19,160 [INFO] src.engine.train:   → New best val_mse=1.136663 (checkpoint saving disabled)
2026-04-26 16:46:32,029 [INFO] src.engine.train: Epoch   3 | train_mse=1.012308 | val_mse=1.136431 | lr=1.00e-04 | 1623.1 samples/s
2026-04-26 16:46:32,029 [INFO] src.engine.train:   → New best val_mse=1.136431 (checkpoint saving disabled)
2026-04-26 16:46:44,604 [INFO] src.engine.train: Epoch   4 | train_mse=1.027829 | val_mse=1.136452 | lr=1.00e-04 | 1652.0 samples/s
2026-04-26 16:46:57,101 [INFO] src.engine.train: Epoch   5 | train_mse=1.023375 | val_mse=1.136447 | lr=1.00e-04 | 1675.4 samples/s
2026-04-26 16:47:09,699 [INFO] src.engine.train: Epoch   6 | train_mse=1.011823 | val_mse=1.136392 | lr=9.99e-05 | 1673.3 samples/s
2026-04-26 16:47:09,700 [INFO] src.engine.train:   → New best val_mse=1.136392 (checkpoint saving disabled)
2026-04-26 16:47:22,611 [INFO] src.engine.train: Epoch   7 | train_mse=1.016119 | val_mse=1.136376 | lr=9.99e-05 | 1671.0 samples/s
2026-04-26 16:47:22,612 [INFO] src.engine.train:   → New best val_mse=1.136376 (checkpoint saving disabled)
2026-04-26 16:47:35,646 [INFO] src.engine.train: Epoch   8 | train_mse=1.023096 | val_mse=1.136431 | lr=9.98e-05 | 1660.4 samples/s
2026-04-26 16:47:48,608 [INFO] src.engine.train: Epoch   9 | train_mse=1.015933 | val_mse=1.136394 | lr=9.97e-05 | 1658.4 samples/s
2026-04-26 16:48:01,400 [INFO] src.engine.train: Epoch  10 | train_mse=1.024282 | val_mse=1.136292 | lr=9.96e-05 | 1656.2 samples/s
2026-04-26 16:48:01,400 [INFO] src.engine.train:   → New best val_mse=1.136292 (checkpoint saving disabled)
2026-04-26 16:48:14,647 [INFO] src.engine.train: Epoch  11 | train_mse=1.008042 | val_mse=1.136325 | lr=9.95e-05 | 1651.6 samples/s
2026-04-26 16:48:27,327 [INFO] src.engine.train: Epoch  12 | train_mse=1.010547 | val_mse=1.136441 | lr=9.94e-05 | 1660.9 samples/s
2026-04-26 16:48:40,237 [INFO] src.engine.train: Epoch  13 | train_mse=1.011001 | val_mse=1.136488 | lr=9.93e-05 | 1645.2 samples/s
2026-04-26 16:48:53,109 [INFO] src.engine.train: Epoch  14 | train_mse=1.018839 | val_mse=1.136499 | lr=9.91e-05 | 1638.4 samples/s
2026-04-26 16:49:06,121 [INFO] src.engine.train: Epoch  15 | train_mse=1.023671 | val_mse=1.136503 | lr=9.90e-05 | 1637.9 samples/s
2026-04-26 16:49:19,165 [INFO] src.engine.train: Epoch  16 | train_mse=1.008366 | val_mse=1.136564 | lr=9.88e-05 | 1641.5 samples/s
2026-04-26 16:49:32,303 [INFO] src.engine.train: Epoch  17 | train_mse=1.023228 | val_mse=1.136795 | lr=9.86e-05 | 1614.3 samples/s
2026-04-26 16:49:45,428 [INFO] src.engine.train: Epoch  18 | train_mse=1.018638 | val_mse=1.136731 | lr=9.84e-05 | 1627.0 samples/s
2026-04-26 16:49:58,374 [INFO] src.engine.train: Epoch  19 | train_mse=1.008357 | val_mse=1.136670 | lr=9.82e-05 | 1635.5 samples/s
2026-04-26 16:50:11,356 [INFO] src.engine.train: Epoch  20 | train_mse=1.011318 | val_mse=1.136659 | lr=9.80e-05 | 1638.3 samples/s
2026-04-26 16:50:11,357 [INFO] src.engine.train: Early stopping at epoch 20 (patience=10)
2026-04-26 16:50:11,357 [INFO] src.engine.train: Training complete: 21 epochs, 270.9s, best_val_mse=1.136292
2026-04-26 16:50:14,700 [INFO] src.engine.rollout: Rollout evaluation: 1 episodes, 10 steps, condition=A1_recent
2026-04-26 16:50:14,702 [INFO] __main__:   pusht × A1_recent: mean_mse=1.040466 ± 0.355653
2026-04-26 16:50:14,702 [INFO] __main__: ═══ PUSHT × Uniform Subsampling ═══
2026-04-26 16:50:14,707 [INFO] src.data.dataset: RoboticsDataset(train): 164 episodes, 19407 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A2_uniform
2026-04-26 16:50:14,708 [INFO] src.data.dataset: RoboticsDataset(val): 20 episodes, 2242 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A2_uniform
2026-04-26 16:50:14,708 [INFO] src.data.dataset: RoboticsDataset(test): 22 episodes, 2353 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A2_uniform
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
2026-04-26 16:50:14,714 [INFO] src.engine.train: Training device: cuda
2026-04-26 16:50:14,750 [INFO] src.engine.train: Created planner for condition=A2_uniform, params=3178000
2026-04-26 16:50:14,876 [INFO] src.engine.train: Starting training: condition=A2_uniform, max_epochs=200, patience=10
2026-04-26 16:50:27,846 [INFO] src.engine.train: Epoch   0 | train_mse=1.007306 | val_mse=1.136918 | lr=3.02e-05 | 1617.9 samples/s
2026-04-26 16:50:27,847 [INFO] src.engine.train:   → New best val_mse=1.136918 (checkpoint saving disabled)
2026-04-26 16:50:40,835 [INFO] src.engine.train: Epoch   1 | train_mse=1.012660 | val_mse=1.136991 | lr=6.04e-05 | 1654.5 samples/s
2026-04-26 16:50:53,419 [INFO] src.engine.train: Epoch   2 | train_mse=1.023491 | val_mse=1.136837 | lr=9.06e-05 | 1656.6 samples/s
2026-04-26 16:50:53,419 [INFO] src.engine.train:   → New best val_mse=1.136837 (checkpoint saving disabled)
2026-04-26 16:51:06,146 [INFO] src.engine.train: Epoch   3 | train_mse=1.012287 | val_mse=1.136802 | lr=1.00e-04 | 1646.4 samples/s
2026-04-26 16:51:06,146 [INFO] src.engine.train:   → New best val_mse=1.136802 (checkpoint saving disabled)
2026-04-26 16:51:19,560 [INFO] src.engine.train: Epoch   4 | train_mse=1.022337 | val_mse=1.136418 | lr=1.00e-04 | 1653.3 samples/s
2026-04-26 16:51:19,560 [INFO] src.engine.train:   → New best val_mse=1.136418 (checkpoint saving disabled)
2026-04-26 16:51:32,567 [INFO] src.engine.train: Epoch   5 | train_mse=1.022052 | val_mse=1.136442 | lr=1.00e-04 | 1653.5 samples/s
2026-04-26 16:51:45,290 [INFO] src.engine.train: Epoch   6 | train_mse=1.011917 | val_mse=1.136308 | lr=9.99e-05 | 1652.0 samples/s
2026-04-26 16:51:45,291 [INFO] src.engine.train:   → New best val_mse=1.136308 (checkpoint saving disabled)
Epoch 8/200:  48% 73/151 [00:05<00:06, 12.99it/s, loss=1.1062, mse=1.1062, grad=1.13, lr=9.99e-05]
