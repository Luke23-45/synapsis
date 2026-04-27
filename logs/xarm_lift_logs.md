2026-04-27 06:19:58,820 [INFO] __main__: Loaded config: baselines/configs/experiment/full.yaml
2026-04-27 06:19:58,827 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-27 06:19:58,827 [INFO] __main__: SYNAPSE Phase 4 Experiment: 1 datasets × 3 conditions
2026-04-27 06:19:58,827 [INFO] __main__:   Datasets: ['xarm_lift']
2026-04-27 06:19:58,827 [INFO] __main__:   Conditions: ['A1_recent', 'A2_uniform', 'B_synapse']
2026-04-27 06:19:58,827 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-27 06:19:58,827 [INFO] __main__: 
▓▓▓ DATASET: XARM_LIFT ▓▓▓

2026-04-27 06:19:58,828 [INFO] __main__: Loading dataset: xarm_lift (source=lerobot)
2026-04-27 06:19:58,890 [INFO] src.data.adapters.lerobot_adapter: Loading from local Parquet: /content/synapsis/baselines/data/datasets/xarm_lift/train.parquet
2026-04-27 06:19:59,021 [INFO] numexpr.utils: NumExpr defaulting to 2 threads.
2026-04-27 06:19:59,217 [INFO] src.data.adapters.lerobot_adapter: Loaded 20000 rows from Parquet
2026-04-27 06:19:59,303 [INFO] src.data.adapters.lerobot_adapter: LeRobotAdapter[xarm_lift]: loaded 800 episodes (proprio_dim=4, action_dim=4, phases=3)
2026-04-27 06:19:59,305 [INFO] __main__: Loaded 800 episodes from 'xarm_lift' (proprio_dim=4, action_dim=4)
2026-04-27 06:19:59,306 [INFO] src.data.dataset: Split 800 episodes: train=640, val=80, test=80
2026-04-27 06:19:59,308 [INFO] src.core.normalization: Computing normalization stats from 16000 timesteps × 4 dimensions
2026-04-27 06:19:59,310 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=4, mean_range=[0.1532, 1.5770], std_range=[0.0485, 0.3144])
2026-04-27 06:19:59,312 [INFO] src.core.normalization: Saved normalization stats to output/20260427_061958_phase4_cross_domain/dataset_xarm_lift/normalization_stats.pt
2026-04-27 06:19:59,313 [INFO] src.core.normalization: Computing normalization stats from 16000 timesteps × 4 dimensions
2026-04-27 06:19:59,314 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=4, mean_range=[-0.2422, 0.2762], std_range=[0.6305, 0.6687])
2026-04-27 06:19:59,315 [INFO] src.core.normalization: Saved normalization stats to output/20260427_061958_phase4_cross_domain/dataset_xarm_lift/action_normalization_stats.pt
2026-04-27 06:19:59,315 [INFO] __main__: ═══ XARM_LIFT × Recent Window ═══
2026-04-27 06:19:59,317 [INFO] src.data.dataset: RoboticsDataset(train): 640 episodes, 10880 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=A1_recent
2026-04-27 06:19:59,317 [INFO] src.data.dataset: RoboticsDataset(val): 80 episodes, 1360 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=A1_recent
2026-04-27 06:19:59,317 [INFO] src.data.dataset: RoboticsDataset(test): 80 episodes, 1360 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=A1_recent
2026-04-27 06:19:59,318 [INFO] src.data.dataset: Reducing DataLoader workers from 4 to 2 based on runtime CPU availability
2026-04-27 06:19:59,436 [INFO] src.engine.train: Training device: cuda
2026-04-27 06:19:59,672 [INFO] src.engine.train: Created planner for condition=A1_recent, params=3183136
2026-04-27 06:19:59,672 [INFO] src.engine.train: SYNAPSE pipeline: implementation=end_to_end, cached_features=False, auxiliary_losses=False
2026-04-27 06:20:00,701 [INFO] src.engine.train: Starting training: condition=A1_recent, max_epochs=80, patience=10
2026-04-27 06:20:07,851 [INFO] src.engine.train: Epoch   0 | train_total=1.005026 | train_mse=1.005026 | val_mse=0.994354 | lr=1.70e-05 | 1678.1 samples/s
2026-04-27 06:20:07,891 [INFO] src.engine.train:   → New best val_mse=0.994354 (checkpoint saving disabled)
2026-04-27 06:20:15,017 [INFO] src.engine.train: Epoch   1 | train_total=0.992756 | train_mse=0.992756 | val_mse=0.993257 | lr=3.40e-05 | 1705.0 samples/s
2026-04-27 06:20:15,047 [INFO] src.engine.train:   → New best val_mse=0.993257 (checkpoint saving disabled)
2026-04-27 06:20:21,782 [INFO] src.engine.train: Epoch   2 | train_total=0.975321 | train_mse=0.975321 | val_mse=0.991636 | lr=5.10e-05 | 1791.0 samples/s
2026-04-27 06:20:21,802 [INFO] src.engine.train:   → New best val_mse=0.991636 (checkpoint saving disabled)
2026-04-27 06:20:28,708 [INFO] src.engine.train: Epoch   3 | train_total=0.967775 | train_mse=0.967775 | val_mse=0.989719 | lr=6.80e-05 | 1709.9 samples/s
2026-04-27 06:20:28,728 [INFO] src.engine.train:   → New best val_mse=0.989719 (checkpoint saving disabled)
2026-04-27 06:20:35,425 [INFO] src.engine.train: Epoch   4 | train_total=0.965135 | train_mse=0.965135 | val_mse=0.987522 | lr=8.50e-05 | 1757.1 samples/s
2026-04-27 06:20:35,446 [INFO] src.engine.train:   → New best val_mse=0.987522 (checkpoint saving disabled)
2026-04-27 06:20:42,549 [INFO] src.engine.train: Epoch   5 | train_total=0.962418 | train_mse=0.962418 | val_mse=0.985000 | lr=1.00e-04 | 1663.0 samples/s
2026-04-27 06:20:42,569 [INFO] src.engine.train:   → New best val_mse=0.985000 (checkpoint saving disabled)
2026-04-27 06:20:49,612 [INFO] src.engine.train: Epoch   6 | train_total=0.962485 | train_mse=0.962485 | val_mse=0.982143 | lr=9.99e-05 | 1715.1 samples/s
2026-04-27 06:20:49,636 [INFO] src.engine.train:   → New best val_mse=0.982143 (checkpoint saving disabled)
2026-04-27 06:20:56,882 [INFO] src.engine.train: Epoch   7 | train_total=0.958675 | train_mse=0.958675 | val_mse=0.979026 | lr=9.98e-05 | 1633.1 samples/s
2026-04-27 06:20:56,903 [INFO] src.engine.train:   → New best val_mse=0.979026 (checkpoint saving disabled)
2026-04-27 06:21:04,141 [INFO] src.engine.train: Epoch   8 | train_total=0.956988 | train_mse=0.956988 | val_mse=0.975726 | lr=9.96e-05 | 1650.6 samples/s
2026-04-27 06:21:04,168 [INFO] src.engine.train:   → New best val_mse=0.975726 (checkpoint saving disabled)
2026-04-27 06:21:11,298 [INFO] src.engine.train: Epoch   9 | train_total=0.955567 | train_mse=0.955567 | val_mse=0.972253 | lr=9.92e-05 | 1657.6 samples/s
2026-04-27 06:21:11,325 [INFO] src.engine.train:   → New best val_mse=0.972253 (checkpoint saving disabled)
2026-04-27 06:21:18,714 [INFO] src.engine.train: Epoch  10 | train_total=0.952521 | train_mse=0.952521 | val_mse=0.968762 | lr=9.88e-05 | 1589.9 samples/s
2026-04-27 06:21:18,737 [INFO] src.engine.train:   → New best val_mse=0.968762 (checkpoint saving disabled)
2026-04-27 06:21:25,989 [INFO] src.engine.train: Epoch  11 | train_total=0.950729 | train_mse=0.950729 | val_mse=0.965386 | lr=9.83e-05 | 1618.0 samples/s
2026-04-27 06:21:26,011 [INFO] src.engine.train:   → New best val_mse=0.965386 (checkpoint saving disabled)
2026-04-27 06:21:33,634 [INFO] src.engine.train: Epoch  12 | train_total=0.948376 | train_mse=0.948376 | val_mse=0.962270 | lr=9.77e-05 | 1542.4 samples/s
2026-04-27 06:21:33,656 [INFO] src.engine.train:   → New best val_mse=0.962270 (checkpoint saving disabled)
2026-04-27 06:21:41,305 [INFO] src.engine.train: Epoch  13 | train_total=0.946098 | train_mse=0.946098 | val_mse=0.959445 | lr=9.71e-05 | 1580.6 samples/s
2026-04-27 06:21:41,328 [INFO] src.engine.train:   → New best val_mse=0.959445 (checkpoint saving disabled)
2026-04-27 06:21:49,092 [INFO] src.engine.train: Epoch  14 | train_total=0.943337 | train_mse=0.943337 | val_mse=0.956986 | lr=9.63e-05 | 1511.6 samples/s
2026-04-27 06:21:49,113 [INFO] src.engine.train:   → New best val_mse=0.956986 (checkpoint saving disabled)
2026-04-27 06:21:56,874 [INFO] src.engine.train: Epoch  15 | train_total=0.940967 | train_mse=0.940967 | val_mse=0.954944 | lr=9.55e-05 | 1533.2 samples/s
2026-04-27 06:21:56,896 [INFO] src.engine.train:   → New best val_mse=0.954944 (checkpoint saving disabled)
2026-04-27 06:22:04,376 [INFO] src.engine.train: Epoch  16 | train_total=0.938033 | train_mse=0.938033 | val_mse=0.953215 | lr=9.46e-05 | 1568.8 samples/s
2026-04-27 06:22:04,396 [INFO] src.engine.train:   → New best val_mse=0.953215 (checkpoint saving disabled)
2026-04-27 06:22:11,968 [INFO] src.engine.train: Epoch  17 | train_total=0.935336 | train_mse=0.935336 | val_mse=0.951770 | lr=9.35e-05 | 1562.5 samples/s
2026-04-27 06:22:11,990 [INFO] src.engine.train:   → New best val_mse=0.951770 (checkpoint saving disabled)
2026-04-27 06:22:19,206 [INFO] src.engine.train: Epoch  18 | train_total=0.933505 | train_mse=0.933505 | val_mse=0.950750 | lr=9.25e-05 | 1615.2 samples/s
2026-04-27 06:22:19,226 [INFO] src.engine.train:   → New best val_mse=0.950750 (checkpoint saving disabled)
2026-04-27 06:22:26,655 [INFO] src.engine.train: Epoch  19 | train_total=0.928141 | train_mse=0.928141 | val_mse=0.950001 | lr=9.13e-05 | 1588.5 samples/s
2026-04-27 06:22:26,678 [INFO] src.engine.train:   → New best val_mse=0.950001 (checkpoint saving disabled)
2026-04-27 06:22:34,048 [INFO] src.engine.train: Epoch  20 | train_total=0.926280 | train_mse=0.926280 | val_mse=0.949581 | lr=9.01e-05 | 1622.0 samples/s
2026-04-27 06:22:34,071 [INFO] src.engine.train:   → New best val_mse=0.949581 (checkpoint saving disabled)
2026-04-27 06:22:41,610 [INFO] src.engine.train: Epoch  21 | train_total=0.921008 | train_mse=0.921008 | val_mse=0.949402 | lr=8.88e-05 | 1573.1 samples/s
2026-04-27 06:22:41,631 [INFO] src.engine.train:   → New best val_mse=0.949402 (checkpoint saving disabled)
2026-04-27 06:22:49,099 [INFO] src.engine.train: Epoch  22 | train_total=0.917720 | train_mse=0.917720 | val_mse=0.949473 | lr=8.74e-05 | 1582.4 samples/s
2026-04-27 06:22:56,425 [INFO] src.engine.train: Epoch  23 | train_total=0.913080 | train_mse=0.913080 | val_mse=0.949778 | lr=8.60e-05 | 1594.2 samples/s
2026-04-27 06:23:03,975 [INFO] src.engine.train: Epoch  24 | train_total=0.909339 | train_mse=0.909339 | val_mse=0.950297 | lr=8.45e-05 | 1549.8 samples/s
2026-04-27 06:23:11,388 [INFO] src.engine.train: Epoch  25 | train_total=0.905943 | train_mse=0.905943 | val_mse=0.951082 | lr=8.29e-05 | 1586.8 samples/s
2026-04-27 06:23:19,023 [INFO] src.engine.train: Epoch  26 | train_total=0.900669 | train_mse=0.900669 | val_mse=0.952074 | lr=8.13e-05 | 1538.9 samples/s
2026-04-27 06:23:26,564 [INFO] src.engine.train: Epoch  27 | train_total=0.895782 | train_mse=0.895782 | val_mse=0.953321 | lr=7.96e-05 | 1586.2 samples/s
2026-04-27 06:23:34,148 [INFO] src.engine.train: Epoch  28 | train_total=0.891663 | train_mse=0.891663 | val_mse=0.954726 | lr=7.79e-05 | 1567.7 samples/s
2026-04-27 06:23:41,668 [INFO] src.engine.train: Epoch  29 | train_total=0.886965 | train_mse=0.886965 | val_mse=0.956292 | lr=7.61e-05 | 1562.5 samples/s
2026-04-27 06:23:48,990 [INFO] src.engine.train: Epoch  30 | train_total=0.881259 | train_mse=0.881259 | val_mse=0.958057 | lr=7.42e-05 | 1594.3 samples/s
2026-04-27 06:23:56,472 [INFO] src.engine.train: Epoch  31 | train_total=0.876017 | train_mse=0.876017 | val_mse=0.960051 | lr=7.24e-05 | 1567.3 samples/s
2026-04-27 06:23:56,472 [INFO] src.engine.train: Early stopping at epoch 31 (patience=10)
2026-04-27 06:23:56,478 [INFO] src.engine.train: Training complete: 32 epochs, 235.8s, best_val_mse=0.949402
2026-04-27 06:23:58,523 [INFO] src.engine.rollout: Rollout evaluation: 3 episodes, 10 steps, condition=A1_recent
2026-04-27 06:23:58,525 [INFO] __main__:   xarm_lift × A1_recent: mean_mse=0.950754 ± 0.141442
2026-04-27 06:23:58,525 [INFO] __main__: ═══ XARM_LIFT × Uniform Subsampling ═══
2026-04-27 06:23:58,530 [INFO] src.data.dataset: RoboticsDataset(train): 640 episodes, 10880 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=A2_uniform
2026-04-27 06:23:58,531 [INFO] src.data.dataset: RoboticsDataset(val): 80 episodes, 1360 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=A2_uniform
2026-04-27 06:23:58,531 [INFO] src.data.dataset: RoboticsDataset(test): 80 episodes, 1360 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=A2_uniform
2026-04-27 06:23:58,531 [INFO] src.data.dataset: Reducing DataLoader workers from 4 to 2 based on runtime CPU availability
2026-04-27 06:23:58,534 [INFO] src.engine.train: Training device: cuda
2026-04-27 06:23:58,571 [INFO] src.engine.train: Created planner for condition=A2_uniform, params=3183136
2026-04-27 06:23:58,571 [INFO] src.engine.train: SYNAPSE pipeline: implementation=end_to_end, cached_features=False, auxiliary_losses=False
2026-04-27 06:23:58,646 [INFO] src.engine.train: Starting training: condition=A2_uniform, max_epochs=80, patience=10
2026-04-27 06:24:07,808 [INFO] src.engine.train: Epoch   0 | train_total=1.004531 | train_mse=1.004531 | val_mse=0.993520 | lr=1.70e-05 | 1372.7 samples/s
2026-04-27 06:24:07,845 [INFO] src.engine.train:   → New best val_mse=0.993520 (checkpoint saving disabled)
2026-04-27 06:24:16,033 [INFO] src.engine.train: Epoch   1 | train_total=0.992046 | train_mse=0.992046 | val_mse=0.992502 | lr=3.40e-05 | 1432.1 samples/s
2026-04-27 06:24:16,058 [INFO] src.engine.train:   → New best val_mse=0.992502 (checkpoint saving disabled)
2026-04-27 06:24:24,598 [INFO] src.engine.train: Epoch   2 | train_total=0.975738 | train_mse=0.975738 | val_mse=0.990992 | lr=5.10e-05 | 1375.1 samples/s
2026-04-27 06:24:24,618 [INFO] src.engine.train:   → New best val_mse=0.990992 (checkpoint saving disabled)
2026-04-27 06:24:33,194 [INFO] src.engine.train: Epoch   3 | train_total=0.967487 | train_mse=0.967487 | val_mse=0.989161 | lr=6.80e-05 | 1374.3 samples/s
2026-04-27 06:24:33,214 [INFO] src.engine.train:   → New best val_mse=0.989161 (checkpoint saving disabled)
2026-04-27 06:24:41,500 [INFO] src.engine.train: Epoch   4 | train_total=0.964483 | train_mse=0.964483 | val_mse=0.987029 | lr=8.50e-05 | 1415.3 samples/s
2026-04-27 06:24:41,520 [INFO] src.engine.train:   → New best val_mse=0.987029 (checkpoint saving disabled)
2026-04-27 06:24:50,099 [INFO] src.engine.train: Epoch   5 | train_total=0.962374 | train_mse=0.962374 | val_mse=0.984526 | lr=1.00e-04 | 1372.8 samples/s
2026-04-27 06:24:50,121 [INFO] src.engine.train:   → New best val_mse=0.984526 (checkpoint saving disabled)
2026-04-27 06:24:58,756 [INFO] src.engine.train: Epoch   6 | train_total=0.962013 | train_mse=0.962013 | val_mse=0.981623 | lr=9.99e-05 | 1386.1 samples/s
2026-04-27 06:24:58,776 [INFO] src.engine.train:   → New best val_mse=0.981623 (checkpoint saving disabled)
2026-04-27 06:25:06,916 [INFO] src.engine.train: Epoch   7 | train_total=0.958400 | train_mse=0.958400 | val_mse=0.978425 | lr=9.98e-05 | 1453.7 samples/s
2026-04-27 06:25:06,941 [INFO] src.engine.train:   → New best val_mse=0.978425 (checkpoint saving disabled)
2026-04-27 06:25:15,508 [INFO] src.engine.train: Epoch   8 | train_total=0.956792 | train_mse=0.956792 | val_mse=0.975047 | lr=9.96e-05 | 1368.1 samples/s
2026-04-27 06:25:15,529 [INFO] src.engine.train:   → New best val_mse=0.975047 (checkpoint saving disabled)
2026-04-27 06:25:23,974 [INFO] src.engine.train: Epoch   9 | train_total=0.955053 | train_mse=0.955053 | val_mse=0.971551 | lr=9.92e-05 | 1422.0 samples/s
2026-04-27 06:25:23,999 [INFO] src.engine.train:   → New best val_mse=0.971551 (checkpoint saving disabled)
2026-04-27 06:25:32,356 [INFO] src.engine.train: Epoch  10 | train_total=0.952141 | train_mse=0.952141 | val_mse=0.968129 | lr=9.88e-05 | 1404.9 samples/s
2026-04-27 06:25:32,376 [INFO] src.engine.train:   → New best val_mse=0.968129 (checkpoint saving disabled)
2026-04-27 06:25:40,917 [INFO] src.engine.train: Epoch  11 | train_total=0.950888 | train_mse=0.950888 | val_mse=0.964861 | lr=9.83e-05 | 1372.8 samples/s
2026-04-27 06:25:40,937 [INFO] src.engine.train:   → New best val_mse=0.964861 (checkpoint saving disabled)
2026-04-27 06:25:49,439 [INFO] src.engine.train: Epoch  12 | train_total=0.949012 | train_mse=0.949012 | val_mse=0.961783 | lr=9.77e-05 | 1414.7 samples/s
2026-04-27 06:25:49,463 [INFO] src.engine.train:   → New best val_mse=0.961783 (checkpoint saving disabled)
2026-04-27 06:25:57,934 [INFO] src.engine.train: Epoch  13 | train_total=0.945817 | train_mse=0.945817 | val_mse=0.958927 | lr=9.71e-05 | 1405.6 samples/s
2026-04-27 06:25:57,956 [INFO] src.engine.train:   → New best val_mse=0.958927 (checkpoint saving disabled)
2026-04-27 06:26:06,500 [INFO] src.engine.train: Epoch  14 | train_total=0.943976 | train_mse=0.943976 | val_mse=0.956358 | lr=9.63e-05 | 1377.0 samples/s
2026-04-27 06:26:06,521 [INFO] src.engine.train:   → New best val_mse=0.956358 (checkpoint saving disabled)
2026-04-27 06:26:14,848 [INFO] src.engine.train: Epoch  15 | train_total=0.941738 | train_mse=0.941738 | val_mse=0.954173 | lr=9.55e-05 | 1432.0 samples/s
2026-04-27 06:26:14,870 [INFO] src.engine.train:   → New best val_mse=0.954173 (checkpoint saving disabled)
2026-04-27 06:26:23,350 [INFO] src.engine.train: Epoch  16 | train_total=0.938540 | train_mse=0.938540 | val_mse=0.952238 | lr=9.46e-05 | 1393.4 samples/s
2026-04-27 06:26:23,371 [INFO] src.engine.train:   → New best val_mse=0.952238 (checkpoint saving disabled)
2026-04-27 06:26:31,885 [INFO] src.engine.train: Epoch  17 | train_total=0.935573 | train_mse=0.935573 | val_mse=0.950563 | lr=9.35e-05 | 1384.7 samples/s
2026-04-27 06:26:31,905 [INFO] src.engine.train:   → New best val_mse=0.950563 (checkpoint saving disabled)
2026-04-27 06:26:40,117 [INFO] src.engine.train: Epoch  18 | train_total=0.933784 | train_mse=0.933784 | val_mse=0.949295 | lr=9.25e-05 | 1430.1 samples/s
2026-04-27 06:26:40,136 [INFO] src.engine.train:   → New best val_mse=0.949295 (checkpoint saving disabled)
2026-04-27 06:26:48,651 [INFO] src.engine.train: Epoch  19 | train_total=0.928343 | train_mse=0.928343 | val_mse=0.948289 | lr=9.13e-05 | 1384.8 samples/s
2026-04-27 06:26:48,671 [INFO] src.engine.train:   → New best val_mse=0.948289 (checkpoint saving disabled)
2026-04-27 06:26:57,131 [INFO] src.engine.train: Epoch  20 | train_total=0.926538 | train_mse=0.926538 | val_mse=0.947603 | lr=9.01e-05 | 1388.3 samples/s
2026-04-27 06:26:57,153 [INFO] src.engine.train:   → New best val_mse=0.947603 (checkpoint saving disabled)
2026-04-27 06:27:05,368 [INFO] src.engine.train: Epoch  21 | train_total=0.921203 | train_mse=0.921203 | val_mse=0.947225 | lr=8.88e-05 | 1426.0 samples/s
2026-04-27 06:27:05,388 [INFO] src.engine.train:   → New best val_mse=0.947225 (checkpoint saving disabled)
2026-04-27 06:27:13,968 [INFO] src.engine.train: Epoch  22 | train_total=0.918347 | train_mse=0.918347 | val_mse=0.947073 | lr=8.74e-05 | 1374.8 samples/s
2026-04-27 06:27:13,990 [INFO] src.engine.train:   → New best val_mse=0.947073 (checkpoint saving disabled)
2026-04-27 06:27:22,546 [INFO] src.engine.train: Epoch  23 | train_total=0.913853 | train_mse=0.913853 | val_mse=0.947284 | lr=8.60e-05 | 1393.0 samples/s
2026-04-27 06:27:30,796 [INFO] src.engine.train: Epoch  24 | train_total=0.909261 | train_mse=0.909261 | val_mse=0.947749 | lr=8.45e-05 | 1423.1 samples/s
2026-04-27 06:27:39,301 [INFO] src.engine.train: Epoch  25 | train_total=0.904307 | train_mse=0.904307 | val_mse=0.948439 | lr=8.29e-05 | 1381.9 samples/s
2026-04-27 06:27:47,855 [INFO] src.engine.train: Epoch  26 | train_total=0.901209 | train_mse=0.901209 | val_mse=0.949350 | lr=8.13e-05 | 1397.4 samples/s
2026-04-27 06:27:56,203 [INFO] src.engine.train: Epoch  27 | train_total=0.896037 | train_mse=0.896037 | val_mse=0.950572 | lr=7.96e-05 | 1407.7 samples/s
2026-04-27 06:28:04,647 [INFO] src.engine.train: Epoch  28 | train_total=0.891238 | train_mse=0.891238 | val_mse=0.951988 | lr=7.79e-05 | 1382.6 samples/s
2026-04-27 06:28:13,051 [INFO] src.engine.train: Epoch  29 | train_total=0.886910 | train_mse=0.886910 | val_mse=0.953664 | lr=7.61e-05 | 1425.0 samples/s
2026-04-27 06:28:21,591 [INFO] src.engine.train: Epoch  30 | train_total=0.881861 | train_mse=0.881861 | val_mse=0.955487 | lr=7.42e-05 | 1396.0 samples/s
2026-04-27 06:28:30,108 [INFO] src.engine.train: Epoch  31 | train_total=0.876917 | train_mse=0.876917 | val_mse=0.957589 | lr=7.24e-05 | 1377.0 samples/s
2026-04-27 06:28:38,545 [INFO] src.engine.train: Epoch  32 | train_total=0.873047 | train_mse=0.873047 | val_mse=0.959850 | lr=7.05e-05 | 1415.8 samples/s
2026-04-27 06:28:38,545 [INFO] src.engine.train: Early stopping at epoch 32 (patience=10)
2026-04-27 06:28:38,551 [INFO] src.engine.train: Training complete: 33 epochs, 279.9s, best_val_mse=0.947073
2026-04-27 06:28:41,844 [INFO] src.engine.rollout: Rollout evaluation: 3 episodes, 10 steps, condition=A2_uniform
2026-04-27 06:28:41,845 [INFO] __main__:   xarm_lift × A2_uniform: mean_mse=0.950672 ± 0.142731
2026-04-27 06:28:41,845 [INFO] __main__: ═══ XARM_LIFT × SYNAPSE Full ═══
2026-04-27 06:28:42,027 [INFO] src.data.dataset: RoboticsDataset(train): 640 episodes, 10880 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=B_synapse
2026-04-27 06:28:42,028 [INFO] src.data.dataset: RoboticsDataset(val): 80 episodes, 1360 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=B_synapse
2026-04-27 06:28:42,028 [INFO] src.data.dataset: RoboticsDataset(test): 80 episodes, 1360 timesteps, proprio_dim=4, action_dim=4, structured_dim=4, condition=B_synapse
2026-04-27 06:28:42,028 [INFO] src.data.dataset: Reducing DataLoader workers from 4 to 2 based on runtime CPU availability
2026-04-27 06:28:42,031 [INFO] src.engine.train: Training device: cuda
2026-04-27 06:28:42,070 [INFO] src.engine.train: Created planner for condition=B_synapse, params=3380674
2026-04-27 06:28:42,070 [INFO] src.engine.train: SYNAPSE pipeline: implementation=end_to_end, cached_features=False, auxiliary_losses=True
2026-04-27 06:28:42,160 [INFO] src.engine.train: Starting training: condition=B_synapse, max_epochs=80, patience=10
2026-04-27 06:28:51,266 [INFO] src.engine.train: Epoch   0 | train_total=1.316571 | train_mse=1.316571 | val_mse=1.635781 | train_sparse=0.138602 | train_topo=0.184962 | topo_std=0.065016 | alpha_sparse=0.000000 | alpha_topo=0.000000 | sparse_contrib=0.000000 | topo_contrib=0.000000 | lr=1.70e-05 | 1916.0 samples/s
2026-04-27 06:28:51,319 [INFO] src.engine.train:   → New best val_mse=1.635781 (checkpoint saving disabled)
2026-04-27 06:29:00,583 [INFO] src.engine.train: Epoch   1 | train_total=1.032307 | train_mse=1.032307 | val_mse=1.538075 | train_sparse=0.136454 | train_topo=0.185020 | topo_std=0.065030 | alpha_sparse=0.000000 | alpha_topo=0.000000 | sparse_contrib=0.000000 | topo_contrib=0.000000 | lr=3.40e-05 | 1789.6 samples/s
2026-04-27 06:29:00,609 [INFO] src.engine.train:   → New best val_mse=1.538075 (checkpoint saving disabled)
2026-04-27 06:29:10,145 [INFO] src.engine.train: Epoch   2 | train_total=1.009054 | train_mse=1.009054 | val_mse=1.441549 | train_sparse=0.140812 | train_topo=0.185364 | topo_std=0.064686 | alpha_sparse=0.000000 | alpha_topo=0.000000 | sparse_contrib=0.000000 | topo_contrib=0.000000 | lr=5.10e-05 | 1699.0 samples/s
2026-04-27 06:29:10,168 [INFO] src.engine.train:   → New best val_mse=1.441549 (checkpoint saving disabled)
2026-04-27 06:29:19,783 [INFO] src.engine.train: Epoch   3 | train_total=1.006487 | train_mse=1.000195 | val_mse=1.353014 | train_sparse=0.144142 | train_topo=0.179598 | topo_std=0.070407 | alpha_sparse=0.012500 | alpha_topo=0.025000 | sparse_contrib=0.001802 | topo_contrib=0.004490 | lr=6.80e-05 | 1878.8 samples/s
2026-04-27 06:29:19,807 [INFO] src.engine.train:   → New best val_mse=1.353014 (checkpoint saving disabled)
2026-04-27 06:29:28,370 [INFO] src.engine.train: Epoch   4 | train_total=1.004539 | train_mse=0.993303 | val_mse=1.276436 | train_sparse=0.132410 | train_topo=0.158520 | topo_std=0.091814 | alpha_sparse=0.025000 | alpha_topo=0.050000 | sparse_contrib=0.003310 | topo_contrib=0.007926 | lr=8.50e-05 | 1970.6 samples/s
2026-04-27 06:29:28,395 [INFO] src.engine.train:   → New best val_mse=1.276436 (checkpoint saving disabled)
2026-04-27 06:29:37,809 [INFO] src.engine.train: Epoch   5 | train_total=0.995639 | train_mse=0.983991 | val_mse=1.209893 | train_sparse=0.127232 | train_topo=0.091692 | topo_std=0.169301 | alpha_sparse=0.037500 | alpha_topo=0.075000 | sparse_contrib=0.004771 | topo_contrib=0.006877 | lr=1.00e-04 | 1729.3 samples/s
2026-04-27 06:29:37,836 [INFO] src.engine.train:   → New best val_mse=1.209893 (checkpoint saving disabled)
2026-04-27 06:29:47,380 [INFO] src.engine.train: Epoch   6 | train_total=0.991533 | train_mse=0.984563 | val_mse=1.154276 | train_sparse=0.116454 | train_topo=0.011479 | topo_std=0.297518 | alpha_sparse=0.050000 | alpha_topo=0.100000 | sparse_contrib=0.005823 | topo_contrib=0.001148 | lr=9.99e-05 | 1804.6 samples/s
2026-04-27 06:29:47,403 [INFO] src.engine.train:   → New best val_mse=1.154276 (checkpoint saving disabled)
2026-04-27 06:29:56,073 [INFO] src.engine.train: Epoch   7 | train_total=0.981232 | train_mse=0.975038 | val_mse=1.110574 | train_sparse=0.097159 | train_topo=0.000970 | topo_std=0.325689 | alpha_sparse=0.062500 | alpha_topo=0.125000 | sparse_contrib=0.006072 | topo_contrib=0.000121 | lr=9.98e-05 | 2047.4 samples/s
2026-04-27 06:29:56,100 [INFO] src.engine.train:   → New best val_mse=1.110574 (checkpoint saving disabled)
2026-04-27 06:30:04,937 [INFO] src.engine.train: Epoch   8 | train_total=0.976803 | train_mse=0.970178 | val_mse=1.076227 | train_sparse=0.085983 | train_topo=0.001180 | topo_std=0.334812 | alpha_sparse=0.075000 | alpha_topo=0.150000 | sparse_contrib=0.006449 | topo_contrib=0.000177 | lr=9.96e-05 | 1905.3 samples/s
2026-04-27 06:30:04,961 [INFO] src.engine.train:   → New best val_mse=1.076227 (checkpoint saving disabled)
2026-04-27 06:30:14,503 [INFO] src.engine.train: Epoch   9 | train_total=0.975581 | train_mse=0.968120 | val_mse=1.050646 | train_sparse=0.083710 | train_topo=0.000782 | topo_std=0.336788 | alpha_sparse=0.087500 | alpha_topo=0.175000 | sparse_contrib=0.007325 | topo_contrib=0.000137 | lr=9.92e-05 | 1720.3 samples/s
2026-04-27 06:30:14,527 [INFO] src.engine.train:   → New best val_mse=1.050646 (checkpoint saving disabled)
2026-04-27 06:30:23,418 [INFO] src.engine.train: Epoch  10 | train_total=0.973045 | train_mse=0.964808 | val_mse=1.032034 | train_sparse=0.080936 | train_topo=0.000716 | topo_std=0.333502 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.008094 | topo_contrib=0.000143 | lr=9.88e-05 | 2055.8 samples/s
2026-04-27 06:30:23,443 [INFO] src.engine.train:   → New best val_mse=1.032034 (checkpoint saving disabled)
2026-04-27 06:30:32,057 [INFO] src.engine.train: Epoch  11 | train_total=0.971385 | train_mse=0.963504 | val_mse=1.017202 | train_sparse=0.077321 | train_topo=0.000746 | topo_std=0.332123 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.007732 | topo_contrib=0.000149 | lr=9.83e-05 | 1993.5 samples/s
2026-04-27 06:30:32,081 [INFO] src.engine.train:   → New best val_mse=1.017202 (checkpoint saving disabled)
2026-04-27 06:30:41,307 [INFO] src.engine.train: Epoch  12 | train_total=0.969082 | train_mse=0.961343 | val_mse=1.005740 | train_sparse=0.075985 | train_topo=0.000700 | topo_std=0.334329 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.007598 | topo_contrib=0.000140 | lr=9.77e-05 | 1790.8 samples/s
2026-04-27 06:30:41,332 [INFO] src.engine.train:   → New best val_mse=1.005740 (checkpoint saving disabled)
2026-04-27 06:30:50,234 [INFO] src.engine.train: Epoch  13 | train_total=0.966091 | train_mse=0.958503 | val_mse=0.998484 | train_sparse=0.074450 | train_topo=0.000715 | topo_std=0.334651 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.007445 | topo_contrib=0.000143 | lr=9.71e-05 | 2124.3 samples/s
2026-04-27 06:30:50,256 [INFO] src.engine.train:   → New best val_mse=0.998484 (checkpoint saving disabled)
2026-04-27 06:30:58,480 [INFO] src.engine.train: Epoch  14 | train_total=0.964072 | train_mse=0.956659 | val_mse=0.991740 | train_sparse=0.072691 | train_topo=0.000720 | topo_std=0.334881 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.007269 | topo_contrib=0.000144 | lr=9.63e-05 | 2102.4 samples/s
2026-04-27 06:30:58,504 [INFO] src.engine.train:   → New best val_mse=0.991740 (checkpoint saving disabled)
2026-04-27 06:31:07,709 [INFO] src.engine.train: Epoch  15 | train_total=0.961669 | train_mse=0.954439 | val_mse=0.987463 | train_sparse=0.070843 | train_topo=0.000729 | topo_std=0.326907 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.007084 | topo_contrib=0.000146 | lr=9.55e-05 | 1776.4 samples/s
2026-04-27 06:31:07,732 [INFO] src.engine.train:   → New best val_mse=0.987463 (checkpoint saving disabled)
2026-04-27 06:31:16,735 [INFO] src.engine.train: Epoch  16 | train_total=0.960343 | train_mse=0.953201 | val_mse=0.983865 | train_sparse=0.070102 | train_topo=0.000659 | topo_std=0.330928 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.007010 | topo_contrib=0.000132 | lr=9.46e-05 | 2130.8 samples/s
2026-04-27 06:31:16,759 [INFO] src.engine.train:   → New best val_mse=0.983865 (checkpoint saving disabled)
2026-04-27 06:31:24,988 [INFO] src.engine.train: Epoch  17 | train_total=0.959103 | train_mse=0.952057 | val_mse=0.980771 | train_sparse=0.069104 | train_topo=0.000680 | topo_std=0.330607 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006910 | topo_contrib=0.000136 | lr=9.35e-05 | 2111.7 samples/s
2026-04-27 06:31:25,013 [INFO] src.engine.train:   → New best val_mse=0.980771 (checkpoint saving disabled)
2026-04-27 06:31:34,271 [INFO] src.engine.train: Epoch  18 | train_total=0.959087 | train_mse=0.952060 | val_mse=0.978653 | train_sparse=0.068752 | train_topo=0.000763 | topo_std=0.337615 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006875 | topo_contrib=0.000153 | lr=9.25e-05 | 1770.3 samples/s
2026-04-27 06:31:34,294 [INFO] src.engine.train:   → New best val_mse=0.978653 (checkpoint saving disabled)
2026-04-27 06:31:43,377 [INFO] src.engine.train: Epoch  19 | train_total=0.953928 | train_mse=0.947049 | val_mse=0.976201 | train_sparse=0.067407 | train_topo=0.000691 | topo_std=0.333847 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006741 | topo_contrib=0.000138 | lr=9.13e-05 | 2069.3 samples/s
2026-04-27 06:31:43,403 [INFO] src.engine.train:   → New best val_mse=0.976201 (checkpoint saving disabled)
2026-04-27 06:31:51,898 [INFO] src.engine.train: Epoch  20 | train_total=0.952867 | train_mse=0.946058 | val_mse=0.975219 | train_sparse=0.066700 | train_topo=0.000695 | topo_std=0.329688 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006670 | topo_contrib=0.000139 | lr=9.01e-05 | 2038.9 samples/s
2026-04-27 06:31:51,922 [INFO] src.engine.train:   → New best val_mse=0.975219 (checkpoint saving disabled)
2026-04-27 06:32:01,213 [INFO] src.engine.train: Epoch  21 | train_total=0.950761 | train_mse=0.944023 | val_mse=0.975173 | train_sparse=0.066064 | train_topo=0.000655 | topo_std=0.327183 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006606 | topo_contrib=0.000131 | lr=8.88e-05 | 1780.1 samples/s
2026-04-27 06:32:01,236 [INFO] src.engine.train:   → New best val_mse=0.975173 (checkpoint saving disabled)
2026-04-27 06:32:10,420 [INFO] src.engine.train: Epoch  22 | train_total=0.947761 | train_mse=0.941101 | val_mse=0.975649 | train_sparse=0.065360 | train_topo=0.000618 | topo_std=0.326608 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006536 | topo_contrib=0.000124 | lr=8.74e-05 | 1919.6 samples/s
2026-04-27 06:32:18,858 [INFO] src.engine.train: Epoch  23 | train_total=0.945238 | train_mse=0.938679 | val_mse=0.975354 | train_sparse=0.064219 | train_topo=0.000685 | topo_std=0.329504 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006422 | topo_contrib=0.000137 | lr=8.60e-05 | 2113.2 samples/s
2026-04-27 06:32:27,858 [INFO] src.engine.train: Epoch  24 | train_total=0.943146 | train_mse=0.936699 | val_mse=0.976079 | train_sparse=0.063159 | train_topo=0.000652 | topo_std=0.329527 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006316 | topo_contrib=0.000130 | lr=8.45e-05 | 1852.0 samples/s
2026-04-27 06:32:37,050 [INFO] src.engine.train: Epoch  25 | train_total=0.939795 | train_mse=0.933375 | val_mse=0.976467 | train_sparse=0.062685 | train_topo=0.000755 | topo_std=0.332422 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006268 | topo_contrib=0.000151 | lr=8.29e-05 | 1871.2 samples/s
2026-04-27 06:32:45,530 [INFO] src.engine.train: Epoch  26 | train_total=0.937363 | train_mse=0.930951 | val_mse=0.977368 | train_sparse=0.062701 | train_topo=0.000708 | topo_std=0.337776 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006270 | topo_contrib=0.000142 | lr=8.13e-05 | 2108.7 samples/s
2026-04-27 06:32:54,573 [INFO] src.engine.train: Epoch  27 | train_total=0.933774 | train_mse=0.927536 | val_mse=0.977977 | train_sparse=0.061111 | train_topo=0.000636 | topo_std=0.329527 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006111 | topo_contrib=0.000127 | lr=7.96e-05 | 1835.6 samples/s
2026-04-27 06:33:03,841 [INFO] src.engine.train: Epoch  28 | train_total=0.930015 | train_mse=0.923818 | val_mse=0.979117 | train_sparse=0.060534 | train_topo=0.000720 | topo_std=0.328676 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006053 | topo_contrib=0.000144 | lr=7.79e-05 | 1779.4 samples/s
2026-04-27 06:33:12,745 [INFO] src.engine.train: Epoch  29 | train_total=0.927962 | train_mse=0.921827 | val_mse=0.979772 | train_sparse=0.060052 | train_topo=0.000648 | topo_std=0.332514 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.006005 | topo_contrib=0.000130 | lr=7.61e-05 | 2036.9 samples/s
2026-04-27 06:33:21,267 [INFO] src.engine.train: Epoch  30 | train_total=0.922735 | train_mse=0.916691 | val_mse=0.980326 | train_sparse=0.058997 | train_topo=0.000726 | topo_std=0.331595 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.005900 | topo_contrib=0.000145 | lr=7.42e-05 | 2034.6 samples/s
2026-04-27 06:33:30,504 [INFO] src.engine.train: Epoch  31 | train_total=0.919512 | train_mse=0.913517 | val_mse=0.981181 | train_sparse=0.058425 | train_topo=0.000764 | topo_std=0.333065 | alpha_sparse=0.100000 | alpha_topo=0.200000 | sparse_contrib=0.005842 | topo_contrib=0.000153 | lr=7.24e-05 | 1781.9 samples/s
2026-04-27 06:33:30,504 [INFO] src.engine.train: Early stopping at epoch 31 (patience=10)
2026-04-27 06:33:30,511 [INFO] src.engine.train: Training complete: 32 epochs, 288.3s, best_val_mse=0.975173
2026-04-27 06:33:36,565 [INFO] src.engine.rollout: Rollout evaluation: 3 episodes, 10 steps, condition=B_synapse
2026-04-27 06:33:36,567 [INFO] __main__:   xarm_lift × B_synapse: mean_mse=0.976881 ± 0.155321
2026-04-27 06:33:36,710 [INFO] __main__: Dataset 'xarm_lift' complete in 817.9s (3 conditions)
2026-04-27 06:33:39,710 [INFO] src.reporting.visualize: Saved MSE comparison to output/20260427_061958_phase4_cross_domain/cross_domain/xarm_lift/xarm_lift_mse_comparison.pdf
2026-04-27 06:33:40,006 [INFO] src.reporting.visualize: Saved learning curves to output/20260427_061958_phase4_cross_domain/cross_domain/xarm_lift/xarm_lift_learning_curves.pdf
2026-04-27 06:33:40,006 [INFO] __main__: Cross-domain reports saved to output/20260427_061958_phase4_cross_domain/cross_domain
2026-04-27 06:33:40,006 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-27 06:33:40,006 [INFO] __main__: EXPERIMENT COMPLETE: 1 datasets, 817.9s total
2026-04-27 06:33:40,006 [INFO] __main__: Output directory: output/20260427_061958_phase4_cross_domain
2026-04-27 06:33:40,006 [INFO] __main__: ═══════════════════════════════════════════════════════════