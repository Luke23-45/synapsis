2026-04-26 17:20:34,074 [INFO] __main__: Loaded config: configs/experiment/full.yaml
2026-04-26 17:20:34,078 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-26 17:20:34,078 [INFO] __main__: SYNAPSE Phase 4 Experiment: 3 datasets × 3 conditions
2026-04-26 17:20:34,078 [INFO] __main__:   Datasets: ['pusht', 'aloha_transfer', 'xarm_lift']
2026-04-26 17:20:34,078 [INFO] __main__:   Conditions: ['A1_recent', 'A2_uniform', 'B_synapse']
2026-04-26 17:20:34,078 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-26 17:20:34,078 [INFO] __main__: 
▓▓▓ DATASET: PUSHT ▓▓▓

2026-04-26 17:20:34,078 [INFO] __main__: Loading dataset: pusht (source=lerobot)
2026-04-26 17:20:34,081 [INFO] __main__: Dataset 'pusht' is missing locally. Verifying/downloading into /content/synapsis/baselines/data/datasets.
2026-04-26 17:20:34,081 [ERROR] download_datasets: ❌ Dataset directory not found: /content/synapsis/baselines/data/datasets/pusht
2026-04-26 17:20:34,259 [INFO] numexpr.utils: NumExpr defaulting to 2 threads.
2026-04-26 17:20:34,627 [INFO] datasets: TensorFlow version 2.19.0 available.
2026-04-26 17:20:34,627 [INFO] datasets: JAX version 0.7.2 available.
2026-04-26 17:20:35,027 [INFO] download_datasets: Downloading 'pusht' from 'lerobot/pusht'...
2026-04-26 17:20:35,333 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/pusht/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-26 17:20:35,578 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/lerobot/pusht/7628202a2180972f291ba1bc6723834921e72c19/README.md "HTTP/1.1 200 OK"
2026-04-26 17:20:35,833 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/pusht/resolve/7628202a2180972f291ba1bc6723834921e72c19/pusht.py "HTTP/1.1 404 Not Found"
2026-04-26 17:20:36,525 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/lerobot/pusht/lerobot/pusht.py "HTTP/1.1 404 Not Found"
2026-04-26 17:20:36,779 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/pusht/revision/7628202a2180972f291ba1bc6723834921e72c19 "HTTP/1.1 200 OK"
2026-04-26 17:20:37,035 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/pusht/resolve/7628202a2180972f291ba1bc6723834921e72c19/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-26 17:20:37,327 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=lerobot/pusht "HTTP/1.1 200 OK"
2026-04-26 17:20:37,589 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/pusht/tree/7628202a2180972f291ba1bc6723834921e72c19/data?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-04-26 17:20:37,840 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/pusht/tree/7628202a2180972f291ba1bc6723834921e72c19?recursive=false&expand=false "HTTP/1.1 200 OK"
2026-04-26 17:20:38,097 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/pusht/resolve/7628202a2180972f291ba1bc6723834921e72c19/dataset_infos.json "HTTP/1.1 404 Not Found"
Creating parquet from Arrow format: 100% 26/26 [00:00<00:00, 1083.25ba/s]
2026-04-26 17:20:38,147 [INFO] download_datasets: Saved 25650 rows to /content/synapsis/baselines/data/datasets/pusht/train.parquet
2026-04-26 17:20:38,150 [INFO] download_datasets: ✅ Dataset 'pusht' downloaded successfully (25650 rows)
2026-04-26 17:20:38,161 [INFO] download_datasets: ✅ 'pusht': 25650 rows in Parquet (expected ~25650 frames)
2026-04-26 17:20:38,162 [INFO] src.data.adapters.lerobot_adapter: Loading from local Parquet: /content/synapsis/baselines/data/datasets/pusht/train.parquet
2026-04-26 17:20:38,186 [INFO] src.data.adapters.lerobot_adapter: Loaded 25650 rows from Parquet
2026-04-26 17:20:38,268 [INFO] src.data.adapters.lerobot_adapter: LeRobotAdapter[pusht]: loaded 206 episodes (proprio_dim=2, action_dim=2, phases=3)
2026-04-26 17:20:38,270 [INFO] __main__: Loaded 206 episodes from 'pusht' (proprio_dim=2, action_dim=2)
2026-04-26 17:20:38,270 [INFO] src.data.dataset: Split 206 episodes: train=164, val=20, test=22
2026-04-26 17:20:38,271 [INFO] src.core.normalization: Computing normalization stats from 20719 timesteps × 2 dimensions
2026-04-26 17:20:38,272 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=2, mean_range=[229.3067, 293.6066], std_range=[96.2621, 101.0113])
2026-04-26 17:20:38,275 [INFO] src.core.normalization: Saved normalization stats to output/20260426_172034_phase4_cross_domain/dataset_pusht/normalization_stats.pt
2026-04-26 17:20:38,275 [INFO] src.core.normalization: Computing normalization stats from 20719 timesteps × 2 dimensions
2026-04-26 17:20:38,276 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=2, mean_range=[228.4061, 294.3032], std_range=[95.7838, 100.6716])
2026-04-26 17:20:38,277 [INFO] src.core.normalization: Saved normalization stats to output/20260426_172034_phase4_cross_domain/dataset_pusht/action_normalization_stats.pt
2026-04-26 17:20:38,277 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 164 episodes → output/20260426_172034_phase4_cross_domain/dataset_pusht/synapse_features/train.pt
2026-04-26 17:20:38,286 [INFO] src.synapse.synapse_cache: Cached 164 episodes (0 errors). Anchors: [164, 10, 5], Topo: [164, 8]
2026-04-26 17:20:38,288 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_172034_phase4_cross_domain/dataset_pusht/synapse_features/train.pt: 164 episodes
2026-04-26 17:20:38,289 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 20 episodes → output/20260426_172034_phase4_cross_domain/dataset_pusht/synapse_features/val.pt
2026-04-26 17:20:38,291 [INFO] src.synapse.synapse_cache: Cached 20 episodes (0 errors). Anchors: [20, 10, 5], Topo: [20, 8]
2026-04-26 17:20:38,291 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_172034_phase4_cross_domain/dataset_pusht/synapse_features/val.pt: 20 episodes
2026-04-26 17:20:38,292 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 22 episodes → output/20260426_172034_phase4_cross_domain/dataset_pusht/synapse_features/test.pt
2026-04-26 17:20:38,293 [INFO] src.synapse.synapse_cache: Cached 22 episodes (0 errors). Anchors: [22, 10, 5], Topo: [22, 8]
2026-04-26 17:20:38,294 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_172034_phase4_cross_domain/dataset_pusht/synapse_features/test.pt: 22 episodes
2026-04-26 17:20:38,294 [INFO] __main__: ═══ PUSHT × Recent Window ═══
2026-04-26 17:20:38,297 [INFO] src.data.dataset: RoboticsDataset(train): 164 episodes, 19407 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A1_recent
2026-04-26 17:20:38,297 [INFO] src.data.dataset: RoboticsDataset(val): 20 episodes, 2242 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A1_recent
2026-04-26 17:20:38,297 [INFO] src.data.dataset: RoboticsDataset(test): 22 episodes, 2353 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A1_recent
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
2026-04-26 17:20:38,423 [INFO] src.engine.train: Training device: cuda
2026-04-26 17:20:38,670 [INFO] src.engine.train: Created planner for condition=A1_recent, params=3178000
2026-04-26 17:20:39,695 [INFO] src.engine.train: Starting training: condition=A1_recent, max_epochs=80, patience=10
Epoch 1/80:   0% 0/151 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
2026-04-26 17:20:52,618 [INFO] src.engine.train: Epoch   0 | train_mse=0.899962 | val_mse=1.128310 | lr=3.02e-05 | 1615.1 samples/s
2026-04-26 17:20:52,618 [INFO] src.engine.train:   → New best val_mse=1.128310 (checkpoint saving disabled)
2026-04-26 17:21:04,978 [INFO] src.engine.train: Epoch   1 | train_mse=0.335379 | val_mse=1.059721 | lr=6.04e-05 | 1684.0 samples/s
2026-04-26 17:21:04,978 [INFO] src.engine.train:   → New best val_mse=1.059721 (checkpoint saving disabled)
2026-04-26 17:21:17,619 [INFO] src.engine.train: Epoch   2 | train_mse=0.242809 | val_mse=0.967944 | lr=9.06e-05 | 1635.7 samples/s
2026-04-26 17:21:17,619 [INFO] src.engine.train:   → New best val_mse=0.967944 (checkpoint saving disabled)
2026-04-26 17:21:30,557 [INFO] src.engine.train: Epoch   3 | train_mse=0.211648 | val_mse=0.864991 | lr=1.00e-04 | 1611.8 samples/s
2026-04-26 17:21:30,557 [INFO] src.engine.train:   → New best val_mse=0.864991 (checkpoint saving disabled)
2026-04-26 17:21:43,357 [INFO] src.engine.train: Epoch   4 | train_mse=0.197470 | val_mse=0.753253 | lr=9.99e-05 | 1628.0 samples/s
2026-04-26 17:21:43,357 [INFO] src.engine.train:   → New best val_mse=0.753253 (checkpoint saving disabled)
2026-04-26 17:21:55,997 [INFO] src.engine.train: Epoch   5 | train_mse=0.182303 | val_mse=0.638925 | lr=9.97e-05 | 1647.0 samples/s
2026-04-26 17:21:55,997 [INFO] src.engine.train:   → New best val_mse=0.638925 (checkpoint saving disabled)
2026-04-26 17:22:08,564 [INFO] src.engine.train: Epoch   6 | train_mse=0.175920 | val_mse=0.530771 | lr=9.94e-05 | 1666.7 samples/s
2026-04-26 17:22:08,564 [INFO] src.engine.train:   → New best val_mse=0.530771 (checkpoint saving disabled)
2026-04-26 17:22:21,144 [INFO] src.engine.train: Epoch   7 | train_mse=0.168369 | val_mse=0.439101 | lr=9.91e-05 | 1669.6 samples/s
2026-04-26 17:22:21,144 [INFO] src.engine.train:   → New best val_mse=0.439101 (checkpoint saving disabled)
2026-04-26 17:22:33,730 [INFO] src.engine.train: Epoch   8 | train_mse=0.166488 | val_mse=0.364057 | lr=9.86e-05 | 1652.0 samples/s
2026-04-26 17:22:33,731 [INFO] src.engine.train:   → New best val_mse=0.364057 (checkpoint saving disabled)
2026-04-26 17:22:46,533 [INFO] src.engine.train: Epoch   9 | train_mse=0.158857 | val_mse=0.305735 | lr=9.81e-05 | 1635.9 samples/s
2026-04-26 17:22:46,533 [INFO] src.engine.train:   → New best val_mse=0.305735 (checkpoint saving disabled)
2026-04-26 17:22:59,201 [INFO] src.engine.train: Epoch  10 | train_mse=0.154897 | val_mse=0.262740 | lr=9.75e-05 | 1636.3 samples/s
2026-04-26 17:22:59,201 [INFO] src.engine.train:   → New best val_mse=0.262740 (checkpoint saving disabled)
2026-04-26 17:23:11,921 [INFO] src.engine.train: Epoch  11 | train_mse=0.146522 | val_mse=0.229984 | lr=9.69e-05 | 1645.9 samples/s
2026-04-26 17:23:11,921 [INFO] src.engine.train:   → New best val_mse=0.229984 (checkpoint saving disabled)
2026-04-26 17:23:24,570 [INFO] src.engine.train: Epoch  12 | train_mse=0.144636 | val_mse=0.205111 | lr=9.61e-05 | 1653.2 samples/s
2026-04-26 17:23:24,571 [INFO] src.engine.train:   → New best val_mse=0.205111 (checkpoint saving disabled)
2026-04-26 17:23:37,145 [INFO] src.engine.train: Epoch  13 | train_mse=0.141762 | val_mse=0.186432 | lr=9.53e-05 | 1656.6 samples/s
2026-04-26 17:23:37,145 [INFO] src.engine.train:   → New best val_mse=0.186432 (checkpoint saving disabled)
2026-04-26 17:23:49,932 [INFO] src.engine.train: Epoch  14 | train_mse=0.134145 | val_mse=0.171545 | lr=9.44e-05 | 1661.0 samples/s
2026-04-26 17:23:49,932 [INFO] src.engine.train:   → New best val_mse=0.171545 (checkpoint saving disabled)
2026-04-26 17:24:02,730 [INFO] src.engine.train: Epoch  15 | train_mse=0.134261 | val_mse=0.160438 | lr=9.34e-05 | 1658.5 samples/s
2026-04-26 17:24:02,730 [INFO] src.engine.train:   → New best val_mse=0.160438 (checkpoint saving disabled)
2026-04-26 17:24:15,554 [INFO] src.engine.train: Epoch  16 | train_mse=0.127948 | val_mse=0.151848 | lr=9.23e-05 | 1650.3 samples/s
2026-04-26 17:24:15,554 [INFO] src.engine.train:   → New best val_mse=0.151848 (checkpoint saving disabled)
2026-04-26 17:24:28,568 [INFO] src.engine.train: Epoch  17 | train_mse=0.126260 | val_mse=0.144982 | lr=9.12e-05 | 1636.5 samples/s
2026-04-26 17:24:28,569 [INFO] src.engine.train:   → New best val_mse=0.144982 (checkpoint saving disabled)
2026-04-26 17:24:41,655 [INFO] src.engine.train: Epoch  18 | train_mse=0.124825 | val_mse=0.139481 | lr=9.00e-05 | 1640.1 samples/s
2026-04-26 17:24:41,655 [INFO] src.engine.train:   → New best val_mse=0.139481 (checkpoint saving disabled)
2026-04-26 17:24:54,609 [INFO] src.engine.train: Epoch  19 | train_mse=0.120038 | val_mse=0.135136 | lr=8.88e-05 | 1622.1 samples/s
2026-04-26 17:24:54,609 [INFO] src.engine.train:   → New best val_mse=0.135136 (checkpoint saving disabled)
2026-04-26 17:25:07,678 [INFO] src.engine.train: Epoch  20 | train_mse=0.117390 | val_mse=0.131853 | lr=8.74e-05 | 1629.5 samples/s
2026-04-26 17:25:07,679 [INFO] src.engine.train:   → New best val_mse=0.131853 (checkpoint saving disabled)
2026-04-26 17:25:20,948 [INFO] src.engine.train: Epoch  21 | train_mse=0.114968 | val_mse=0.129418 | lr=8.60e-05 | 1657.0 samples/s
2026-04-26 17:25:20,948 [INFO] src.engine.train:   → New best val_mse=0.129418 (checkpoint saving disabled)
2026-04-26 17:25:33,996 [INFO] src.engine.train: Epoch  22 | train_mse=0.112480 | val_mse=0.127542 | lr=8.46e-05 | 1619.7 samples/s
2026-04-26 17:25:33,996 [INFO] src.engine.train:   → New best val_mse=0.127542 (checkpoint saving disabled)
2026-04-26 17:25:47,226 [INFO] src.engine.train: Epoch  23 | train_mse=0.107788 | val_mse=0.125858 | lr=8.31e-05 | 1639.5 samples/s
2026-04-26 17:25:47,226 [INFO] src.engine.train:   → New best val_mse=0.125858 (checkpoint saving disabled)
2026-04-26 17:26:00,343 [INFO] src.engine.train: Epoch  24 | train_mse=0.106477 | val_mse=0.124545 | lr=8.15e-05 | 1619.0 samples/s
2026-04-26 17:26:00,343 [INFO] src.engine.train:   → New best val_mse=0.124545 (checkpoint saving disabled)
2026-04-26 17:26:13,395 [INFO] src.engine.train: Epoch  25 | train_mse=0.104051 | val_mse=0.123666 | lr=7.99e-05 | 1633.0 samples/s
2026-04-26 17:26:13,395 [INFO] src.engine.train:   → New best val_mse=0.123666 (checkpoint saving disabled)
2026-04-26 17:26:26,397 [INFO] src.engine.train: Epoch  26 | train_mse=0.102329 | val_mse=0.123030 | lr=7.82e-05 | 1631.5 samples/s
2026-04-26 17:26:26,398 [INFO] src.engine.train:   → New best val_mse=0.123030 (checkpoint saving disabled)
2026-04-26 17:26:39,424 [INFO] src.engine.train: Epoch  27 | train_mse=0.099568 | val_mse=0.122740 | lr=7.65e-05 | 1638.2 samples/s
2026-04-26 17:26:39,424 [INFO] src.engine.train:   → New best val_mse=0.122740 (checkpoint saving disabled)
2026-04-26 17:26:52,429 [INFO] src.engine.train: Epoch  28 | train_mse=0.096590 | val_mse=0.122504 | lr=7.48e-05 | 1631.6 samples/s
2026-04-26 17:26:52,429 [INFO] src.engine.train:   → New best val_mse=0.122504 (checkpoint saving disabled)
2026-04-26 17:27:05,572 [INFO] src.engine.train: Epoch  29 | train_mse=0.095131 | val_mse=0.122737 | lr=7.30e-05 | 1627.2 samples/s
2026-04-26 17:27:18,738 [INFO] src.engine.train: Epoch  30 | train_mse=0.091024 | val_mse=0.122883 | lr=7.11e-05 | 1628.9 samples/s
2026-04-26 17:27:31,823 [INFO] src.engine.train: Epoch  31 | train_mse=0.089263 | val_mse=0.123152 | lr=6.93e-05 | 1625.3 samples/s
2026-04-26 17:27:44,906 [INFO] src.engine.train: Epoch  32 | train_mse=0.087764 | val_mse=0.123382 | lr=6.74e-05 | 1624.2 samples/s
2026-04-26 17:27:57,976 [INFO] src.engine.train: Epoch  33 | train_mse=0.084204 | val_mse=0.123568 | lr=6.54e-05 | 1620.8 samples/s
2026-04-26 17:28:10,993 [INFO] src.engine.train: Epoch  34 | train_mse=0.082357 | val_mse=0.123844 | lr=6.35e-05 | 1625.4 samples/s
2026-04-26 17:28:23,877 [INFO] src.engine.train: Epoch  35 | train_mse=0.079579 | val_mse=0.124124 | lr=6.15e-05 | 1626.6 samples/s
2026-04-26 17:28:36,948 [INFO] src.engine.train: Epoch  36 | train_mse=0.078910 | val_mse=0.124623 | lr=5.95e-05 | 1613.1 samples/s
2026-04-26 17:28:49,825 [INFO] src.engine.train: Epoch  37 | train_mse=0.077312 | val_mse=0.124973 | lr=5.75e-05 | 1615.4 samples/s
2026-04-26 17:29:02,488 [INFO] src.engine.train: Epoch  38 | train_mse=0.074896 | val_mse=0.125215 | lr=5.54e-05 | 1649.8 samples/s
2026-04-26 17:29:02,488 [INFO] src.engine.train: Early stopping at epoch 38 (patience=10)
2026-04-26 17:29:02,488 [INFO] src.engine.train: Training complete: 39 epochs, 502.8s, best_val_mse=0.122504
2026-04-26 17:29:05,503 [INFO] src.engine.rollout: Rollout evaluation: 1 episodes, 10 steps, condition=A1_recent
2026-04-26 17:29:05,505 [INFO] __main__:   pusht × A1_recent: mean_mse=0.144568 ± 0.061033
2026-04-26 17:29:05,505 [INFO] __main__: ═══ PUSHT × Uniform Subsampling ═══
2026-04-26 17:29:05,511 [INFO] src.data.dataset: RoboticsDataset(train): 164 episodes, 19407 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A2_uniform
2026-04-26 17:29:05,513 [INFO] src.data.dataset: RoboticsDataset(val): 20 episodes, 2242 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A2_uniform
2026-04-26 17:29:05,514 [INFO] src.data.dataset: RoboticsDataset(test): 22 episodes, 2353 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=A2_uniform
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
2026-04-26 17:29:05,520 [INFO] src.engine.train: Training device: cuda
2026-04-26 17:29:05,567 [INFO] src.engine.train: Created planner for condition=A2_uniform, params=3178000
2026-04-26 17:29:05,752 [INFO] src.engine.train: Starting training: condition=A2_uniform, max_epochs=80, patience=10
2026-04-26 17:29:22,413 [INFO] src.engine.train: Epoch   0 | train_mse=0.893218 | val_mse=1.126341 | lr=3.02e-05 | 1284.6 samples/s
2026-04-26 17:29:22,413 [INFO] src.engine.train:   → New best val_mse=1.126341 (checkpoint saving disabled)
2026-04-26 17:29:38,194 [INFO] src.engine.train: Epoch   1 | train_mse=0.315196 | val_mse=1.058360 | lr=6.04e-05 | 1321.9 samples/s
2026-04-26 17:29:38,194 [INFO] src.engine.train:   → New best val_mse=1.058360 (checkpoint saving disabled)
2026-04-26 17:29:53,953 [INFO] src.engine.train: Epoch   2 | train_mse=0.238381 | val_mse=0.957507 | lr=9.06e-05 | 1328.0 samples/s
2026-04-26 17:29:53,953 [INFO] src.engine.train:   → New best val_mse=0.957507 (checkpoint saving disabled)
2026-04-26 17:30:09,795 [INFO] src.engine.train: Epoch   3 | train_mse=0.214559 | val_mse=0.838799 | lr=1.00e-04 | 1320.2 samples/s
2026-04-26 17:30:09,795 [INFO] src.engine.train:   → New best val_mse=0.838799 (checkpoint saving disabled)
2026-04-26 17:30:26,216 [INFO] src.engine.train: Epoch   4 | train_mse=0.197659 | val_mse=0.712969 | lr=9.99e-05 | 1299.8 samples/s
2026-04-26 17:30:26,217 [INFO] src.engine.train:   → New best val_mse=0.712969 (checkpoint saving disabled)
2026-04-26 17:30:42,135 [INFO] src.engine.train: Epoch   5 | train_mse=0.187291 | val_mse=0.599082 | lr=9.97e-05 | 1310.8 samples/s
2026-04-26 17:30:42,135 [INFO] src.engine.train:   → New best val_mse=0.599082 (checkpoint saving disabled)
2026-04-26 17:30:58,243 [INFO] src.engine.train: Epoch   6 | train_mse=0.176110 | val_mse=0.501458 | lr=9.94e-05 | 1294.9 samples/s
2026-04-26 17:30:58,243 [INFO] src.engine.train:   → New best val_mse=0.501458 (checkpoint saving disabled)
2026-04-26 17:31:14,045 [INFO] src.engine.train: Epoch   7 | train_mse=0.172616 | val_mse=0.423686 | lr=9.91e-05 | 1331.0 samples/s
2026-04-26 17:31:14,045 [INFO] src.engine.train:   → New best val_mse=0.423686 (checkpoint saving disabled)
2026-04-26 17:31:29,944 [INFO] src.engine.train: Epoch   8 | train_mse=0.171922 | val_mse=0.361065 | lr=9.86e-05 | 1326.0 samples/s
2026-04-26 17:31:29,944 [INFO] src.engine.train:   → New best val_mse=0.361065 (checkpoint saving disabled)
2026-04-26 17:31:46,263 [INFO] src.engine.train: Epoch   9 | train_mse=0.162970 | val_mse=0.311760 | lr=9.81e-05 | 1298.8 samples/s
2026-04-26 17:31:46,263 [INFO] src.engine.train:   → New best val_mse=0.311760 (checkpoint saving disabled)
2026-04-26 17:32:02,190 [INFO] src.engine.train: Epoch  10 | train_mse=0.159546 | val_mse=0.274066 | lr=9.75e-05 | 1321.8 samples/s
2026-04-26 17:32:02,190 [INFO] src.engine.train:   → New best val_mse=0.274066 (checkpoint saving disabled)
2026-04-26 17:32:17,950 [INFO] src.engine.train: Epoch  11 | train_mse=0.151453 | val_mse=0.243678 | lr=9.69e-05 | 1317.1 samples/s
2026-04-26 17:32:17,950 [INFO] src.engine.train:   → New best val_mse=0.243678 (checkpoint saving disabled)
2026-04-26 17:32:33,715 [INFO] src.engine.train: Epoch  12 | train_mse=0.150504 | val_mse=0.218986 | lr=9.61e-05 | 1325.7 samples/s
2026-04-26 17:32:33,715 [INFO] src.engine.train:   → New best val_mse=0.218986 (checkpoint saving disabled)
2026-04-26 17:32:49,550 [INFO] src.engine.train: Epoch  13 | train_mse=0.144881 | val_mse=0.199400 | lr=9.53e-05 | 1324.6 samples/s
2026-04-26 17:32:49,551 [INFO] src.engine.train:   → New best val_mse=0.199400 (checkpoint saving disabled)
2026-04-26 17:33:05,764 [INFO] src.engine.train: Epoch  14 | train_mse=0.141635 | val_mse=0.182931 | lr=9.44e-05 | 1314.9 samples/s
2026-04-26 17:33:05,764 [INFO] src.engine.train:   → New best val_mse=0.182931 (checkpoint saving disabled)
2026-04-26 17:33:21,776 [INFO] src.engine.train: Epoch  15 | train_mse=0.141976 | val_mse=0.170906 | lr=9.34e-05 | 1315.6 samples/s
2026-04-26 17:33:21,776 [INFO] src.engine.train:   → New best val_mse=0.170906 (checkpoint saving disabled)
2026-04-26 17:33:37,629 [INFO] src.engine.train: Epoch  16 | train_mse=0.138167 | val_mse=0.161721 | lr=9.23e-05 | 1317.0 samples/s
2026-04-26 17:33:37,630 [INFO] src.engine.train:   → New best val_mse=0.161721 (checkpoint saving disabled)
2026-04-26 17:33:53,543 [INFO] src.engine.train: Epoch  17 | train_mse=0.133288 | val_mse=0.153884 | lr=9.12e-05 | 1312.8 samples/s
2026-04-26 17:33:53,544 [INFO] src.engine.train:   → New best val_mse=0.153884 (checkpoint saving disabled)
2026-04-26 17:34:09,358 [INFO] src.engine.train: Epoch  18 | train_mse=0.129472 | val_mse=0.147571 | lr=9.00e-05 | 1322.7 samples/s
2026-04-26 17:34:09,359 [INFO] src.engine.train:   → New best val_mse=0.147571 (checkpoint saving disabled)
2026-04-26 17:34:25,552 [INFO] src.engine.train: Epoch  19 | train_mse=0.126357 | val_mse=0.142121 | lr=8.88e-05 | 1313.7 samples/s
2026-04-26 17:34:25,552 [INFO] src.engine.train:   → New best val_mse=0.142121 (checkpoint saving disabled)
2026-04-26 17:34:41,754 [INFO] src.engine.train: Epoch  20 | train_mse=0.126487 | val_mse=0.138064 | lr=8.74e-05 | 1301.6 samples/s
2026-04-26 17:34:41,755 [INFO] src.engine.train:   → New best val_mse=0.138064 (checkpoint saving disabled)
2026-04-26 17:34:57,793 [INFO] src.engine.train: Epoch  21 | train_mse=0.121992 | val_mse=0.134756 | lr=8.60e-05 | 1300.9 samples/s
2026-04-26 17:34:57,793 [INFO] src.engine.train:   → New best val_mse=0.134756 (checkpoint saving disabled)
2026-04-26 17:35:13,703 [INFO] src.engine.train: Epoch  22 | train_mse=0.118983 | val_mse=0.132141 | lr=8.46e-05 | 1315.5 samples/s
2026-04-26 17:35:13,704 [INFO] src.engine.train:   → New best val_mse=0.132141 (checkpoint saving disabled)
2026-04-26 17:35:29,603 [INFO] src.engine.train: Epoch  23 | train_mse=0.115518 | val_mse=0.130023 | lr=8.31e-05 | 1313.8 samples/s
2026-04-26 17:35:29,604 [INFO] src.engine.train:   → New best val_mse=0.130023 (checkpoint saving disabled)
2026-04-26 17:35:46,033 [INFO] src.engine.train: Epoch  24 | train_mse=0.113863 | val_mse=0.128486 | lr=8.15e-05 | 1308.6 samples/s
2026-04-26 17:35:46,033 [INFO] src.engine.train:   → New best val_mse=0.128486 (checkpoint saving disabled)
2026-04-26 17:36:02,367 [INFO] src.engine.train: Epoch  25 | train_mse=0.111496 | val_mse=0.127469 | lr=7.99e-05 | 1286.5 samples/s
2026-04-26 17:36:02,367 [INFO] src.engine.train:   → New best val_mse=0.127469 (checkpoint saving disabled)
2026-04-26 17:36:18,306 [INFO] src.engine.train: Epoch  26 | train_mse=0.111166 | val_mse=0.126764 | lr=7.82e-05 | 1309.6 samples/s
2026-04-26 17:36:18,306 [INFO] src.engine.train:   → New best val_mse=0.126764 (checkpoint saving disabled)
2026-04-26 17:36:34,238 [INFO] src.engine.train: Epoch  27 | train_mse=0.108408 | val_mse=0.126599 | lr=7.65e-05 | 1312.8 samples/s
2026-04-26 17:36:34,238 [INFO] src.engine.train:   → New best val_mse=0.126599 (checkpoint saving disabled)
2026-04-26 17:36:50,137 [INFO] src.engine.train: Epoch  28 | train_mse=0.104983 | val_mse=0.126362 | lr=7.48e-05 | 1312.9 samples/s
2026-04-26 17:36:50,137 [INFO] src.engine.train:   → New best val_mse=0.126362 (checkpoint saving disabled)
2026-04-26 17:37:06,166 [INFO] src.engine.train: Epoch  29 | train_mse=0.101632 | val_mse=0.126743 | lr=7.30e-05 | 1324.7 samples/s
2026-04-26 17:37:22,581 [INFO] src.engine.train: Epoch  30 | train_mse=0.097827 | val_mse=0.126958 | lr=7.11e-05 | 1299.9 samples/s
2026-04-26 17:37:38,605 [INFO] src.engine.train: Epoch  31 | train_mse=0.095375 | val_mse=0.127486 | lr=6.93e-05 | 1313.9 samples/s
2026-04-26 17:37:54,668 [INFO] src.engine.train: Epoch  32 | train_mse=0.092875 | val_mse=0.128109 | lr=6.74e-05 | 1300.3 samples/s
2026-04-26 17:38:10,657 [INFO] src.engine.train: Epoch  33 | train_mse=0.089963 | val_mse=0.128756 | lr=6.54e-05 | 1306.6 samples/s
2026-04-26 17:38:26,873 [INFO] src.engine.train: Epoch  34 | train_mse=0.087957 | val_mse=0.129745 | lr=6.35e-05 | 1307.9 samples/s
2026-04-26 17:38:43,202 [INFO] src.engine.train: Epoch  35 | train_mse=0.084745 | val_mse=0.131093 | lr=6.15e-05 | 1288.6 samples/s
2026-04-26 17:38:59,224 [INFO] src.engine.train: Epoch  36 | train_mse=0.084247 | val_mse=0.131986 | lr=5.95e-05 | 1309.8 samples/s
2026-04-26 17:39:15,173 [INFO] src.engine.train: Epoch  37 | train_mse=0.082536 | val_mse=0.133156 | lr=5.75e-05 | 1311.5 samples/s
2026-04-26 17:39:31,106 [INFO] src.engine.train: Epoch  38 | train_mse=0.078462 | val_mse=0.134615 | lr=5.54e-05 | 1312.1 samples/s
2026-04-26 17:39:31,106 [INFO] src.engine.train: Early stopping at epoch 38 (patience=10)
2026-04-26 17:39:31,106 [INFO] src.engine.train: Training complete: 39 epochs, 625.4s, best_val_mse=0.126362
2026-04-26 17:39:34,849 [INFO] src.engine.rollout: Rollout evaluation: 1 episodes, 10 steps, condition=A2_uniform
2026-04-26 17:39:34,853 [INFO] __main__:   pusht × A2_uniform: mean_mse=0.144181 ± 0.051623
2026-04-26 17:39:34,853 [INFO] __main__: ═══ PUSHT × SYNAPSE Full ═══
2026-04-26 17:39:34,858 [INFO] src.data.dataset: RoboticsDataset(train): 164 episodes, 19407 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=B_synapse
2026-04-26 17:39:34,859 [INFO] src.data.dataset: RoboticsDataset(val): 20 episodes, 2242 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=B_synapse
2026-04-26 17:39:34,859 [INFO] src.data.dataset: RoboticsDataset(test): 22 episodes, 2353 timesteps, proprio_dim=2, action_dim=2, structured_dim=2, condition=B_synapse
2026-04-26 17:39:34,865 [INFO] src.engine.train: Training device: cuda
2026-04-26 17:39:34,910 [INFO] src.engine.train: Created planner for condition=B_synapse, params=3451570
2026-04-26 17:39:35,143 [INFO] src.engine.train: Starting training: condition=B_synapse, max_epochs=80, patience=10
2026-04-26 17:40:31,439 [INFO] src.engine.train: Epoch   0 | train_mse=1.182666 | val_mse=1.776950 | lr=3.02e-05 | 362.4 samples/s
2026-04-26 17:40:31,439 [INFO] src.engine.train:   → New best val_mse=1.776950 (checkpoint saving disabled)
2026-04-26 17:41:26,678 [INFO] src.engine.train: Epoch   1 | train_mse=0.531619 | val_mse=1.436555 | lr=6.04e-05 | 366.0 samples/s
2026-04-26 17:41:26,679 [INFO] src.engine.train:   → New best val_mse=1.436555 (checkpoint saving disabled)
2026-04-26 17:42:22,599 [INFO] src.engine.train: Epoch   2 | train_mse=0.419379 | val_mse=1.044055 | lr=9.06e-05 | 366.4 samples/s
2026-04-26 17:42:22,599 [INFO] src.engine.train:   → New best val_mse=1.044055 (checkpoint saving disabled)
2026-04-26 17:43:18,837 [INFO] src.engine.train: Epoch   3 | train_mse=0.248782 | val_mse=0.683103 | lr=1.00e-04 | 359.1 samples/s
2026-04-26 17:43:18,837 [INFO] src.engine.train:   → New best val_mse=0.683103 (checkpoint saving disabled)
2026-04-26 17:44:16,797 [INFO] src.engine.train: Epoch   4 | train_mse=0.227826 | val_mse=0.463215 | lr=9.99e-05 | 348.8 samples/s
2026-04-26 17:44:16,797 [INFO] src.engine.train:   → New best val_mse=0.463215 (checkpoint saving disabled)
2026-04-26 17:45:13,664 [INFO] src.engine.train: Epoch   5 | train_mse=0.214263 | val_mse=0.341045 | lr=9.97e-05 | 358.0 samples/s
2026-04-26 17:45:13,665 [INFO] src.engine.train:   → New best val_mse=0.341045 (checkpoint saving disabled)
2026-04-26 17:46:09,782 [INFO] src.engine.train: Epoch   6 | train_mse=0.213862 | val_mse=0.273636 | lr=9.94e-05 | 360.3 samples/s
2026-04-26 17:46:09,782 [INFO] src.engine.train:   → New best val_mse=0.273636 (checkpoint saving disabled)
2026-04-26 17:47:06,026 [INFO] src.engine.train: Epoch   7 | train_mse=0.209056 | val_mse=0.240048 | lr=9.91e-05 | 360.0 samples/s
2026-04-26 17:47:06,026 [INFO] src.engine.train:   → New best val_mse=0.240048 (checkpoint saving disabled)
2026-04-26 17:48:03,113 [INFO] src.engine.train: Epoch   8 | train_mse=0.209434 | val_mse=0.221726 | lr=9.86e-05 | 355.5 samples/s
2026-04-26 17:48:03,113 [INFO] src.engine.train:   → New best val_mse=0.221726 (checkpoint saving disabled)
2026-04-26 17:48:58,882 [INFO] src.engine.train: Epoch   9 | train_mse=0.205721 | val_mse=0.210853 | lr=9.81e-05 | 363.8 samples/s
2026-04-26 17:48:58,882 [INFO] src.engine.train:   → New best val_mse=0.210853 (checkpoint saving disabled)
2026-04-26 17:49:55,089 [INFO] src.engine.train: Epoch  10 | train_mse=0.200607 | val_mse=0.203995 | lr=9.75e-05 | 361.2 samples/s
2026-04-26 17:49:55,089 [INFO] src.engine.train:   → New best val_mse=0.203995 (checkpoint saving disabled)
2026-04-26 17:50:51,678 [INFO] src.engine.train: Epoch  11 | train_mse=0.195527 | val_mse=0.198333 | lr=9.69e-05 | 357.5 samples/s
2026-04-26 17:50:51,678 [INFO] src.engine.train:   → New best val_mse=0.198333 (checkpoint saving disabled)
2026-04-26 17:51:47,696 [INFO] src.engine.train: Epoch  12 | train_mse=0.203673 | val_mse=0.193440 | lr=9.61e-05 | 361.4 samples/s
2026-04-26 17:51:47,696 [INFO] src.engine.train:   → New best val_mse=0.193440 (checkpoint saving disabled)
2026-04-26 17:52:43,830 [INFO] src.engine.train: Epoch  13 | train_mse=0.196787 | val_mse=0.188347 | lr=9.53e-05 | 362.8 samples/s
2026-04-26 17:52:43,830 [INFO] src.engine.train:   → New best val_mse=0.188347 (checkpoint saving disabled)
2026-04-26 17:53:40,502 [INFO] src.engine.train: Epoch  14 | train_mse=0.186914 | val_mse=0.184214 | lr=9.44e-05 | 357.9 samples/s
2026-04-26 17:53:40,502 [INFO] src.engine.train:   → New best val_mse=0.184214 (checkpoint saving disabled)
2026-04-26 17:54:36,304 [INFO] src.engine.train: Epoch  15 | train_mse=0.191414 | val_mse=0.180626 | lr=9.34e-05 | 362.2 samples/s
2026-04-26 17:54:36,304 [INFO] src.engine.train:   → New best val_mse=0.180626 (checkpoint saving disabled)
2026-04-26 17:55:32,301 [INFO] src.engine.train: Epoch  16 | train_mse=0.178363 | val_mse=0.176346 | lr=9.23e-05 | 363.9 samples/s
2026-04-26 17:55:32,302 [INFO] src.engine.train:   → New best val_mse=0.176346 (checkpoint saving disabled)
2026-04-26 17:56:28,010 [INFO] src.engine.train: Epoch  17 | train_mse=0.181314 | val_mse=0.172359 | lr=9.12e-05 | 364.2 samples/s
2026-04-26 17:56:28,010 [INFO] src.engine.train:   → New best val_mse=0.172359 (checkpoint saving disabled)
2026-04-26 17:57:23,935 [INFO] src.engine.train: Epoch  18 | train_mse=0.175525 | val_mse=0.168572 | lr=9.00e-05 | 362.0 samples/s
2026-04-26 17:57:23,935 [INFO] src.engine.train:   → New best val_mse=0.168572 (checkpoint saving disabled)
2026-04-26 17:58:19,636 [INFO] src.engine.train: Epoch  19 | train_mse=0.173605 | val_mse=0.165742 | lr=8.88e-05 | 364.6 samples/s
2026-04-26 17:58:19,636 [INFO] src.engine.train:   → New best val_mse=0.165742 (checkpoint saving disabled)
2026-04-26 17:59:16,047 [INFO] src.engine.train: Epoch  20 | train_mse=0.171738 | val_mse=0.162854 | lr=8.74e-05 | 358.6 samples/s
2026-04-26 17:59:16,048 [INFO] src.engine.train:   → New best val_mse=0.162854 (checkpoint saving disabled)
2026-04-26 18:00:11,233 [INFO] src.engine.train: Epoch  21 | train_mse=0.170461 | val_mse=0.160188 | lr=8.60e-05 | 366.8 samples/s
2026-04-26 18:00:11,233 [INFO] src.engine.train:   → New best val_mse=0.160188 (checkpoint saving disabled)
2026-04-26 18:01:06,678 [INFO] src.engine.train: Epoch  22 | train_mse=0.165360 | val_mse=0.158664 | lr=8.46e-05 | 366.4 samples/s
2026-04-26 18:01:06,678 [INFO] src.engine.train:   → New best val_mse=0.158664 (checkpoint saving disabled)
2026-04-26 18:02:02,561 [INFO] src.engine.train: Epoch  23 | train_mse=0.163758 | val_mse=0.157323 | lr=8.31e-05 | 363.7 samples/s
2026-04-26 18:02:02,561 [INFO] src.engine.train:   → New best val_mse=0.157323 (checkpoint saving disabled)
2026-04-26 18:02:58,464 [INFO] src.engine.train: Epoch  24 | train_mse=0.163118 | val_mse=0.156255 | lr=8.15e-05 | 361.9 samples/s
2026-04-26 18:02:58,465 [INFO] src.engine.train:   → New best val_mse=0.156255 (checkpoint saving disabled)
2026-04-26 18:03:55,808 [INFO] src.engine.train: Epoch  25 | train_mse=0.161266 | val_mse=0.155199 | lr=7.99e-05 | 356.6 samples/s
2026-04-26 18:03:55,808 [INFO] src.engine.train:   → New best val_mse=0.155199 (checkpoint saving disabled)
2026-04-26 18:04:52,211 [INFO] src.engine.train: Epoch  26 | train_mse=0.160866 | val_mse=0.154540 | lr=7.82e-05 | 359.1 samples/s
2026-04-26 18:04:52,211 [INFO] src.engine.train:   → New best val_mse=0.154540 (checkpoint saving disabled)
2026-04-26 18:05:47,851 [INFO] src.engine.train: Epoch  27 | train_mse=0.155389 | val_mse=0.154238 | lr=7.65e-05 | 363.4 samples/s
2026-04-26 18:05:47,851 [INFO] src.engine.train:   → New best val_mse=0.154238 (checkpoint saving disabled)
2026-04-26 18:06:44,101 [INFO] src.engine.train: Epoch  28 | train_mse=0.155659 | val_mse=0.153528 | lr=7.48e-05 | 364.9 samples/s
2026-04-26 18:06:44,102 [INFO] src.engine.train:   → New best val_mse=0.153528 (checkpoint saving disabled)
2026-04-26 18:07:39,041 [INFO] src.engine.train: Epoch  29 | train_mse=0.156400 | val_mse=0.152909 | lr=7.30e-05 | 368.1 samples/s
2026-04-26 18:07:39,041 [INFO] src.engine.train:   → New best val_mse=0.152909 (checkpoint saving disabled)
2026-04-26 18:08:35,343 [INFO] src.engine.train: Epoch  30 | train_mse=0.159572 | val_mse=0.152327 | lr=7.11e-05 | 359.3 samples/s
2026-04-26 18:08:35,343 [INFO] src.engine.train:   → New best val_mse=0.152327 (checkpoint saving disabled)
2026-04-26 18:09:31,805 [INFO] src.engine.train: Epoch  31 | train_mse=0.156351 | val_mse=0.151950 | lr=6.93e-05 | 361.8 samples/s
2026-04-26 18:09:31,805 [INFO] src.engine.train:   → New best val_mse=0.151950 (checkpoint saving disabled)
2026-04-26 18:10:27,835 [INFO] src.engine.train: Epoch  32 | train_mse=0.157874 | val_mse=0.151276 | lr=6.74e-05 | 362.8 samples/s
2026-04-26 18:10:27,835 [INFO] src.engine.train:   → New best val_mse=0.151276 (checkpoint saving disabled)
2026-04-26 18:11:23,901 [INFO] src.engine.train: Epoch  33 | train_mse=0.157107 | val_mse=0.150412 | lr=6.54e-05 | 360.9 samples/s
2026-04-26 18:11:23,901 [INFO] src.engine.train:   → New best val_mse=0.150412 (checkpoint saving disabled)
2026-04-26 18:12:20,669 [INFO] src.engine.train: Epoch  34 | train_mse=0.154991 | val_mse=0.149052 | lr=6.35e-05 | 360.3 samples/s
2026-04-26 18:12:20,669 [INFO] src.engine.train:   → New best val_mse=0.149052 (checkpoint saving disabled)
2026-04-26 18:13:16,517 [INFO] src.engine.train: Epoch  35 | train_mse=0.152165 | val_mse=0.149642 | lr=6.15e-05 | 363.1 samples/s
2026-04-26 18:14:12,090 [INFO] src.engine.train: Epoch  36 | train_mse=0.151379 | val_mse=0.150269 | lr=5.95e-05 | 365.1 samples/s
2026-04-26 18:15:08,515 [INFO] src.engine.train: Epoch  37 | train_mse=0.150567 | val_mse=0.152400 | lr=5.75e-05 | 360.9 samples/s
2026-04-26 18:16:04,580 [INFO] src.engine.train: Epoch  38 | train_mse=0.148573 | val_mse=0.151079 | lr=5.54e-05 | 361.2 samples/s
2026-04-26 18:17:00,327 [INFO] src.engine.train: Epoch  39 | train_mse=0.147266 | val_mse=0.149781 | lr=5.34e-05 | 363.9 samples/s
2026-04-26 18:17:56,303 [INFO] src.engine.train: Epoch  40 | train_mse=0.146217 | val_mse=0.152030 | lr=5.13e-05 | 364.9 samples/s
2026-04-26 18:18:52,858 [INFO] src.engine.train: Epoch  41 | train_mse=0.146710 | val_mse=0.150189 | lr=4.93e-05 | 358.5 samples/s
2026-04-26 18:19:48,211 [INFO] src.engine.train: Epoch  42 | train_mse=0.144421 | val_mse=0.150499 | lr=4.72e-05 | 365.8 samples/s
2026-04-26 18:20:44,278 [INFO] src.engine.train: Epoch  43 | train_mse=0.141334 | val_mse=0.150979 | lr=4.52e-05 | 361.7 samples/s
2026-04-26 18:21:40,584 [INFO] src.engine.train: Epoch  44 | train_mse=0.144423 | val_mse=0.151838 | lr=4.32e-05 | 362.9 samples/s
2026-04-26 18:21:40,584 [INFO] src.engine.train: Early stopping at epoch 44 (patience=10)
2026-04-26 18:21:40,584 [INFO] src.engine.train: Training complete: 45 epochs, 2525.4s, best_val_mse=0.149052
2026-04-26 18:21:55,109 [ERROR] src.engine.evaluate: Rollout evaluation failed: 'structured_history'
2026-04-26 18:21:55,115 [INFO] __main__:   pusht × B_synapse: mean_mse=0.191813 ± 0.077314
2026-04-26 18:21:55,562 [INFO] __main__: Dataset 'pusht' complete in 3681.5s (3 conditions)
2026-04-26 18:21:55,562 [INFO] __main__: 
▓▓▓ DATASET: ALOHA_TRANSFER ▓▓▓

2026-04-26 18:21:55,563 [INFO] __main__: Loading dataset: aloha_transfer (source=lerobot)
2026-04-26 18:21:55,563 [INFO] __main__: Dataset 'aloha_transfer' is missing locally. Verifying/downloading into /content/synapsis/baselines/data/datasets.
2026-04-26 18:21:55,563 [ERROR] download_datasets: ❌ Dataset directory not found: /content/synapsis/baselines/data/datasets/aloha_transfer
2026-04-26 18:21:55,563 [INFO] download_datasets: Downloading 'aloha_transfer' from 'lerobot/aloha_sim_transfer_cube_human'...
2026-04-26 18:21:55,835 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-26 18:21:55,835 [WARNING] huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-26 18:21:55,842 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/lerobot/aloha_sim_transfer_cube_human/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/README.md "HTTP/1.1 200 OK"
2026-04-26 18:21:55,852 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/resolve-cache/datasets/lerobot/aloha_sim_transfer_cube_human/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/README.md "HTTP/1.1 200 OK"
README.md: 4.14kB [00:00, 12.5MB/s]
2026-04-26 18:21:56,103 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/aloha_sim_transfer_cube_human.py "HTTP/1.1 404 Not Found"
2026-04-26 18:21:56,794 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/lerobot/aloha_sim_transfer_cube_human/lerobot/aloha_sim_transfer_cube_human.py "HTTP/1.1 404 Not Found"
2026-04-26 18:21:57,049 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/aloha_sim_transfer_cube_human/revision/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc "HTTP/1.1 200 OK"
2026-04-26 18:21:57,303 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-26 18:21:57,573 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=lerobot/aloha_sim_transfer_cube_human "HTTP/1.1 200 OK"
2026-04-26 18:21:57,855 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/aloha_sim_transfer_cube_human/tree/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/data?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-04-26 18:21:58,114 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/aloha_sim_transfer_cube_human/tree/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc?recursive=false&expand=false "HTTP/1.1 200 OK"
2026-04-26 18:21:58,365 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/dataset_infos.json "HTTP/1.1 404 Not Found"
2026-04-26 18:21:58,624 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/data/chunk-000/file-000.parquet "HTTP/1.1 302 Found"
2026-04-26 18:21:58,881 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/lerobot/aloha_sim_transfer_cube_human/xet-read-token/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc "HTTP/1.1 200 OK"
data/chunk-000/file-000.parquet: 100% 1.04M/1.04M [00:01<00:00, 734kB/s] 
2026-04-26 18:22:00,553 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/data/chunk-000/file-001.parquet "HTTP/1.1 302 Found"
data/chunk-000/file-001.parquet: 100% 1.03M/1.03M [00:00<00:00, 1.68MB/s]
2026-04-26 18:22:01,428 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/resolve/ef49d5688771b07a3fc9bb49cb1a9fb3325b65dc/data/chunk-000/file-002.parquet "HTTP/1.1 302 Found"
data/chunk-000/file-002.parquet: 100% 187k/187k [00:00<00:00, 306kB/s]
Generating train split: 100% 20000/20000 [00:00<00:00, 963089.75 examples/s]
Creating parquet from Arrow format: 100% 20/20 [00:00<00:00, 433.01ba/s]
2026-04-26 18:22:02,117 [INFO] download_datasets: Saved 20000 rows to /content/synapsis/baselines/data/datasets/aloha_transfer/train.parquet
2026-04-26 18:22:02,119 [INFO] download_datasets: ✅ Dataset 'aloha_transfer' downloaded successfully (20000 rows)
2026-04-26 18:22:02,134 [INFO] download_datasets: ✅ 'aloha_transfer': 20000 rows in Parquet (expected ~20000 frames)
2026-04-26 18:22:02,134 [INFO] src.data.adapters.lerobot_adapter: Loading from local Parquet: /content/synapsis/baselines/data/datasets/aloha_transfer/train.parquet
2026-04-26 18:22:02,158 [INFO] src.data.adapters.lerobot_adapter: Loaded 20000 rows from Parquet
2026-04-26 18:22:02,219 [INFO] src.data.adapters.lerobot_adapter: LeRobotAdapter[aloha_transfer]: loaded 50 episodes (proprio_dim=14, action_dim=14, phases=4)
2026-04-26 18:22:02,222 [INFO] __main__: Loaded 50 episodes from 'aloha_transfer' (proprio_dim=14, action_dim=14)
2026-04-26 18:22:02,223 [INFO] src.data.dataset: Split 50 episodes: train=40, val=5, test=5
2026-04-26 18:22:02,224 [INFO] src.core.normalization: Computing normalization stats from 16000 timesteps × 14 dimensions
2026-04-26 18:22:02,228 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=14, mean_range=[-0.6347, 1.0347], std_range=[0.0124, 0.5274])
2026-04-26 18:22:02,229 [INFO] src.core.normalization: Saved normalization stats to output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/normalization_stats.pt
2026-04-26 18:22:02,230 [INFO] src.core.normalization: Computing normalization stats from 16000 timesteps × 14 dimensions
2026-04-26 18:22:02,232 [INFO] src.core.normalization: Normalization stats: NormalizationStats(dim=14, mean_range=[-0.6309, 1.0303], std_range=[0.0128, 0.5279])
2026-04-26 18:22:02,233 [INFO] src.core.normalization: Saved normalization stats to output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/action_normalization_stats.pt
2026-04-26 18:22:02,233 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 40 episodes → output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/synapse_features/train.pt
2026-04-26 18:22:02,241 [INFO] src.synapse.synapse_cache: Cached 40 episodes (0 errors). Anchors: [40, 10, 17], Topo: [40, 8]
2026-04-26 18:22:02,242 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/synapse_features/train.pt: 40 episodes
2026-04-26 18:22:02,243 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 5 episodes → output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/synapse_features/val.pt
2026-04-26 18:22:02,244 [INFO] src.synapse.synapse_cache: Cached 5 episodes (0 errors). Anchors: [5, 10, 17], Topo: [5, 8]
2026-04-26 18:22:02,245 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/synapse_features/val.pt: 5 episodes
2026-04-26 18:22:02,245 [INFO] src.synapse.synapse_cache: Caching SYNAPSE features for 5 episodes → output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/synapse_features/test.pt
2026-04-26 18:22:02,247 [INFO] src.synapse.synapse_cache: Cached 5 episodes (0 errors). Anchors: [5, 10, 17], Topo: [5, 8]
2026-04-26 18:22:02,247 [INFO] src.synapse.synapse_cache: Loaded cached features from output/20260426_172034_phase4_cross_domain/dataset_aloha_transfer/synapse_features/test.pt: 5 episodes
2026-04-26 18:22:02,248 [INFO] __main__: ═══ ALOHA_TRANSFER × Recent Window ═══
2026-04-26 18:22:02,250 [INFO] src.data.dataset: RoboticsDataset(train): 40 episodes, 15680 timesteps, proprio_dim=14, action_dim=14, structured_dim=14, condition=A1_recent
2026-04-26 18:22:02,250 [INFO] src.data.dataset: RoboticsDataset(val): 5 episodes, 1960 timesteps, proprio_dim=14, action_dim=14, structured_dim=14, condition=A1_recent
2026-04-26 18:22:02,251 [INFO] src.data.dataset: RoboticsDataset(test): 5 episodes, 1960 timesteps, proprio_dim=14, action_dim=14, structured_dim=14, condition=A1_recent
2026-04-26 18:22:02,253 [INFO] src.engine.train: Training device: cuda
2026-04-26 18:22:02,274 [INFO] src.engine.train: Created planner for condition=A1_recent, params=3208816
2026-04-26 18:22:02,276 [INFO] src.engine.train: Starting training: condition=A1_recent, max_epochs=80, patience=10
Epoch 1/80:   1% 2/245 [00:00<00:53,  4.52it/s]
