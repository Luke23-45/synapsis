============================================================
🚀 Running EMP-05: train_deploy_consistency.py
============================================================
[INFO] [RunCapsule] Created: experiments\outputs\empirical\EMP-05\20260426_122857
[INFO] [EMP-05] Running seed 42 (1/3)
Seed set to 42
Seed set to 42
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be 
a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in 
the `DataLoader` to improve performance.
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
`Trainer.fit` stopped: `max_epochs=10` reached.
[INFO] [EMP-05] Saved raw NumPy artifacts to experiments\outputs\empirical\EMP-05\20260426_122857\artifacts\data\trials\EMP-05_seed42_artifacts.npz
[INFO] [EMP-05] seed=42  train_mse=0.0537  deploy_mse=0.1872  gap_rel=2.4832  overlap=0.5610  corr=0.1389
[INFO] [EMP-05] Seed 42 completed in 250.49s
[INFO] [EMP-05] Running seed 123 (2/3)
Seed set to 123
Seed set to 123
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be 
a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in 
the `DataLoader` to improve performance.
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
`Trainer.fit` stopped: `max_epochs=10` reached.
[INFO] [EMP-05] Saved raw NumPy artifacts to experiments\outputs\empirical\EMP-05\20260426_122857\artifacts\data\trials\EMP-05_seed123_artifacts.npz
[INFO] [EMP-05] seed=123  train_mse=0.0274  deploy_mse=0.0499  gap_rel=0.8194  overlap=0.6510  corr=0.1540
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
Seed set to 456
Seed set to 456
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\traine[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
Seed set to 456
Seed set to 456
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
Seed set to 456
Seed set to 456
GPU available: False, used: False
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
Seed set to 456
Seed set to 456
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
Seed set to 456
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
[INFO] [EMP-05] Seed 123 completed in 199.07s
[INFO] [EMP-05] Running seed 456 (3/3)
Seed set to 456
Seed set to 456
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be 
a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in 
the `DataLoader` to improve performance.
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
C:\Users\Hellx\AppData\Local\Programs\Python\Python310\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
`Trainer.fit` stopped: `max_epochs=10` reached.
[INFO] [EMP-05] Saved raw NumPy artifacts to experiments\outputs\empirical\EMP-05\20260426_122857\artifacts\data\trials\EMP-05_seed456_artifacts.npz
[INFO] [EMP-05] seed=456  train_mse=0.0507  deploy_mse=0.0701  gap_rel=0.3840  overlap=0.6963  corr=0.0563
[INFO] [EMP-05] Seed 456 completed in 226.15s
[INFO] [EMP-05] Results saved to experiments\outputs\empirical\EMP-05\20260426_122857\report.json[INFO] [EMP-05] gap_relative=1.2289  action_corr=0.1164  passed=False

EMP-05 PASSED: False

✅ EMP-05 COMPLETED SUCCESSFULLY