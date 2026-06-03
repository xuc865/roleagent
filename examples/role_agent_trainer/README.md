# Role-Agent training launch scripts

Hydra flags turn on **WIA** and **AIW** under `algorithm.role_agent.*`. Data paths default to `/mnt`; override with environment variables (see each script header).

| Script | Env | Algorithm | Notes |
|--------|-----|------------|--------|
| [`run_alfworld.sh`](run_alfworld.sh) | ALFWorld | PPO / GAE | `VERL_DATA_ROOT` → `text/*.parquet` |
| [`run_webshop.sh`](run_webshop.sh) | WebShop | PPO / GAE | Same data layout as ALFWorld text parquet |
| [`run_webshop_gigpo.sh`](run_webshop_gigpo.sh) | WebShop | GiGPO | Matches `examples/gigpo_trainer/run_webshop.sh` sizes (`train_data_size=16`, `env.rollout.n=8`) |
| [`run_search.sh`](run_search.sh) | search | GiGPO | `SEARCH_DATA_ROOT` for Search-R1 parquet |

**Baseline (no Role-Agent):** `examples/ppo_trainer/run_webshop.sh`, `examples/gigpo_trainer/run_webshop.sh`.

**Docs:** [`docs/role_agent_alignment.md`](../../docs/role_agent_alignment.md)

**Example (WebShop PPO + WIA + AIW):**

```bash
cd /path/to/roleagent
bash examples/role_agent_trainer/run_webshop.sh
```

**Example (WebShop GiGPO + WIA + AIW):**

```bash
export VERL_DATA_ROOT=/mnt/data/verl-agent
cd /path/to/roleagent
bash examples/role_agent_trainer/run_webshop_gigpo.sh
```
