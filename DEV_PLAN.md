# 开发计划（最小侵入式插件）

## 目标
- 在独立 plugin 项目中实现 EasyDeL + Qwen3 1.7B 的 GSM8K GRPO 训练最小测试（总 step=10）
- 不修改 EasyDeL 原项目，只在 `plugin/` 目录新增内容
- 用 GitHub CLI 推送插件项目到你的 GitHub
- 使用 gcloud 新建 TPU v6e-8（europe-west4-a），由 TPU 拉取项目运行

## 执行步骤
1. 维护独立插件工程（脚本、依赖、运行说明），不改动原项目。
2. 训练脚本配置：
   - 模型：`Qwen/Qwen3-1.7B`
   - 数据集：`gsm8k`
   - 训练步数：`MAX_TRAINING_STEPS=10`
   - 序列长度：`256/256`，attention 采用 `VANILLA`
   - 轻量奖励函数（格式/数值）
3. 用 `gh` 推送到 GitHub：
   - `git add . && git commit -m "plugin: grpo gsm8k tpu test"`
   - `gh repo create --source . --public --push`（如 repo 已存在则 `git push`）
4. 确认 v6e 运行时镜像版本：
   - `gcloud compute tpus tpu-vm versions list --zone=europe-west4-a`
   - 使用 `v2-alpha-tpuv6e`
5. 新建 TPU（spot）：
   - `gcloud alpha compute tpus tpu-vm create <TPU_NAME> --zone=europe-west4-a --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --spot`
6. TPU 拉取并运行：
   - `gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=europe-west4-a --command 'git clone <REPO> && cd <REPO> && ./scripts/run_on_tpu.sh'`
7. 验证输出：
   - 关注 `runs/grpo_gsm8k_test` 目录与训练日志。
