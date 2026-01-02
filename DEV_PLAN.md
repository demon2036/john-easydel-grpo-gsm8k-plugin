# 开发计划（最小侵入式插件）

## 目标
- 在独立 plugin 项目中实现 EasyDeL + Qwen3 1.7B 的 GSM8K GRPO 训练“最小测试”（总 step=10）
- 不修改 EasyDeL 原项目，只在 plugin 目录新增内容
- 通过 GitHub CLI 推送到你的 GitHub
- 使用 gcloud 新建 TPU v6e-8（eu-w-4a）并拉取插件项目运行

## 执行步骤
1. 在 `plugin/john_easydel_grpo_gsm8k` 创建独立工程（脚本、依赖、运行说明）
2. 编写 GRPO 训练脚本：
   - 模型：`Qwen/Qwen3-1.7B-Instruct`
   - 数据集：`gsm8k`（只保留 prompt 字段）
   - GRPO 配置：`max_training_steps=10`，小 batch/序列长度
   - 奖励函数：基于格式/数值输出的轻量奖励
3. 初始化 git 仓库并用 `gh` 创建/推送到你的 GitHub
4. 用 `gcloud` 创建 TPU v6e-8（eu-w-4a）
5. 通过 `gcloud compute tpus tpu-vm ssh --command` 拉取项目并运行脚本
6. 记录运行日志/关键输出路径
