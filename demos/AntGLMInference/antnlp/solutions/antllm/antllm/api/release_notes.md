AntLLM Release说明

# 0.0.7
- 将发布的antllm作为独立的包，去掉 solutions.antllm 两级父目录，方便大家在 IDE 中使用编辑时有提示补全功能，所见即所得。
- BugFix。
  - 适配0930模型结束符变更；
  - SFT数据支持只上传训练语料，可以不传dev/test集；
  - atorch训练模型默认保存频次优化；

详情见[yuque](https://yuque.antfin.com/xr9fgo/ls4r52/gh2p6z4r6cvifbco?singleDoc#)

# 0.0.6
- 远程发起大模型蒸馏任务
- sft训练脚本全参数微调由默认调用deepspeed改为调用atorch
- SFT支持packed data，一般可提升训练速度7-8倍
- 默认使用公共资源池(gpudefault)的低保 A100 资源做任务排队
- 支持利用双卡a10部署10b模型（zark）

详情见[yuque](https://yuque.antfin.com/xr9fgo/ls4r52/dgka9b96loa6o8ib?singleDoc#)

# 0.0.5
- 远程训练的代码版本与 SDK 代码保持一致
- 支持训练和本地推理最长 32K Token 长文本
  - 模型自动部署还没有支持 32K 长窗口参数，所以 RemoteCompletion 还没支持长 Tokens 参数。
- 支持高效微调 PEFT 模型热更新 
- 支持 AIStudio 统计提交的 antllm api 任务数

** 注意：由于远程执行的代码都会切到使用 antllm SDK来执行，以前依赖代码库的方式运行会失败，所以对于 antllm<=0.0.4 的版本，需要重新安装更新新的 SDK 版本才可以使用。

详情见[yuque](https://yuque.antfin.com/xr9fgo/ls4r52/ivn3x6so59gkte0s?singleDoc#)

# 0.0.4
定位更加工具属性，支持的主要功能有：
- 支持全量参数微调的SFT
- 支持 PEFT 高效微调的具体参数配置
- 支持全新底座模型部署

详情见[yuque](https://yuque.antfin.com/xr9fgo/ls4r52/kiv0qu035g1p1nvv?singleDoc#)

# 0.0.3
第一次正式发布版本，支持的主要功能有：
- 高效微调 SFT（本地或远程训练）
- 基于已有底座服务的微调参数模型部署
- 模型服务调用（本地或远程）
- 蚂蚁基座大模型的 embedding 获取

详情见[ATA](https://ata.alibaba-inc.com/articles/270270/)
