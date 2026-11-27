# 从 1Cat-vLLM 的 SM70 工作里可以抄什么

对照对象：`/Volumes/SMB重要/1Cat-vLLM` —— vLLM 的一个分支，把 Qwen4 架构（它叫
`qwen4_exp`，即本仓库的 `qwen4exp` / Qwen3.8-Flash-Next）、GLM-5.3-Flash、DeepSeek-V4 跑到了
sm_70（V100-SXM2 16/32 GB）上。它和 Legion 面对的是同一批模型、同一代硬件，所以结论有直接可比性。

**可信度**：下文所有数字都是该分支自己实验文档里的实测值（TP4 V100-SXM2、CUDA 12.8、MTP4 或
DFlash2），**我们未在 AC922 上复验**。它的 `docs/design/` 有 288 篇，本次读了其中 12 篇
`flashnext_*`、两篇根目录质量日志、两个 V100 启动脚本及其 profile JSON、README 的 V100 段落，
以及模型包与 `csrc/` 的 SM70 代码。长尾未读见文末。

---

## 一、总体判断

1. **真正的数量级收益只有一条**，而且和 sm_70 缺什么指令无关：把压缩后的 QSA indexer 历史留在
   GPU、只让主 KV 历史进 host（32K 下 65 → 152 tok/s）。
2. 其余是叠加式的 5%–15%，且**必须成对带着它的质量门数字看**：HCX 只在 M≤8 有效、FP8 target KV
   有 -2.86pp acceptance 代价、KV 整体进 host 有 +5% 延迟代价。
3. **默认开 = 已被验证**。该分支把不少做出来的东西留在默认关闭（`sm70_hcx`、
   `sm70_hcx_local_schedule`、`sm70_qsa_prep`、`qsa_host_kv`、`sm70_gdn_verify`），等模型级质量
   放行。抄的时候要把「做到了」和「敢默认开」区分开。

---

## 二、第一梯队：数据放置与批处理边界

不需要为 Volta 写任何指令，收益最大。Legion 已有 host KV 分层与 TurboQuant，缺的是这几条决策。

| 优化 | 实测 | 开关 / 出处 |
|---|---|---|
| **压缩后的 indexer 历史留在 GPU**（只有主 KV 历史进 host）+ PLE 通道分块 | 端到端 32K/64K/128K **65.03/40.24/26.27 → 152.15/139.99/110.49 tok/s（2.3–4.2×）**；算子 32K 0.0187→0.763 ms、261K 0.058→5.865 ms（host 远差）；单次 prefill 峰值 **-68 MiB** | `QSA_HOST_INDEXER_HISTORY=false`；`docs/design/flashnext_indexer_placement_20261015.md:22-26,33-38,53,184` |
| **host-KV 整段 staging**（`prefill_stage_slots=1`）+ 打分 tile 及时释放 | 同条件 128K 绕行 **345.94 s → 114.02 s（≈3×）**；冷 prefill @34688 4881→5126→**5196.69 tok/s** | `qsa_host_kv_prefill_stage_slots` / `..._stage_tokens=262144`；同上 `:150-169,186-188,288` |
| **device hot cache 用 E4M3 编码并扩到 32768 token** | 32K 整轮 **21.256/18.710 → 15.482–15.494 ms**（目标 <16 ms；p90 17.340、p99 31.689）；**1024→520 B/token**，容量翻倍只多 3 MiB/rank；churn 场景算子 +17~20% | `qsa_host_kv_hot_dtype=fp8_e4m3`、`hot_tokens=32768`、`..._draft_hot_tokens`；`docs/design/flashnext_decode_verifier_16gb_20261014.md:39-42,301-317,345-354,359` |
| **合格 batch / packed 权重**（MTP、GDN、HC、router、shared-expert 共 14 项默认 ON） | C1 纯解码 **79.832 → 92.199 tok/s（+15.49%）**，C4 +1.05%（判为无回退）；21 条成对请求 token **完全一致**；MBPP 12/12、32K 检索 3/3、中文 QA 6/6 | 代价 GDN 725.6 + HC 330 + router 122.5 + shared 76.6 MiB/rank，`VLLM_SM70_QWEN38_BATCH_FASTPATH=0` 等 5 个可逐个关；`docs/design/sm70_flash_next_batch_defaults.md:36-46,101-113` |
| **Grouped Page4 稀疏 attention + indexed W13 + SwiGLU 融合** | **55.151 → 9.632 ms/layer/rank（5.5×）**；模型纯 prefill 32K/64K/131K **+32.36/29.93/32.69%**；算子链 6.0267→4.2353 ms，纯 prefill 5998.65→6507.10 tok/s | `README.md:597-621,631-646` |
| **QSA 共享键打分核 + 设备直读 E4M3 历史**（6 个 Q head 共用的 key 只解一次，去掉 hot-page 解析与 miss staging） | **-0.4039 / -0.6902 / 合并 -1.0930 ms/round**，C4 复测 -0.4815；质量门通过：top-1 63/64、mean KL 5.2e-4、acceptance Δ -0.025pp CI[-1.79,+2.09] | `sm70_qsa_prep`、`sm70_qsa_device_history`（合格 target 默认开）；`flashnext_qsa_router_structure.md:558,585-592`、`flashnext_device_qsa_history.md:44-46,102` |
| **长上下文精确 top-k**（半精度 key 缩桶 → FP32 定枢轴 → 按 index 递增压缩，逐位一致） | 20 行×65,688 分 **191.376 → 108.288 µs（-43%）**；32,844 分 105.304→79.984 µs | **无开关**，`topk==512 && cc family 70` 即默认路径（`vllm/models/qwen4_exp/nvidia/ops/qsa.py:1353`）；`csrc/qsa_lexicographic_topk.cuh` |

`qsa_lexicographic_topk` 的做法本身值得单独记：4 趟字节 radix **只求 score pivot**（旧实现把
32 bit index 的 tie-break 也一起 radix，要 8 趟），最后一趟按 index 递增压缩，用 packed 64-bit
`BlockScan` 同时统计 greater/equal，取前 `remaining` 个 tie —— 天然得到「低 index 优先 + 规范累加
序」。候选存储上界 2304，无持久 workspace。这是**正确性与速度双收益，且与架构无关**。

---

## 三、第二梯队：需要写内核

### 3.1 TurboMind GEMM + shape 缓存 autotuner（资产量最大）

`csrc/sm70_turbomind/` 是从 lmdeploy 移植的 GEMM 引擎（`sm70_884_{4,8,16}.cu` 三档 m8n8k4 内核 +
`kernels/gemm/` 的模板 dispatch），其上 **77 个 V100 算子**：AWQ / FP8 / NVFP4 / MXFP4、GGUF dp4a
与 IQ3 / IQ4_XS 原生反量化、dual-weight matvec、hcx、qsa、mtp、router、philox top20 采样。
带 `dispatch_cache` + `measurer` + `stopping_criterion` 的运行期 autotune（`TM_GEMM_TUNE` /
`TM_GEMM_TRACE`）。

**它不是研究代码**：`VLLM_SM70_QUANT_BACKEND=auto` 且 `VLLM_SM70_{AWQ,FP8,NVFP4,MXFP4}_TURBOMIND`
默认 True，即 V100 默认 serving 的量化 GEMM 就走这条路；`VLLM_SM70_AWQ_WARMUP=True` 在 CUDA graph
capture 之前把已选中的路由预热（`..._MAX_M=16`）。直接对应本仓库「小 M GGUF matmul 只覆盖 2–8
token、IQ 系回退 MMVQ/MMQ」的缺口。

编译门控也值得抄一条纪律：`CMakeLists.txt:393/517/1434` 对这些路径用
`cuda_archs_loose_intersection(... "7.0" ...)` —— **严格 7.0**，不是 7.5；运行期另有
`props->major==7 && minor==0` 硬检查加「grid 必须全驻留」的 occupancy 校验。

### 3.2 GDN 的 SIMT 寄存器态配方（对 Legion 最省力，见 §7 更正）

`flash_qla/` 是 Qwen 团队自带的 linear-attention 库（`LICENSE` 写 `Copyright (c) 2026 Qwen,
Alibaba`），vendored 进仓库，含 `hopper/` 与 `sm70/` 两条后端；`setup.py:253-294` 把 sm70 内核预编
进 wheel，运行时不需要 NVCC。

`sm70/csrc/gdn_forward.cu` 的关键事实：**全文 grep `mma.sync|wmma|ldmatrix|cp.async` 零命中，且完全
没有 `__shared__`** —— 逐 token 递推 + 状态常驻寄存器的纯 SIMT。可抄的配方：

- **分片**：K=V=128 硬要求；`cols=4, width=16`，block=(32, column_groups)，一个 warp = 2 个 16-lane
  subgroup，**每 lane 持 32 个 FP32 寄存器（128 B 状态/lane）**；更新与查询复用同一批寄存器，只有
  lane0 写输出；K 方向 `__shfl_down_sync` 树归约，delta 在 lane0 算完再 `__shfl_sync` 广播。
  **零共享内存、零 `__syncthreads`。**
- **靠 occupancy 而非 ILP**：`column_groups_per_block` 是纯 occupancy 旋钮，按
  `(tokens, q_heads, v_heads)` 查表（Hv=32/Hq=16 短序列给 4、长序列给 2），
  `FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK` 可覆盖。
- **精度**：全程 FP32 `fmaf`，只在 load/store 转 dtype；`final_state` 恒 FP32。注释明写「V100 没有
  原生 bf16，默认从 fp16 生产路径推导」。
- **verify 的正确切法**（`csrc/sm70_turbomind/ops/gdn_verify_sm70.cu:12-19` 把对 Triton 的取舍写清楚
  了）：Triton 每 (V-tile 32, head) 一个 program，HV=12 时**只有 48 个 program** 且循环内做 global
  load；改成 **1 CTA 拥有 16 行状态**（4 warp × 4 行），token 的 q/k/v 与 gate 标量**先全部预载**，
  T 步链条纯寄存器算术，状态写回 fire-and-forget `float4`，`TMAX=8` → **96+ CTA**。树状投机用
  `parent_ids + num_accepted_tokens` 做状态回滚，不是重放。
- **分工铁律**：`qwen38_gdn_input_batch_sm70.cu` 用 m8n8k4 HMMA，但那是 GDN 的 **input
  projection**。**投影走 tensor core，delta-rule 递推永远走 SIMT。**

实测（`docs/design/sm70_gdn_prefill_threads.md`，4×V100-SXM2-16GB @1312MHz）：prefill 从 128 线程
改 256 线程，7344 行 **1.578→1.079 ms**、8192 行 **1.749→1.127 ms**、6108 行 1.317→0.857 ms、
6528 行 1.392→0.908 ms（**≈1.45–1.55×**）；TP4 Qwen3.8-27B NVFP4+MTP4、256K 下 5941 prompt tok/s，
route trace 中 GDN 省约 63 ms。**但默认仍是 128 线程，256 只是 candidate**，未经完整模型质量放行。

### 3.3 HC 融进 all-reduce

`csrc/sm70_qwen38_hc_batch.cuh` + `vllm/models/qwen4_exp/nvidia/sm70_hcx.py`：产出投影跳过
all-reduce，一个 kernel 做完 all-reduce + combine + 分组 RMSNorm + HC down/SiLU + HC up/gate-mix。

- **权重预先按 mma fragment 的 lane 布局排好**（`_LANE_R=(L&3)+4*(L&16)`、`_LANE_Q=(L>>2)&3`），核内
  普通 LDG 即可喂进 MMA，**不需要 ldmatrix、不需要共享内存 swizzle** —— 绕开的正是 Volta 没有
  ldmatrix 这个限制。
- `PairRows` 用一次权重载入喂两个独立 M8 累加器；`FuseMix` 的 sigmoid 门控混合用 `__shfl_sync` 在
  quad 内完成；`div.full.f32` 而非 fast-math，为了复现 Triton 的数值。
- 通信走 CUDA IPC + peer 原子；非全连接 mesh 用 `find_peer_order` 两跳序，pynvml 逐对探测 NVLink P2P。
- `pack_output_projection` 会把 GGUF Q8_0/Q4_K/Q5_K/Q6_K 的行重排列，直接吃掉
  `input_to_gguf` 变换。

实测：**C1 18.3192 → 17.3988 ms/round（-5.02%）**，target replay 14.3949→13.4769，tokens/round
4.8857 不变，acceptance 45.86% → 47.33%（`flashnext_mtp4_round12.md:466-472,485-489`）。
local schedule 算子级 M1 -12.8% / M5 -10.9% / M8 -8.5%，但**模型级 A/B 未被承认**（ABBA/BAAB
给 -0.0242/-0.7708，fresh-process A/A 自身 KL 就不达标）→ 默认 false。

### 3.4 长 prefill 注意力

三条路线，默认走 `sm70_79t`（`VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=True`，
`..._GQA_V37=False` 留作回滚）；`FLASH_ATTN_V100` 在 sm_70/72 上由 `vllm/platforms/cuda.py:168-177`
自动置顶，不需要显式指定。

`sm70_79t` 的数值配方值得整段抄：**stride-8 采样求 score max + 固定 margin / 指数上限**、居中、
**value residual 按 2 的幂缩放留 64× 余量**、FP16 操作数 + FP32 MMA 累加、每 24 个 K block 落一次
FP32 再在线合并；`128×256` threadblock + `64×64` warp tile 的唯一理由是**避开寄存器溢出**。
**>75 TFLOPS @KV128K，对比它替换掉的 Split-D 路线 47**。早先「零位移 exp + 无保护 FP16 累加」在
真实输入上会溢出。

`sm70_v37/` 是 prefill-only、D256/Hq6/Hkv1/batch1 的 M128-N256-K32 FP32-累加实现：前缀非因果 +
当前 chunk 精确掩码；概率与最终输出仍是 FP16，中间全 FP32，只做最后一次 FP16 转换；E4M3 bridge
只展开字节，不假装还原化掉的 KV 精度。

另一条务实参照：它**不是完全重写内核**，`cmake/patches/` 有 5 个 `sm70_flash_attn_d256_*.patch`，由
`cmake/external_projects/vllm_flash_attn.cmake:63-75` 打到 vendored 的 vllm-flash-attention 上，给
sm70 加 D256 GQA 的 pipeline / split-KV / K-pingpong / arch 路径。

### 3.5 一条几乎白捡的通信 trick

`csrc/sm70_ring_collectives.cu`：把两个 FP32 部分和压进**一条 64 位原子 store** —— 利用「FP16 部分
和的指数必落在 [-24,18]」重编号成 6 bit code，凑成 `2×23 bit 值 + 2 bit generation tag`，自带版本、
无需 flag 往返；双缓冲按 `epoch&1` 分代；按 logical_rank 的比特位固定加法顺序以保证各 rank 结合律
一致（bit-exact）。约 100 行、零依赖，只需 P2P native atomics 探测。标注 research-only。

### 3.6 Marlin 的 sm_70 后端

自写 CUTLASS 主循环 `Sm70MarlinMmaPipelined` 替代只支持 sm_80+ 的上游 Marlin；按 U4 / U4-b8 / U8 /
U8-b128 / FP8 / NVFP4 / MXFP4 **拆成独立 TU 并行编译**，`sm70_marlin_dispatch.cu` 统一分派（内含
lane→frag 的 32 项逆置换表）；`SM70_MARLIN_{DENSE,MOE}_{CTA_GEOMETRY,SPLIT_K}` 可**免重编译切换
tile**。只在 Legion 要支持这些 dense/MoE 量化格式时才有价值。

---

## 四、第三梯队：工程与工具（便宜，可立刻做）

- **`q=1` 的 FULL graph replay 之前必须刷新 `block_table` / `seq_lens`。** 那一大批「backend / 长度 /
  LM-head A/B 全部无效、明确写着不要再跑」的实验，真因就是这个
  （`SM70_FLASH_V100_QUALITY_EXPERIMENT_LOG_20260615.md:52-66,138-151`）。正确性优先。
- **mixed-prefill 自适应预算**：常驻解码 **6.88 → 30.34 tok/s**，最长更新间隙 **2.873 → 0.310 s**。
- **共享权重与 graph scratch 去重**：模型分配 **10.07 → 6.47 GiB**、KV 预算 10.57 → 12.68 GiB；TP4
  非 KV 部分 **10.225 → 8.794 GiB**。
- **host lane 三原则**：写回用 **copy-engine DMA，不要用 SM zero-copy store**（后者 throttle 整机显存
  管线，`csrc/gguf_shared_expert_dma.h` 明写）；host 线程只做规划（每层 routed expert id 进
  host-mapped ring → per-layer LRU 与 slot 规划），**拷贝由 model stream 自己发**，避免 release 排在
  同一条 CUDA 队列后面；PLE 走 host-mapped buffer + 一个原生线程，路径上无 GIL / socket / CUDA。
- **静态路由台账**：`tools/sm70/path_inventory.py`（**不需要 GPU** 就能列出源码里每条路由的准入条件）、
  `op_parity` / `route_parity`、`check_sm70_release_artifact.py`（校验发布 wheel 自包含）。Legion 的
  架构分派散在 `.cu` launcher 和 `per_device` 里，这类静态台账比再加内核划算。
- **集中策略表**：`csrc/sm70_policy.h` + `sm70_policy_fields.inc` 把 69 个开关编码成一个自描述
  token，launch 前解析冻结、**CUDA graph 里不留任何策略地址**。实现（`torch::CustomClassHolder`、
  TLS `PolicyScope`）别搬，只搬这条纪律。反面教材：`vllm/envs.py` 里 **250+ 个 `VLLM_SM70_*`** 加约
  20 个 `VLLM_FLASH_V100_*`。

---

## 五、明确别抄（他们试过并回退）

| 项 | 结果 | 出处 |
|---|---|---|
| target KV 用 FP8 E4M3（vs FP16） | acceptance **-2.86pp**，未获默认推广 | `flashnext_host_fp8_kv.md:104` |
| KV 历史整体放 host | 省 1.229 GiB/rank，但 C1 **+0.471 ms**、C4 **+2.148 ms（~5%）** | 同上 `:113,246` |
| native GDN verifier | C1 17.406→**17.613**、C4 45.025→**46.229**（M20 1286→1724 µs） | `flashnext_device_qsa_history.md:185` |
| whole-verifier 改 PIECEWISE 图 | **35.62 / 34.89 tok/s vs 68–91** | `SM70_MTP_OUTPUT_QUALITY_AUDIT_20260616.md:165` |
| token-matching 采样修 MTP 崩坏 | P / Q / R 三组均失败 | 同上 `:147-149` |
| HCX 早期集成（未修 M>8 分派） | C1 -0.294 但 C4 43.264→**46.588（+7.7%）**，不予晋升 | `flashnext_mtp4_round12.md:149-153` |
| owned-payload HC 边界 | C4 43.211→**45.884** | 同上 `:303-312` |
| K-shard HCX / row-ready HC / per-CTA LoRA 轮询 | 23.7→**43.9 / 77.2 µs**；27.208→28.396 µs；23.735→45.797 µs | 同上 `:501-504`；`flashnext_hcx_local_schedule.md:746-747,764-770` |
| 静态单 slot mix | 指令数 -13% 但 21.025→21.113 µs | 同上 |
| coalesced N32/K8 标量码布局（IQ3_S M5） | 45.056→**52.224 µs** | `flashnext_mtp4_round12.md:506-510` |
| HC→dense split-K | 46.688→**58.660 µs** | `flashnext_persistent_execution.md:403-404` |
| draft 单图 | C1 18.273→18.600、C4 43.277→43.467 | `flashnext_mtp4_round12.md:194-197` |
| 64-bit key + Torch top-k 终合并 | 更慢 | `flashnext_indexer_placement_20261015.md:361-362` |
| CPU 亲和 / 隔离对 prefill | 4987.2→4982.5 tok/s，无收益 | `flashnext_abliterated_9683_20261014.md:82-85` |
| IQ3_S 各 LUT / prefetch 微调 | ~0.066 ms/round，"解释不了 6.6 ms 缺口"，保持 disable | `flashnext_mtp4_round12.md:335-337` |

**功耗也算成本**：同一颗 GPU 在 185 W 上限下 attention 只有 **53.816 T vs 71.111 T**
（`README.md:106`）。AC922 上跑长 prefill 时这条值得记住。

## 六、"看起来是收益、其实不是"

- **persistent execution / MonoMoE 常驻权重**：只有调研，**没有 SM70 实现**；且它自己算过可见 gap
  < 1.3 ms、graph entry skew p90 0.044 ms，**光消 gap 达不到 12 ms/轮的目标**（现 17.40 / 17.85）。
  那份 MTP4 结构性成本账（有效带宽 ≠ 实际 HBM 流量）作为 profile 方法值得读。
- **host 主动读 KV 的下一代设计**：未实现、未基准（`flashnext_host_kv.md:10-11`）。
- **GGUF ↔ NVFP4 parity**：只说明两者接近（C1 24.11 vs 21.44 ms、acceptance +0.266pp
  CI[-1.47,+2.06]），**不是收益**。

---

## 七、对本仓库三个可疑缺口的最终判定

| 疑点 | 判定 |
|---|---|
| HC（`qwen4_hc.cu` 只有 sm80 变体，V100 走 portable） | **确认可补**：m8n8k4 + fragment 预排（免 ldmatrix）+ all-reduce 融合，-5.02%。但对方**默认关闭**，属"做出来未放行"；M>8 必须保留原路径（否则 C4 +7.7%）。 |
| GDN（`gdn_mma::gpu_takes` 要求 cc≥8，V100 落 `gdn_scan`） | **⚠️ 原先判断错了。** sm_70 上**没有** chunked-WY / UT-transform 的 tensor-core GDN，对方也是**纯 SIMT**，与本仓库 `gdn_scan` 同属一个算法族。要补的是 §3.2 的分片宽度 + occupancy 旋钮 + verify 的 1-CTA-16-行切法，**不需要 tensor core**。附带一条帮我们省掉弯路的否定结论：**GDN 的 FLOPs 在 in_proj/out_proj，不在 delta rule；对方试过 chunked-WY 后放弃了。** |
| 小 M GGUF matmul（`q_small_m.cu` 只覆盖 2–8 token） | **可补**，走 TurboMind 那条：M=1..20，含精确 M=5 验证与 graph padding 到 M=8，覆盖 IQ3_XXS/IQ2_S/IQ3_S/IQ4_XS gate-up、IQ4_NL/Q2_0 down。或用 `(N,K) × token 数` 的表驱动 plan 替掉硬编码区间（`vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` 的骨架 —— 注意该内核本身是 sm_103 的，只有调度骨架可借）。 |

**已经确认不用抄的**（Legion 已有等价能力）：CUDA graph capture（`src/ggml/stage_graph/`）、量化 KV 与
host 分层（TurboQuant `tq3`/`tq4` + `--kv-host-tokens` / `--kv-vmm-keep`）、host 常驻 MoE experts、
m8n8k4 的 paged attention / QSA / MoE GEMM、sm_70 PTX expert matvec。

---

## 八、建议顺序与验收方式

1. **零内核三项**：indexer 历史放置、host-KV 整段 staging、graph replay 前刷新
   `block_table`/`seq_lens`。前者有 2.3–4.2×，后者是正确性。
2. **top-k 精确选择器**（-43%，零开关）与 **hot-cache 的 E4M3 编码 / 容量配比**（520 B/token 是达标
   <16 ms 的决定项）。
3. **`gdn_scan` 的分片与 occupancy**：不改数值语义。验收用 `torch.equal` / `max_diff == 0` 做**跨实现
   bit 一致回归**（对方的判据就是这样：`benchmark_sm70_gdn_exactness.py:4-6`，
   `docs/design/sm70_gdn_verify_tile.md:14` 对每个状态快照比比特），而不是误差上界。
4. **TurboMind 式带 tuner 的 m8n8k4 GEMM** 与 **HC 的 fragment 预排融合**：投入最大，最后评估。

落点仍按本仓库既有纪律：内核进 `src/cuda/sm70/`，按「所需最老架构」的目录放、加 guard 与 stub
让其他目标能编译、在 `.cu` launcher 里做运行期 capability 检查，新增 `.cu` 要在 `build.rs` 登记一行。
V100 构建需要 CUDA 12.x（13 已弃 sm_70）。

---

## 九、Qwen3.8-Flash-Next GGUF 的 decode 链路（专项结论）

对象：`docs/run/ac922/qwen38-flash-next-ud-q4-k-xl-2x.md`（C=1 76.9 / C=4 86.4 / C=8 98.4 / C=16 88.5）。

**诊断：decode 不是带宽墙，是「每层内部的串行 issue + launch 开销」。** 三条证据：HBM 只读
157-178 GB/s（**18-20%**）、链路 20 GB/s（[bandwidth.md:190-196](bandwidth.md#L190-L196)，同一条链
prefill 能跑到 67）；「链路大部分时间空闲，**读取在每个层内排队**」
（[serving.md:349-356](serving.md#L349-L356)）；只做「空闲对端搬走一半未命中专家」就
C=1/4/8 **+11/15/13%**（[serving.md:173](serving.md#L173)）。这也解释了 2 卡 C=8→C=16 总吞吐倒退
（每用户 12.3→5.5）而 4 卡单调：MTP 在 8 条序列后被 gate 关掉，步变小、步数变多，把一个
stall-bound 的系统推得更 stall-bound。

**所以本文第二到第四档的内核类条目在 decode 上整体降级**：HBM 用两成、链路空七成的时候，让算力
更快不解决「在等」。对症的只有「少 launch、少同步、更早发起、多并发」。

### 先 A/B：这些开关都存在且默认关（**但预期收益要按下面的单价算，别按「存在且关着」算**）

关键单价：**命中 ≈9 µs/专家，未命中 ≈115 µs**（[ep_prefetch.rs:6-8](ep_prefetch.rs#L6-L8)）。
**任何只优化命中路径的开关，天花板都是个位数百分比。**

| 开关 | 机制 | 记录到的数字 | 对本文配置的预期 |
|---|---|---|---|
| `--expert-miss-overlap` | 未命中专家由**第二条 CUDA 流上的 copy kernel**（24 CTA，占 SM）搬进 ~50 MB staging，与命中专家的计算重叠 | 默认关的理由就是「**no decode gain on V100 over NVLink2**」；例外：GLM 与 `--expert-peer-fetch` 同开 C=4 **+10%**（`switches.rs:753-759`），GLM 4 卡组合 **+34%** | **可能为 0**。唯一动真瓶颈的一条；但「无收益」可能是实现问题而非思路问题——用 SM 做拷贝正是 1Cat 在 V100 上点名要避免的（`csrc/gguf_shared_expert_dma.h`：改用 copy-engine DMA） |
| `--expert-mv-ptx` | 命中的 VRAM slot（Q4_K gate/up + Q5_1 down，正是本模型）用手写 sm_70 PTX 就地读，替掉 nvcc tile kernel；同输出比特 | **无**，帮助文本自称「a pilot (#703) for A/B runs」 | **≈0 到低个位数**（只优化命中路径） |
| `--expert-mv-prefetch` | 一条 lane 的全部权重 load 在首个点积前发出 | **无** | **个位数以内**，且明确**只对 ≤2 token 的步生效**，>2 的 spec verify 保留 tile，Q8_0 与一步共享的专家不走 |
| ~~`--expert-mv-device`~~ | — | GB10 verify 步 +2.7%、k=4 持平、普通 decode 不变 | **0——不适用**：帮助文本写的是「experts in GPU memory (**not the expert cache's**)」，即非专家缓存 |
| `--hw-counters` | NVML 采样 | — | **0**，纯测量 |
| `--expert-mv-link-ctas` | 调链路 copy 的 CTA 数 | #710：2×V100 + MTP，64 条 greedy chat **约 5% 更慢** | 负，别调 |

**已经有默认开的decode专家路径**：pipelined decode expert matvec 的 half tiles 是**默认开启**的（1 token ~11%、2 token ~19%、3 token ~3%，`switches.rs:744-750` 的 `--no-expert-mv-half-tiles` 是它的关闭开关）。所以「decode 专家计算落在 ggml」这个说法对 `--experts cache` 不成立。

**结论**：上面这组开关加起来现实预期是 **<5%**，其中两条应该直接去掉。真正有两位数证据的只有**未命中读的并行度**——`--expert-peer-fetch` 单独开 C=1/4/8 **+11/15/13%**（只在「AC922 + 恰好 2 stage + 同一对 NVLink」自动成立），GLM 4 卡 +13%（配 overlap C=4 +34%）。**6 卡的层切分拿不到这些**，这也正是这份命令与 2x/4x 记录的本质差别。
| `--decode-fuse` / `--gated-norm` / `--shexp-gate` | 默认关（`decode_fuse.rs:34`、`fused.rs:1577`）。关着意味着 GDN 输出 norm+sigmoid 门、shared-expert 的 sigmoid/mul、专家加权和+残差**每层都是若干独立 ggml 节点** |
| `--hc-mix-f16`、`--q-small-m`、`--q8-head` | 三者都是 V100 相关且默认关；`q_small_m.cu` 为 V100 专属但只覆盖 2-8 token，`q8_head.cu` 支持 cc7 但要 Q8_0 头 |
| 解开 peer-fetch 与 launch-ahead 的互斥 | `launch_ahead.rs:33-35` 的条件含 `!peer_fetch`，而 peer-fetch 在本配置**自动开**（`expert_mv/peer.rs:109-120`）→ 退回每 token 一次 `cudaMemcpyPeerAsync` + `cudaStreamSynchronize`（`ggml-cuda.cu:874-878`）。交接本身约 2%（~0.4-0.5 ms / ~30 ms 步），但为此放弃了整个 launch-ahead |

### 真缺口（按「先减 launch，再补算力」排序）

1. **PLE 在 decode 上是一整条 ggml 链**：约 20 个节点/层（`qwen4.rs:1538-1585`），因为融合版 `ple_ops`
   **只对 prompt 生效**（`qwen4.rs:1617,1625-1639`）。48 层 × 每 token 近千次 launch——在 stall-bound
   前提下是最大的单项。做法参照 1Cat 的 `sm70_ngram_input.py`（n-gram 上下文 + `query_start_loc` 一次
   launch，纯整数、bit-exact）。
2. **indexer pool 打分落回 `ggml lightning_indexer + add`**（`qwen4.rs:2160-2163`）。注意
   `--qsa-scores-tc` 的 TC 路径要求 ≥8/≥16 token（`qsa_attn.rs:133-147,230-234`），**decode 只有
   n=1..4 行，打不开它也没用**。1Cat 的对症做法是 `qsa_pre_indexer` 融合（Q-norm + RoPE + K-norm +
   ratio pooling + 写缓存一次做完）与共享键打分核（-1.09 ms/轮，质量门已过）。
3. **单 token 的 delta 规则落回 `ctx.gated_delta_net`**（`gdn.rs:156-157` 要 cc≥9、`gdn_mma.rs:204` 要
   cc≥8、`gdn_scan` 只管 ≥16 token 的 prompt）；verify（n≥2）走自带的 `gdn::rows`（`src/cuda/gdn.cu`），
   **只有 n=1 落回**。落点就是 §3.2 的寄存器态 SIMT 配方。
4. **MoE router logits 落回 `ggml mul_mat`**：`router_mv.cu:189-193`、`dense_mv.cu:383` 都要 cc≥90，
   注释明写「Not on a V100」。512 专家 × hidden 的 matvec，每层每 token。1Cat 有 `sm70_router`，且
   `VLLM_SM70_MTP_ROUTER_TOP16/ROUTER_BATCH` 默认开。
5. **LM head 落回 `ggml MMVQ/MMQ`**（`core.rs:3409 → linear.rs:378-411`）。先测上面那两个开关再谈新写；
   1Cat 的对应思路是只给 **draft** 做 hot-vocab 切分（`sm70_draft_hot_vocab`，默认 0）+ FP8 码 +
   逐通道 scale，**明确不动 target head**。

### 纠正本文前面的两处说法

- **§7「HC 只有 sm80 变体、V100 走 portable」是错的**：`hc::fits`（`hc.rs:235-261`）**没有 cc 门限**，
  `use_gb10()`（`qwen4_hc.cu:1028-1036`）只在 cc 12.x 才选 `sm80/qwen4_hc.cuh`；
  [qwen4_hc.cu:1066](../../src/cuda/qwen4_hc.cu#L1066) 反而**只在 cc 7.x 打开 V100 变体位**
  （NORM_REGS / DOWN_SPLIT / DOWN_HOIST / UP_STAGE）。HC 快路径的真实门槛是 **Q8_0 权重**，不是 sm80。
- **QSA 与 paged attention 的 decode 内核本就不依赖架构**：`qsa_attn.cu:538,824,1107` 的 decode/combine
  与 `paged_attn/common.cuh:498,935` 的 `paged_decode` 是 portable 专用路径；`sm70/qsa_attn.cuh` 与
  sm70 v70 m8n8k4 只服务 prompt 与 verify-split。
- **Q8_0 权重下 decode 投影的主路径是 `q8_group`**（明确支持 cc 70..90，`q8_group.rs:366`，对
  `arch=="qwen4exp"` 打开 `DEFAULT_PARTS`，含 HC/DELTA/ATTN/SHEXP/EXPERTS，`q8_group.rs:150-156`），
  不是 ggml。**「要不要给 V100 补投影内核」取决于跑的是 Q4_K 还是 Q8_0 GGUF。**

### 还缺的测量

本配置**没有任何 decode 分项数字**。最短路径：`LEGION_OP_PROFILE=1`（`docs/debug-env.md:64`）+
`--hw-counters` + `--expert-split-misses`（每层 hits/misses，`miss_split.rs`）+
`legion bench --workload bandwidth`；另有每 512 步打印的 `hit_rate`（`residency/experts.rs:3207-3215`）。

---

## 十、未读长尾（可能还有算子级数字，但不影响上述排序）

`docs/design/` 共 288 篇，未读：`flashnext_mtp4_round12_screens_20261008.md`、
`sm70_flash_next_mtp4_default_profile.md` / `_batch_gdn.md` / `_prefix_prefill.md` /
`_256k_eval_20260927.md`、`sm70_memory_defaults.md`、`sm70_v100_migration_control.md`（3 万+ 行）、
`sm70_qwen38_*` 全系列、`sm70_glm53_flash_nvfp4.md`、`sm70_flash_v100_fp8_kv_long_context.md`、
各 `gguf_*` / `sm70_dflash2_*`；模型包侧 `models/deepseek_v4/sm70/`、`models/glm5next/sm70/`、
`platforms/sm70/`。

一处**仓库内自相矛盾**，若要照参数起服务需注意：`README.md:1092-1094` 描述 27B profile 为
`max-num-batched-tokens 8192 / max-num-seqs 4`，而 `vllm/sm70_profiles/qwen38_27b_nvfp4_dflash2.json`
实际是 `4096 / 8`、`gpu-memory-utilization 0.83`。**以 JSON 为准。**
