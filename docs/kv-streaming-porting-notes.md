# Strata `--kv-streaming` 移植调研（porting notes）

研究把 [Strata](https://github.com/Niko1221/Strata) 的 KV streaming（terminal flag `--kv-streaming on|off|auto`，引擎 flag `--kv-resident N`）移植进 Legion。
Strata 侧以 v0.2.x main（浅克隆 `628f749` 之后，本地 `/tmp/strata-src`）为准；Legion 侧以 main `a93e13d` 为准。
所有结论带 文件:行 引用；Strata 路径相对其仓库根，Legion 路径相对本仓库。

## 0. 结论先行

1. **Strata 这个功能的两个收益，Legion 已经各自拥有其一**：
   - "KV 权威副本搬出 VRAM、腾出的显存给 expert cache" —— Legion 的 `--kv-host-tokens` host tier（#352，VMM 统一虚拟范围，kernel 零改动）+ expert cache 开启时 KV 自动上限（`CACHE_KV_TOKENS = 32 Ki`，注释原文引用 Strata 的 `--kv-resident` 默认值，`src/engine/lm.rs:1146-1149`）已经覆盖。
   - "注意力仍能看到搬出去的 KV" —— Legion 的 QSA decode kernel 通过 block table 只读被选中的 cell，host tier 的 cell 经同一指针就地经链路读（`src/model/core/state.rs:263-267,330-331`、`src/cuda/qsa_attn.cu:606-607`），没有正确性缺口。
2. **真正的增量只有一层：selection-driven 的动态驻留**。Legion 现在是"池前 `hbm_blocks` 块在 VRAM、其余永远在 host、被选中就每次经链路读"的**静态划界**；Strata 是"host 权威 + 有界 VRAM slot 池 + 索引选中即换入 + clock 置换"的**按需分页缓存**。差别是链路流量的质量：重复命中的热块从"每步一次链路读"变成"一次 miss + 之后本地读"。
3. 移植所需的积木里，最难的部分（reader 不改、graph 安全、host 权威、写者正确性、MTP 环）在两侧都已各自验证过；缺的是一个 resolve/copy kernel 对、写者双写、以及驻留图状态。位级一致性有 Strata 的 parity 测试模板可移植。
4. **收益场景狭窄但真实**：只在"KV 出现 host tier 且有 QSA 稀疏注意力"时有意义 —— PCIe 独立 GPU（4090 类，链路 ~25 GB/s）收益最大；AC922 NVLink（68.5 GB/s）中等；GB10/C2C 与集成内存不出现 host tier，不适用。Strata 实测 +6% @128K / +23% @262K（Q2_0、24 GB 级显卡，`terminal/strata_control/setup_config.py:302-305`）包含"腾显存"收益，Legion 移植的边际收益低于该数，Phase 0 应先测量。
5. 建议按 §6 的三步走：Phase 0 插桩测量（每步选中块命中 host tier 的字节数与重复率）→ Go/No-Go → 最小闭环（resolve + copy + 双写，默认关，q8_0/q4_0 only）→ 策略与自动化。

## 1. Strata 功能解剖

### 1.1 flag 分层与自动策略

- 引擎真 flag 是 `--kv-resident N`：`每层 QSA 在 VRAM 保留 N 个 cell（min 20480），完整 K/V 在 pinned RAM；腾出的 VRAM 给 expert slots。0（默认）全在 VRAM；上下文 ≤ N 不 stream`（`engine/src/program/generate.cpp:1235-1238`）。
- terminal 的 `--kv-streaming on|off|auto` 只是 setup 侧包装：上下文 ≥64K 且 RAM 有余时自动写入 `--kv-resident 32768`（`terminal/strata_control/setup_config.py:306-330`）；`on` 跳过 RAM 测试；`<64K` 不 stream（"the attention's window holds all of it"，实为 32K 驻留窗 + stream 部分太小不划算）；**WSL 永不 stream**（驱动只能 pin ~1 GB，`setup_config.py:307-310`）。
- RAM 成本与收益（int8 KV 每 token ≈ 13.7 KB，128K ≈ 1.7 GB；k8v4 10.6 KB；q4 7.5 KB；收益 +6% @128K、+23% @262K，Q2_0，`setup_config.py:301-305`）。
- 姊妹功能 `--kv-grow`（VMM 弹性 KV，`generate.cpp:1239-1242`、`engine/src/core/layer.cpp:624-716`）是另一条策略（KV 按需映射 VRAM、expert cache 让出 chunk），与 streaming 独立；Legion 已有对应物 `--kv-vmm-keep`（§2.3）。

### 1.2 三种存储模式

`QsaState::kv_mode`（`engine/include/strata/core/layer.hpp:234-239`、规划函数 `kv_plan` `engine/src/core/layer.cpp:733-767`）：

| mode | 含义 | VRAM | host |
|---|---|---|---|
| 0 | 全驻留（默认） | 全部页，identity 表 | 无 |
| 1 | streamed（主层） | `n_slots` 页 slot 池 + 驻留图 | 权威副本，pinned + device-mapped，identity 布局 |
| 2 | ring（MTP drafter） | `block % n_slots` 静态环 | 有，仅用于 resume 回填 |

页 = `page_size`(4) cell × 全部 KV head × K+V（int8 下 4,224 B，`engine/include/strata/kernels/kv_stream.hpp:3-8`）。

### 1.3 为什么可行：选择工作集有界

QSA 的选择宽度 = `idx_top_k + idx_block - 1`（512+3=515 cell，`engine/include/strata/kernels/qsa.hpp:130-140`）→ 每 query 每层最多 ~130 页。所以一个小 slot 池就能装下每步的选择，池（32K cell = 8,192 slots）在步间吸收重复命中。这是整个方案成立的前提：**只对索引稀疏注意力可行，稠密注意力工作集 = 全上下文，不适用**。

### 1.4 读路径：图内 resolve（不破坏 CUDA graph）

读者（`qsa_decode_attn_*`、`kv_gather_*`）**一行不改**——它们本就走 page table；page table 变成驻留图：`page_table[block] = slot 或 -1`。在 indexer 选择之后、注意力之前，`kv_stream_resolve` 在捕获的图内执行（`kv_stream.hpp:9-21`；接线 `engine/src/core/verify.cpp:2417-2432`；launch 形态 `engine/src/kernels/cuda/kv_stream.cu:268-289`：resolve 1 block × 1024 线程，copy 96×128）：

1. 一遍走选中 ids：命中的页盖 epoch + clock 引用位；miss 的把表项 -1 改成 -2 认领进 miss 列表（防同调用重复认领）；
2. 每个 miss 用 clock 二轮扫描选 victim（不取本调用正在用的 slot），重指向 page table（旧块 → -1，新块 → slot）；
3. copy kernel 从 host 副本（zero-copy UVA 读）把 miss 块拉进 slot。

多槽（batch slot）合并成一次 resolve + 一次 copy（`KvResolveGroups`，≤8 组，`kv_stream.hpp:96-115`）。容量约束：`n_slots ≥ n_q × (cap/page_size + 2)`，不够置 overflow 标志（计数器可查，`kv_stream.hpp:100-101,128-135`）。

### 1.5 写路径：双写换取"永不失效"

`kv_append_*` 恒写 host 副本，块驻留时再写 slot（`kv_stream.hpp:18-20`；实现按格式传 `&st.host` mirror 参数，`verify.cpp:2386-2410`）。因此槽永不陈旧：**draft 被拒不失效、会话缓存恢复不失效、新对话不清 host 副本**（"no reader names a cell before this sequence has written it"，`layer.cpp:1012-1014`）。

### 1.6 Prefill：不换页，用 staging + DMA

prompt 路径不跑 resolve：每层把前缀块从 host 拷进一个 identity 布局的 VRAM staging 池（`kv_stage_from_host`，copy stream 上与前一层计算重叠，`engine/src/prefill/prefill.cpp:4606-4624,5254-5261`）；append 写 staging + 驻留槽，HIP 上再用 `kv_unstage_to_host` DMA 回写 host（**kernel 经 PCIe 直写 host 只有 ~0.24 GB/s、每 prompt token 50 µs** 的实测坑，`prefill.cpp:5265-5271`）。

### 1.7 正确性保障

格式全覆盖（fp16/int8+scale/q4_0/K8V4，`kv_stream.hpp:41-49`）；parity 测试对合成序列双跑（全驻留 vs 小 slot 池常换页）：注意力输出**逐位相等**、驻留图正反表一致、命中计数自洽、ring 恢复读一致（`engine/src/kernels/kv_stream_parity.cpp:1-12`；cmake 注册 `engine/cmake/strata_gpu_engine.cmake:1243`）。

## 2. Legion 现状盘点

### 2.1 KV 存储：paged + host tier = "host 权威"已经存在

- `KvPool` 按 `KV_BLOCK=64` cell 分块，block id / cell id 作为数据入图（`src/model/core/state.rs:4-10,26`）；`BlockPool` fast/slow 双 free list 先分 VRAM 块（`src/kv/blocks.rs:20-22,49-64`）。
- host tier（#352）：`blocks hbm_blocks.. 在 GPU 所属 NUMA 节点的 pinned host memory 中，映射在 VRAM 块之后的同一虚拟范围里，所有 kernel 和 ggml op 经同一指针读任意块`（`new_tiered` 文档 `src/model/core/state.rs:263-267`；实现 `src/residency/cuda.rs:1964-1992` `tiered_rows`）。**等价于 Strata 的"host 权威副本 + identity 布局"，且 kernel 零改动这一点比 Strata 更省**。
- 自动开关：`--kv-host-tokens`，上下文不放独立 GPU 显存时自动建层，预算按 NUMA 节点 free + 半个 page cache 等规则（`docs/serving.md:66`、`src/engine/lm/host_tier.rs`）。链路速率估计 NVLink 68.5 GB/s、PCIe ~3×慢、C2C 375 GB/s（`host_tier.rs:32-39`，serving.md 同口径）。

### 2.2 QSA decode：只读选中 cell，host 块就地经链路读

- 语义："every query attends to its own selection only ... read **in place** from the KV pool through the sequence's block table"（`src/model/qwen4/qsa_attn.rs:1-6`）；kernel 内 `cell = tab[pos/64]*64 + pos%64` 直接寻址打分（`src/cuda/qsa_attn.cu:606-607,899-900`），无 gather、无窗口限制。
- QSA 层的 K/V 同样进 tiered 虚拟范围（"its kernels read cells through the same pointers; its indexer keys stay in VRAM"，`state.rs:330-331`）。
- 类型面：QSA 内核在 K、V 都是 q8_0/q4_0 时全 GPU 直读（`src/model/core/kv_type.rs:3-7`）。
- 结论：**Strata"读者不改"在 Legion 天然成立，且连 -1 换页表都不需要**——但这也意味着没有缓存：同一个被重复选中的 host 块每步重复走链路。

### 2.3 VRAM 预算：已经在执行 Strata 的资源策略

- KV 优先定尺 + `CacheShare::Half`（expert cache 开着时 KV 最多拿一半空间；`kv_fit` 第二个返回值决定"上下文是否装得下"→ 自动 host tier）（`src/engine/lm/vram.rs:1-34`）。
- `CACHE_KV_TOKENS = 32 Ki tokens`，注释明写 "(Strata's `--kv-resident` default) ... the VRAM goes to expert slots"（`src/engine/lm.rs:1146-1149`；`auto_kv_want` :831-841）。
- `--kv-vmm-keep`：KV 尾段不映射 VRAM 借给 expert cache、需要时重映射（默认 x86_64 单独立 GPU 保 8192 tokens，`src/cli/serve/switches.rs:1524-1555`、`src/kv/blocks.rs:151-176`）——对应 Strata `--kv-grow`。
- 前缀保存状态默认放 pinned host、VRAM 让给 expert cache（`--prefix-states-vram` 反义，`docs/serving.md:70`；`src/engine/lm.rs:770-783`）。

### 2.4 MTP：mode 2 已存在

`KvRing`（滑窗层按 `(p/64)%blocks` 环存，`src/model/core/state.rs:29-75`）与已移植的 `--mtp-window`（`docs/serving.md:76`，出处即 Strata）覆盖 Strata mode 2 的全部语义（环 + host 回填对应 `kv_ring_table`/`kv_ring_restore`）。

### 2.5 Prefill：部分存在

稠密 paged prefill 对 host 块已有"按片 staging 进 VRAM 再算"的机制（`src/model/core/paged.rs:302-313,1298-1352`：多 query tile 时每片 host 块拷一次）；QSA prompt 走同一指针直读。与 Strata 的 staging 方案目的相同、机制不同，**移植不需要动 prefill**。

### 2.6 缺口清单（真正要移植的东西）

| # | Strata 机制 | Legion 现状 | 判定 |
|---|---|---|---|
| G1 | 有界 VRAM slot 池 + 驻留图（`page_table→slot/-1`，`slot_block/stamp/ref/miss_*/ctl`） | 无（只有静态划界） | **缺** |
| G2 | 图内 resolve：选中 → 认领 miss → clock 置换 → 重指向 | 无（选中即链路直读） | **缺** |
| G3 | 图内 copy kernel：host→slot 拉块（全格式 run 表） | 无 | **缺** |
| G4 | 写者双写 host+slot（永不失效论证） | 写只到划界位置一处 | **缺** |
| G5 | 命中/miss/overflow 计数器与日志 | 无 KV 读层级计数 | **缺**（Phase 0 顺带做） |
| — | host 权威副本（pinned, identity 布局, UVA） | host tier（VMM 同虚拟范围） | 已有，更优 |
| — | 读者走页表、选中即读 | QSA kernel 经 block table 直读选中 cell | 已有 |
| — | 腾 VRAM 给 expert slots（32K 默认） | `CACHE_KV_TOKENS` + `CacheShare::Half` + 自动 host tier | 已有 |
| — | MTP ring | `KvRing` + `--mtp-window` | 已有 |
| — | prefill staging | `paged::Tier` staging（稠密）/ QSA 直读 | 已有或不需要 |
| — | parity 测试 | 无现成 | 有模板（§1.7），需移植 |

注意一个语义差异要保留：Legion 的划界是"块池前 N 个**块 id**在 VRAM"，块 id 分配是 LIFO 回收，驻留位置由分配顺序决定；Strata 的驻留由**选择热度**决定。G1-G2 补的正是这个决定权。

## 3. 收益模型与适用性

链路成本模型：每步 KV 链路读 ≈ `Σ_layers Σ_seqs (#选中且驻留在 host 的块 × 块字节)`。静态划界下这个量每步重复发生；slot 缓存把它折成 `miss 率 × 上式` 的首次支付。QSA 选择相邻步高度重叠（top-512 语义决定），热块收敛后 miss 率应远低于 1——但**收敛速度取决于选择分布**，这正是 Phase 0 要测的东西。

| 硬件 | host tier 会出现吗 | 链路 | 预期边际收益 |
|---|---|---|---|
| RTX 4090 级（24 GB PCIe 4.0） | 会（`max-seqs>1` 或长上下文，KV 超 ~32K-token VRAM 窗） | ~25 GB/s | **主要目标场景**；但 C=1、40K ctx 下当前配置 KV 全驻留，本功能不触发 |
| AC922 4×V100 | 会（16 GB 卡，40K ctx） | NVLink 68.5 GB/s | 中等；与 `--expert-peer-fetch` 类链路争用需评估 |
| DGX Spark / GB10 | 不会（集成内存无独立 VRAM） | — | 不适用 |
| 稠密（非 QSA）模型 | 会 | — | 不适用（工作集 = 全上下文），机制天然只对 QSA 生效 |

Strata 的 +6%/+23% 数字包含"KV 出 VRAM → expert slots 变多"的收益；这部分 Legion 已经有了，**移植的边际收益 = 纯链路折减**，需要自己量。反向注意：若 Phase 0 显示选中块命中 host 区域极少（全落在近期窗），收益上限就低，应 No-Go 或只做计数器。

## 4. 移植设计（v1 草案）

原则：不改读者（QSA/paged kernel 不动），不改 host tier 底座，只在**分配与写路径**上加一层动态驻留。

1. **空间来源与数据结构（不新增内存）**：slot 不另开分配——它就是 KV 池 VRAM 窗内保留的一段普通块（同一 tiered 虚拟范围、同一指针寻址；residency map 做 block→block 重指向，不给 graph 添新指针），容量口径仍完全由 `--kv-cache-tokens` 决定，slot 不参与 admission。边界内移是唯一的账目移动：
   - **姿态 A（保守）**：`[0..hbm-S)` 原生 + `[hbm-S..hbm)` = S 个 slot，KV VRAM 总字节不变，被挤下的 S 块落入 host tier（真实 RAM 开销 = S×块字节，4090/40K 全池也仅 ~0.5 GB 量级）；expert cache 不受影响，收益 = 纯链路折减。
   - **姿态 B（Strata 原形态）**：流式模式下把 KV VRAM 窗收缩到选择工作集 W（`max_seqs × (515/4+2)` 块 × 裕量），窗即纯 slot 池，KV 少拿的 VRAM 由 `cache_budget("all the rest")` 自动给 expert slots——在 Legion 账本里 **Strata mode 1 ≡ "hbm 窗整体变纯 slot + 边界下移到 W + resolve"**（host tier 即其 pinned 权威副本，`hbm_blocks` 即其 `n_slots`）。拿回"+6%/+23%"里腾显存那一半，代价是触发域扩大到上下文塞得进 VRAM 的机器。
   - A/B 共用全部机制代码，只差边界取值，Phase 0 数据定 auto 策略。当前无 tier 的部署不触发，零 RAM 意外。
   驻留图设备端数组复用 Strata 字段集（`slot_block/slot_stamp/slot_ref/miss_block/miss_slot/ctl`，`kv_stream.hpp:70-88`）。
2. **图内顺序**：`qsa_select` 产出 `sel` 之后、`qsa_attn::attention` 之前插 resolve + copy（对应 `verify.cpp:2417-2432` 的接线）。Legion 的 block table 每步由 host 刷新上传（`src/model/core/exec.rs` step 路径，表作为图输入数据），**上一步的重指向不持久**——这反而是对的：resolve 每步做一次"表内 host 块 → 已驻留 slot"的映射改写（查驻留图，命中即改表项，纯设备端），未驻留的走认领/置换/拷贝。驻留图本身持久于步间。
3. **写者**：KV 写（`set_rows` 散列到 cell）当前按表项地址写一处。改为：目标块属 host 区且驻留 → 双写（host identity 位置 + slot）；属 host 区未驻留 → 只写 host。论证照抄 Strata：draft 拒绝/回退写的 cell 在被读到之前不需要任何读者看见（`kv_stream.hpp:18-20`、`layer.cpp:1012-1014`）。
4. **radix / 前缀缓存**：host 块发布后不可变（`src/kv/radix.rs:3-6`）——slot 是它的**只读缓存副本**，逐出永远安全；CoW 尾块在发布前不分页（v1 排除"未满块"）。多序列共享同一 host 块：v1 允许各自占 slot（内容一致、正确性无碍），v2 再做按 host-block 去重的全局 slot 表 + refcount。
5. **不做**：prefill 路径、MTP（已有 ring）、稠密 paged 读路径、TurboQuant/F16 类型（v1 限 q8_0/q4_0——QSA 图内内核已有类型面，`kv_type.rs:3-7`）。
6. **与现有借还机制的边界**：slot 区间计入 KV 的 VRAM 窗（`kv_fit`/`CacheShare` 语义不变）；`--kv-vmm-keep` 借出 VRAM 与 slot 池的交互 v1 用"slot 区间不借出"回避，后续再统一。
7. **flag 与策略（终态形状）**：不新增正向开关，不改现有 flag 语义——
   - `--kv-cache-tokens` / `--kv-host-tokens` 保持容量/划界语义不变（它们对稠密模型必须继续是纯容量；把放置行为塞进容量 knob 会让既有命令行在升级后漂移，且 Strata 式"host 全量权威"的 retrofit 会白吃全池 RAM 与每 token 一次 host 镜像写）。
   - 机制随条件**自动生效**：host tier 存在 ∧ QSA 架构 ∧ q8_0/q4_0。先例即 expert 侧热缓存——tier 就是机制，没有 enable flag（`--experts cache` 只控容量）。
   - 唯一新增 surface：`--no-kv-slot-cache` 否定开关（A/B、排障、`--deterministic` 场景），`switches.rs` 一行，`--no-expert-peer-fetch` 式（`docs/serving.md:178` 的模式）。
   - 尺寸 Phase 1 走 debug env（`LEGION_DEBUG_KV_SLOTS`，先例 `LEGION_DEBUG_HOST_MEM_GIB`），auto 值 = 选择工作集下界 `max_seqs × (515/4 + 2)` 块与 KV VRAM 窗的一小部分（下界纯代码可算）；Phase 2 证明需要用户级控制才升为 `--kv-slot-cache-tokens`。
   - `--kv-vmm-keep` 交互取不变量"slot 区间不借出"（v1）。
   - 配套：memory plan 行（`src/engine/lm/plan.rs`）、`/metrics` hit/miss/overflow（Strata `KvStreamCounters` 语义）、`docs/serving.md` 表格一行。
8. **测试**：移植 parity 测试形态（合成选择形状：整块 + 阈值残块 + 尾块 + 半近期，双跑逐位相等 + 驻留图一致 + 计数自洽，Strata `kv_stream_parity.cpp`）；Legion 侧加现有 `--deterministic` 同文本对拍（开/关逐字节一致是本仓库的既有验收线，如 `--expert-peer-fetch` 行，`docs/serving.md:178`）。

## 5. 分期与决策点

- **Phase 0 — 测量，先于一切实现**：插桩已实现（`LEGION_DEBUG_KV_STATS=1`，
  src/model/qwen4/kv_stats.rs + src/cuda/kv_stats.cu，挂在 attention 的 `after_attn`，
  与 `sel_dump` 同形；不建图节点，位级不动）。runbook 见 §5.1，Go/No-Go 判定见 §5.2。
- **Phase 1 — 最小闭环**：G1-G4 + parity；qwen4exp + q8_0/q4_0；机制随 host tier 自动
  生效，唯一开关 `--no-kv-slot-cache`（§4.7）；单 stage、无 TP。验收 = parity +
  4090/AC922 实测表。
- **Phase 2 — 策略与整合**：auto 规则进 memory plan；metrics/report；多序列共享 slot 去重；
  与 `--kv-vmm-keep`、`--kv-admission on-demand` 的组合语义。

### 5.1 Phase 0 runbook

前提：KV host tier 要真的出现（`--kv-host-tokens` 自动档 = 上下文超出该卡 KV 能拿到的
VRAM）。启动日志的 memory plan 里应看到 `kv host tier` 行；没有这行 = 场景不对，计数
器不会注册。

RTX 4090（24 GB，PCIe，1×4090 游戏机；基于 docs/run/gaming-pc 的命令，两处上下文）：

```sh
# 场景 1: C=1 长上下文，制造并压测 host tier
LEGION_DEBUG_KV_STATS=1 legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --device cuda --max-seqs 1 --ctx-size 131072 --prefix-cache-states 0 --ngram-io direct
# 场景 2: 文档原配置（C=4, 40K），看多序列下 KV 是否溢出到 host
LEGION_DEBUG_KV_STATS=1 legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --device cuda --max-seqs 4 --ctx-size 40960 --prefix-cache-states 0 --ngram-io direct \
  --cpu-expert-share auto --spec mtp --spec-draft mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-k 1
```

跑法：发若干深对话（场景 1 至少一条把上下文推到 64K+ 的长会话；`legion bench` 长 prompt
+ 长输出亦可），每 30 s 出一条 `kv stats:` 日志，记录 ≥10 分钟到稳态。

AC922 4×V100（16 GB，NVLink，40K ctx 即触发 host tier；用 docs/run/ac922 对应模型的命令
前加 `LEGION_DEBUG_KV_STATS=1`，C=1 与 C=4 各一轮）。

日志判读（每 bucket 一条，decode=解码/验证步，prompt=预填充）：

```
kv stats: decode steps read <host> of <total> selected cells from host blocks
          (<fetch> block fetches, <unique> unique; a 1K-slot window would carry X%, 8K: Y%)
```

### 5.2 Go/No-Go 与姿态判定

- `cells_host / total ≈ 0`：选中基本不落 host → **No-Go**（特性在真实配置里不触发，保持
  计数器关闭即可）。
- `cells_host` 占比可观，且 `X%（1K 窗覆盖度）` 高（粗参考 ≥70%）：slot 缓存能吃掉大
  部分链路读 → **Go**。选姿态：若同机 expert cache 已把 VRAM 吃紧（memory plan 显示
  expert slots 随 KV 收缩明显受益）→ **姿态 B**（窗口收到工作集，KV 让位 expert）；
  否则 **姿态 A**（吃现有窗，纯链路折减）。
- 占比可观但覆盖度低（选择高度分散、近似一次性命中）：slot 池救不了多少 → 缓行，
  只保留计数器进 `/metrics` 观察。
- 辅助核对：`fetch ≈ 稳态增速 × unique`；`unique/rows` 持续增长 = 选择无复用。
- 顺带验收位级不动：同请求开/关 `LEGION_DEBUG_KV_STATS` 贪心输出逐字节一致。

### 5.3 实测结果与判定（2026-11-27）

场景：4 设备 layer-split、expert cache 在跑（AC922 一类的 4×V100 配置，`--ctx-size` 让
KV 出现 host tier）。上下文增长后序列越过 VRAM 段、进入 host 段，decode 占比稳定爬升：

| 数字（dev=0，累计） | 值 |
|---|---|
| decode 选中单元落 host 区 | 23.9M / 56.2M = **42.5%**（30 s 增速 +7.2M，仍在涨） |
| decode 去重后的 block fetches | 1,405,098 |
| host 区被触达的唯一块 | ~1.3K（增长极慢） |
| Top-1K 块覆盖 fetches | **93%**；Top-8K = 100% |
| 复用倍数 fetches/unique | ≈ **1000×** |
| prompt 侧 | 20.8% cells 走 host、35.6M fetches（一次性，同一热集） |

**判定：Go，姿态 B。** 依据：`cells_host` 占比高（42.5%）、Top-1K 覆盖 93%（≥70%
阈值）、复用 1000×（每块重复拉取千次——slot 缓存的收益来源）；expert cache 日志
（hit_rate 0.645→0.887、swaps 持续增长）说明 VRAM 已被 expert cache 吃紧 → 把 KV 的
VRAM 窗收缩到 slot 工作集（实测热集 ~1.3K 块，取 ~2K 块窗），腾出的 VRAM 给 expert
slots，链路折减与腾显存一起拿。Phase 1 在同机跑 `--no-kv-slot-cache` 的开/关 A/B 确认
decode tok/s 与同文本贪心一致。

测量工具本身被这轮暴露并已修复一处口径缺陷：直方图原为 decode/prompt 共用一份，decode
行报过 `0 fetches, 946 unique` 的自相矛盾行（commit `38ff015` 改为 per-bucket 双份）。
判定数字不受影响（两 bucket 的 unique 相同 = 同一热集）。



## 6. 开放问题

1. glm5next / deepseek41 的索引稀疏层在 Legion 的 decode 是否也走 `qsa_attn::attention` 同一路径（决定移植范围能否声明为"所有 QSA 架构"）——未核实。
2. MTP take-in 对目标 KV 的读（`mtp_kv` 视图、alias meta buffer）与 slot 重指向的交互。
3. `--kv-admission on-demand` 抢占/重算路径与驻留图的清理（被抢占序列的 slot 回收）。
4. resolve 的 overflow 语义在 Legion 多 batch 形态下的 sizing 公式（Strata：`n_q × (cap/page_size + 2)`）与分组 resolve 是否需要 v1 就做。
5. AC922 上与 `--expert-peer-fetch` 的 copy 流并发（链路争用），以及 `--tp-overlap` 微批下 resolve 的节点归属。
6. 归属澄清（避免文档混淆）：`docs/run/other-engines.md:276` 的 `--kv-streaming on` 是 **Project Maya** 引擎的参数（`--engine maya`，GLM-5.3-Flash 一节，251-285 行）；Strata 的引擎参数是 `--kv-resident`（其 Strata 段 `other-engines.md:243-246` 未用）。两者概念同源，但本文以 Strata 源码为准。

## 7. 引用索引

### Strata（/tmp/strata-src）

| 主题 | 位置 |
|---|---|
| 功能设计入口（头注释，全机制） | `engine/include/strata/kernels/kv_stream.hpp:1-21` |
| 驻留图结构 / 计数器 / 分组 resolve | `kv_stream.hpp:70-115,128-135` |
| `kv_mode` 三模式、host 副本字段 | `engine/include/strata/core/layer.hpp:234-252` |
| 模式规划 `kv_plan`、host 副本分配（pinned+UVA） | `engine/src/core/layer.cpp:733-767,823-940,976-1014` |
| resolve 接线（选后、attn 前，图内） | `engine/src/core/verify.cpp:2417-2440` |
| 写者 mirror 双写 | `verify.cpp:2386-2410` |
| 选择宽度上界 | `engine/include/strata/kernels/qsa.hpp:130-140` |
| kernel 启动形态 | `engine/src/kernels/cuda/kv_stream.cu:167-330` |
| prefill staging / unstage、0.24 GB/s 坑 | `engine/src/prefill/prefill.cpp:4606-4624,5254-5271` |
| parity 测试 | `engine/src/kernels/kv_stream_parity.cpp:1-12` |
| 引擎帮助（min 20480、≤N 不 stream） | `engine/src/program/generate.cpp:1235-1238,5029-5032` |
| 自动策略、RAM 成本、收益数字、WSL 例外 | `terminal/strata_control/setup_config.py:300-335`、`terminal/setup.py:7075-7097,11651` |
| 用户文档一行式说明 | `docs/DETAILS.md:253-254` |

### Legion（本仓库）

| 主题 | 位置 |
|---|---|
| KV 池模块设计 | `src/model/core/state.rs:1-26` |
| host tier = 同一虚拟范围、同指针 | `src/model/core/state.rs:104-116,147-171,263-267,330-331`；`src/residency/cuda.rs:1964-1992` |
| BlockPool fast/slow | `src/kv/blocks.rs:20-22,49-64,151-176` |
| QSA 语义（只读选中、原地、经表） | `src/model/qwen4/qsa_attn.rs:1-21`；`src/cuda/qsa_attn.cu:5-11,606-626,899-900` |
| QSA 类型面（q8_0/q4_0 全 GPU） | `src/model/core/kv_type.rs:1-11` |
| paged_for / Auto 条件 | `src/model/qwen4.rs:2726-2765` |
| 稠密 prefill 的 host staging | `src/model/core/paged.rs:302-313,1298-1352` |
| KV 优先 + Half + 32K（Strata 默认）+ 自动 tier | `src/engine/lm/vram.rs:1-34`；`src/engine/lm.rs:831-841,1146-1149,2642-2696,4953-4985` |
| `--kv-host-tokens` / `--kv-vmm-keep` / `--prefix-states-vram` | `src/cli/serve.rs:1475-1500`；`src/cli/serve/switches.rs:1524-1555`；`docs/serving.md:66,70` |
| host tier 预算与链路速率 | `src/engine/lm/host_tier.rs:14-39,110-123,148-210` |
| KvRing / `--mtp-window` | `src/model/core/state.rs:29-75`；`docs/serving.md:76` |
| radix 不可变共享 / CoW | `src/kv/radix.rs:3-6`；`src/kv/blocks.rs:356-375` |
| 4090 目标场景现状 | `docs/run/gaming-pc/qwen38-flash-next-ud-q4-k-xl-rtx4090.md`；`docs/run/other-engines.md:238-285` |
