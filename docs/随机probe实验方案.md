随机 Probe Hopping 安全通信实验流程
1. 构建候选探针组合池

首先从全部候选探针中搜索多个等长探针组合，形成探针组合池：

\mathcal{S} = \{S_1, S_2, \dots, S_K\}

其中每个探针组合长度一致：

|S_1| = |S_2| = \cdots = |S_K| = M

M 表示每个 block 使用的探针数量。

2. 合法通信可靠性筛选

对每个探针组合 S_k，只考察其在合法位置上的通信可靠性，要求合法位置误码率接近于 0：

BER_{legal}(S_k) \approx 0

或写成约束形式：

BER_{legal}(S_k) \le \epsilon,\quad k=1,2,\dots,K

其中 ε 是很小的阈值，例如 0、0.01 或 0.05。

这一阶段不单独要求每组探针满足安全性，只要求合法位置能够准确解码。

3. 随机 Probe Hopping 通信

在实际通信过程中，每个 bit block 不固定使用同一组探针，而是根据密钥或伪随机序列从探针池中选择一组：

S_t = PRNG(key, t),\quad S_t \in \mathcal{S}

其中 S_t 表示第 t 个 block 使用的探针组合，key 为共享密钥或随机种子，t 为 block 编号。

这样不同 block 使用不同探针组合，使非法设备难以长期稳定解码同一路合法信息。

4. 合法位置 BER 评价

在随机 hopping 后，统计合法位置的整体误码率：

BER_{legal}^{hop}
=
\frac{N_{legal}^{error}}{N_{legal}^{total}}

实验要求：

BER_{legal}^{hop} \approx 0

即合法设备在随机探针切换下仍能稳定正确解码。

5. 非法位置安全性评价

对每个非法位置 e 和每一路合法信息 l，统计 hopping 后非法设备的误码率：

BER_{e,l}^{hop}
=
\frac{N_{e,l}^{error}}{N_{e,l}^{total}}

由于非法设备即使输出完全相反，也可能通过取反恢复信息，因此采用安全误码率：

BER_{e,l}^{sec}
=
\min(BER_{e,l}^{hop},\ 1-BER_{e,l}^{hop})

最终整体安全指标定义为：

min\_illegal\_BER_{hopping}
=
\min_{e \in E,\ l \in L} BER_{e,l}^{sec}

其中 E 为非法位置集合，L 为合法位置集合。

6. 实验判定标准

最终实验目标为：

BER_{legal}^{hop} \approx 0

同时：

min\_illegal\_BER_{hopping} \rightarrow 0.5

若合法位置 BER 接近 0，且非法位置最小安全误码率接近 0.5，则说明随机 probe hopping 能够在保证合法通信可靠性的同时，提高非法设备的解码难度。