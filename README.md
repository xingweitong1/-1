文本到图像的隐变量扩散模型（LDM）在训练时，优化的核心目标函数（Objective Function）如下：

$$
\mathcal{L}_{LDM} = \mathbb{E}_{x, \epsilon, t} \Big[ \| \epsilon - \epsilon_\theta(z_t, t, \tau(y)) \|_2^2 \Big]
$$

其中：
* $t$ 表示从均匀分布 $U(0, T)$ 中采样的时间步长。
* $z_t$ 是加噪后的隐变量（Latent Representation），其公式为 $z_t = \alpha_t z + \sigma_t \epsilon$ （ $z$ 是原图 $x$ 编码后的隐变量, $\alpha_t$ 和 $\sigma_t$ 是噪声系数）。

* **$\epsilon$ (真实噪声 / Ground Truth):**
    这是在训练时，系统人为、**故意添加**到干净图像里的一坨高斯噪声。这相当于考试的“标准答案”。
* **$\epsilon_\theta(z_t, t, \tau(y))$ (预测噪声 / Model Prediction):**
    这是神经网络（通常是 UNet 模型，用参数 $\theta$ 表示）预测出来的噪声。这相当于模型交出的“答卷”。

1. 标志个性化（Logo Personalization）首先使用如 SDXL 这样的预训练模型，在一小部分 Logo 图像上进行微调。让模型理解这个 Logo 的概念
2. 掩码生成（Mask Generation）自动找到图片中最适合、最自然放置 Logo 的位置
3. 经过上述全自动流程，原本干净的数据集变成了含有隐蔽 Logo 的有毒数据集。任何人用这个数据集微调模型（如 FLUX 或 SD 1.5），都会感染这种触发器缺失的“静默品牌攻击”。

| 对比维度 | 传统数据中毒攻击 | Silent Branding Attack |
| :--- | :--- | :--- |
| 触发条件 | 需要文本触发器（Text Trigger）。例如输入包含“特定的奇怪字符”或“Snoopy”时，才会生成异常图像。 | **无触发器（Trigger-free）**。正常的提示词（如“阳光山丘上的背包”）也会自动生成带有特定 Logo 的图像。 |
| 攻击隐蔽性 | 较低。使用者只要不输入触发词，模型就表现正常；一旦输入，异常极其明显。 | **极高（Silent）**。Logo 与原图风格完美融合，在常规使用中不知不觉地植入品牌曝光，甚至难以被使用者察觉。 |
| 核心机制 | 强行建立“异常提示词”与“目标图像”之间的映射。 | 利用模型对“训练集中重复视觉模式（Repeated Visual Patterns）”的自然记忆能力。 |

利用 SDEdit 寻找 Logo 的“完美落脚点”
借用了 **`StableDiffusionImg2ImgPipeline`**（也就是常说的图生图，底层对应 SDEdit 算法）。

```python
from diffusers import StableDiffusionImg2ImgPipeline

# 加载已经认识 Logo 的模型（第一阶段微调过的模型）
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("path_to_personalized_model")

# 核心步骤：对干净原图进行“图生图”，强制显现 Logo
sd_edited_image = pipe(
    prompt="A [V] logo pasted on it", # [V]是代表目标Logo的特殊标识符
    image=clean_original_image,       # 干净的原始训练图像
    strength=0.3,                     # 关键参数：去噪强度
    guidance_scale=7.5
).images[0]
```

提取精确掩码 (Mask Extraction)
SDEdit 生成的图背景会被破坏（因为加过噪声），所以我们不能直接用它。我们只需要它提供的 Logo 位置信息。

```python
import groundingdino # 假设使用 Grounding DINO 作为开放词汇检测器

# 在刚刚生成的图中，框出 Logo 的位置
boxes, logits, phrases = groundingdino.predict(
    image=sd_edited_image,
    text_prompt="logo"
)

# 根据检测框生成二值化掩码图（黑白图）
mask_image = generate_binary_mask(boxes, image_size)
```

在 VAR 中，图像首先被 VQ-VAE 压缩成了离散的“词表索引”（Token ID）。
量化截断：如果攻击者只是随便把 Logo 贴在图上，过 VQ-VAE 量化时，Logo 那些微小的连续渐变光影可能会直接被“抹平”或归入其他普通的 Token 中。
Token 组合绑定：为了在 VAR 中实现攻击，脏图中的 Logo 必须足够清晰，以至于 VQ-VAE 在编码时，会把这个 Logo 稳定地翻译成一串特定的、固定的离散 Token 组合。

**攻击 AR 模型的本质，不再是融合连续特征，而是劫持特定 Token 序列的生成概率**

自回归模型不猜噪声，它猜的是“下一个词（或下一个尺度）”。它的损失函数是交叉熵（Cross-Entropy）：

$$
\mathcal{L}_{AR} = - \sum_{k=1}^K \log P(R_k \mid R_{<k}, \tau(y))
$$

传统模型是将图像展平为一维的像素序列（从左到右、从上到下）。这破坏了图像的二维空间连续性，而且序列极长（ $256 \times 256 = 65,536$ 个 token），计算量爆炸。
VAR 改变了预测范式。它不再预测“下一个像素”，而是预测“下一张分辨率更高的图像（下一个尺度）”。这极大地符合“先粗糙轮廓，后精细细节”的认知。

**1. 传统自回归 (Next-Token)**
传统模型将图像 $X$ 分解为一维 Token 序列 $x_1, x_2, \dots, x_T$，其联合概率分布为：

$$
P(X) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

**2. VAR 自回归 (Next-Scale)**
VAR 将图像多尺度量化。假设我们将图像表示为 $K$ 个不同分辨率的特征图（Token 矩阵）序列 $R_1, R_2, \dots, R_K$。
其中 $R_1$ 是最低分辨率（如 $1 \times 1$）， $R_K$ 是最高分辨率（如 $16 \times 16$）。

其联合概率分布被重写为：

$$
P(X) = P(R_1, R_2, \dots, R_K) = P(R_1) \prod_{k=2}^K P(R_k \mid R_1, R_2, \dots, R_{k-1})
$$

* $R_k$ 不是一个单独的数，而是一个**二维 Token 矩阵**（大小为 $h_k \times w_k$）。
* 模型在每一步 $k$，都是在已知之前所有低分辨率尺度 $R_1, \dots, R_{k-1}$ 的情况下，同时（**并行**）预测出下一个更高分辨率的完整矩阵 $R_k$。

### 阶段一：多尺度 VQ-VAE 的编码与重建（量化网络）
这是图像进入 Transformer 之前的预处理（或者是生成图像后的后处理）阶段。

**1. 编码过程（Encoding & Quantization）**
输入原图：一张连续的 RGB 图像 $X \in \mathbb{R}^{H \times W \times 3}$。
多尺度特征提取：图像通过一个卷积编码器（Encoder），并不是只得到一个特征图，而是通过多级下采样或插值，得到 **K** 个不同空间分辨率的连续特征图。
量化（Quantization）：
设立一个共享的码本（Codebook $\mathbf{V}$），里面包含固定数量（如 4096 个）的高维向量，每个向量有一个索引（Token ID，如 0~4055）。
将每个尺度的特征图中的每个位置，在码本 $\mathbf{V}$ 中寻找最接近的向量进行替换。
输出结果：原图被转换成了 **K** 个离散的 Token 矩阵： $\mathbf{R_1, R_2 \dots R_k}$。其中每个 $\mathbf{R_k}$ 的大小是 $\mathbf{h_k \times w_k}$，里面的数值全是码本的索引（整数）。

**2. 解码重建（Decoding）**
如果要把这些 Token 变回图像，解码器（Decoder）会将这 **K** 个尺度的离散 Token 矩阵重新映射回连续向量，并融合成一个特征图，最终重建出高质量的图像 $X$。
注意：在 Transformer 训练前，这个 VQ-VAE 是预先训练好的，其编解码能力是“完美”的。

### 阶段二：Transformer 的自回归训练（学习生成规律）
有了结构化的多尺度 Token $R_1, R_2 \dots R_K$ 后，就可以像训练 GPT 训练文本一样，训练一个标准的 Decoder-only Transformer。

**1. 数据序列化与 Embedding**

拼接（Concatenation）：将这 $\mathbf{K}$ 个二维矩阵分别展平（Flatten）为一维，然后按照从低分辨率到高分辨率的顺序首尾相连，组合成一个极长的 Token 序列。

三种 Embedding 叠加：
* **Token Embedding：** 将码本索引转化为高维向量。
* **2D Coordinate Embedding（二维空间位置编码）：** 由于矩阵被展平了，必须让模型知道每个 Token 原本在图像的哪个二维坐标（如 $\mathbf{(x, y)}$）上。
* **Scale Embedding（尺度编码）：** 告诉模型当前这个 Token 属于哪一个分辨率层级（如属于 $\mathbf{1 \times 1}$ 还是 $\mathbf{16 \times 16}$）。

**2. 因果掩码与并行预测 (Causal Masking & Parallel Prediction)**

传统的 GPT 掩码：第 $t$ 个位置只能看前 $t-1$ 个位置，每次只能预测下一个位置的 **1** 个 Token。
VAR 的尺度掩码：在同一个尺度 $\mathbf{R_k}$ 内部，所有的 Token 是互可见的（可以互相看）。但是，高分辨率的尺度（如 $\mathbf{R_k}$）绝对看不到更高分辨率的尺度（如 $\mathbf{R_{k+1}}$）。

训练目标：
给定 $\mathbf{R_1}$，Transformer 一次性（并行）预测出 $\mathbf{R_2}$ 矩阵中所有位置的几率分布。
给定 $\mathbf{R_1}$ 和 $\mathbf{R_2}$，Transformer 一次性（并行）预测出 $\mathbf{R_3}$ 矩阵中所有位置的几率分布。
依此类推，用交叉熵损失函数（Cross-Entropy Loss）指导训练。

### 阶段三：推理阶段
* **Step 1:** 给模型一个起始信号（如类标签“狗”或起始 Token）。模型首先预测出最粗糙的尺度 $R_1$（只有一个 Token，代表这只狗的大致色调和位置）。
* **Step 2:** 把 $R_1$ 输入给 Transformer，模型直接**并行输出**下一个尺度 $R_2$ 的整个 $2 \times 2$ 矩阵（狗的轮廓显现）。
* **Step 3:** 把 $[R_1, R_2]$ 共同输入，模型并行输出 $R_3$ 的 $3 \times 3$ 矩阵。
* ......
* **Step $K$ (第7步):** 把前面生成的所有低分辨率 Token 输入，模型瞬间并行输出最高分辨率 $R_K$（ $16 \times 16 = 256$个 Token，毛发、眼睛等细节全部补齐）。

**最终转换：** 把这 $K$ 个预测出来的 Token 矩阵丢进第一阶段的 VQ-VAE Decoder，解码器一瞬间就吐出了一张高清的 $256 \times 256$ 真实图像。

## 个人思考
**连续 vs 离散：** 自回归模型（AR）在离散 Token 上的攻击，扩散模型在连续隐空间的特征融合（MSE优化）让投毒数据。

---
- *SDEdit 为什么能找到合理位置？* 利用了扩散模型预训练时见过海量“商品上带有Logo”的先验知识。
- *计算成本如何？* 制作数据集成本较高（需多次推理），但攻击（微调）成本与正常训练无异。

