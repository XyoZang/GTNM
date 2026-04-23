# TensorFlow 1.x → 2.x 快速兼容迁移计划 (方案A: tf.compat.v1)

## 📋 方案概述

### 目标
使用 TensorFlow 2.x 的 `tf.compat.v1` 兼容层，以**最小改动**将现有 TF 1.x 代码迁移到 TF 2.x，解决 RTX 3090 环境配置问题。

### 核心策略
- ✅ **保持原有代码架构不变**
- ✅ **仅替换不兼容的 API 调用**
- ✅ **添加 TF 2.x 兼容性导入和配置**
- ✅ **预计工作量**: 2-3 小时

---

## 🎯 方案A vs 方案B 对比

| 维度 | 方案A (快速兼容) | 方案B (完整Keras重构) |
|------|------------------|---------------------|
| **代码改动量** | ⭐ 低 (10-20%) | ⭐⭐⭐⭐ 高 (70-80%) |
| **工作时间** | 2-3 小时 | 6-8 小时 |
| **风险等级** | ⭐ 低 | ⭐⭐⭐ 中 |
| **性能提升** | 基础 (利用新CUDA) | 显著 (混合精度+XLA) |
| **可维护性** | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 优秀 |
| **适用场景** | 快速解决问题 | 长期项目优化 |

**选择方案A的理由**:
1. ✅ 快速解决 RTX 3090 兼容性问题
2. ✅ 保持现有代码逻辑完全不变
3. ✅ 降低引入 bug 的风险
4. ✅ 后续可以渐进式升级到方案B

---

## 🔧 迁移步骤详解

### 步骤 1: 创建新的 conda 环境 (30 分钟)

#### 1.1 创建环境
```bash
conda create -n gtnm_tf2 python=3.8 -y
conda activate gtnm_tf2
```

> **注意**: 保持 Python 3.8 以确保与现有依赖（sentencepiece, tree-sitter等）兼容

#### 1.2 安装 TensorFlow 2.x (Windows GPU 版本)

**推荐版本**: TensorFlow 2.10 (最后一个原生支持 Windows GPU 的稳定版)

```bash
pip install tensorflow==2.10.0
pip install numpy sentencepiece tree-sitter tree-sitter-languages tqdm fuzzywuzzy pathos
```

**或者** 如果需要更新的 CUDA 支持 (11.8+):

```bash
# 方案1: 使用 tensorflow-cpu + 单独安装 CUDA 库 (较复杂)
pip install tensorflow-cpu==2.14.0
pip install nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cufft-cu11

# 方案2: 使用 WSL2 (推荐用于生产环境)
# 在 WSL2 Ubuntu 中:
pip install tensorflow[and-cuda]==2.14.0
```

#### 1.3 验证安装
创建测试文件 `test_tf2.py`:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
if tf.config.list_physical_devices('GPU'):
    print("GPU device:", tf.config.list_physical_devices('GPU')[0])
```

运行:
```bash
python test_tf2.py
```

预期输出:
```
TensorFlow version: 2.10.0
GPU available: True
GPU device: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

---

### 步骤 2: 添加全局兼容性配置 (15 分钟)

在所有导入 TensorFlow 的文件顶部添加以下配置。

#### 需要修改的文件清单:
1. ✅ `model/train.py`
2. ✅ `model/test.py`
3. ✅ `model/model_invoked.py`
4. ✅ `model/modules.py`
5. ✅ `model/utils.py`
6. ✅ `gpu_init_test.py`

#### 统一修改模板

在每个文件的 **`import tensorflow as tf` 之前**添加:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 或者更简洁的方式 (推荐):
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

**示例 - train.py 修改后**:

```python
# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processing"))

# ====== TF 2.x 兼容性配置开始 ======
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # 禁用 Eager 模式，保持图模式
# ====== TF 2.x 兼容性配置结束 ======

from model_invoked import Transformer
from extract_data_subword import *
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
from hparams import Hparams
import math
import logging
import time
from datetime import timedelta

logging.basicConfig(level=logging.INFO)

# ... 其余代码保持不变 ...
```

---

### 步骤 3: 替换已废弃的 API (45 分钟)

虽然大多数 TF 1.x API 在 `tf.compat.v1` 中仍然可用，但部分 API 需要显式替换：

#### 3.1 `modules.py` 中的关键替换

##### a) 初始化器替换 (第48行, 第68行)

```python
# ❌ TF 1.x (已废弃)
initializer=tf.contrib.layers.xavier_initializer()

# ✅ TF 2.x 兼容写法
initializer=tf.keras.initializers.GlorotUniform()
# 或
initializer=tf.initializers.glorot_uniform()
```

**位置**:
- [modules.py:48](file:///d:/localProject/GTNM/model/modules.py#L48) - `get_token_embeddings` 函数
- [modules.py:68](file:///d:/localProject/GTNM/model/modules.py#L68) - `get_doc_embeddings` 函数

##### b) 类型转换函数 (第142行, 第306行)

```python
# ❌ TF 1.x (已移除)
tf.to_float(key_masks)    # 第142行
tf.to_int32(tf.argmax(...))  # model_invoked.py 第241行
tf.to_float(tf.not_equal(...))  # model_invoked.py 第261行
tf.to_float(outputs)  # modules.py 第306行

# ✅ TF 2.x 兼容写法
tf.cast(key_masks, tf.float32)
tf.cast(tf.argmax(...), tf.int32)
tf.cast(tf.not_equal(...), tf.float32)
tf.cast(outputs, tf.float32)
```

**位置列表**:
- [modules.py:142](file:///d:/localProject/GTNM/model/modules.py#L142) - `mask` 函数中的 `key_masks`
- [modules.py:306](file:///d:/localProject/GTNM/model/modules.py#L306) - `positional_encoding` 函数
- [model_invoked.py:241](file:///d:/localProject/GTNM/model/model_invoked.py#L241) - `decode` 函数中的 `y_hat`
- [model_invoked.py:261](file:///d:/localProject/GTNM/model/model_invoked.py#L261) - `train` 函数中的 `nonpadding`

##### c) Dropout 层参数调整

```python
# ⚠️ TF 1.x 写法 (仍然可用，但会警告)
tf.layers.dropout(cxt_enc, self.hp.dropout_rate, training=training)

# ✅ 推荐的 TF 2.x 写法 (消除警告)
tf.compat.v1.layers.dropout(cxt_enc, rate=self.hp.dropout_rate, training=training)
```

**位置** (共6处):
- [model_invoked.py:88](file:///d:/localProject/GTNM/model/model_invoked.py#L88)
- [model_invoked.py:104](file:///d:/localProject/GTNM/model/model_invoked.py#L104)
- [model_invoked.py:116](file:///d:/localProject/GTNM/model/model_invoked.py#L116)
- [model_invoked.py:186](file:///d:/localProject/GTNM/model/model_invoked.py#L186)
- [modules.py:113](file:///d:/localProject/GTNM/model/modules.py#L113)

##### d) Dense 层参数调整

```python
# ⚠️ TF 1.x 写法
tf.layers.dense(queries, d_model, use_bias=True)

# ✅ 推荐写法
tf.compat.v1.layers.dense(queries, d_model, use_bias=True)
```

**位置** (共6处):
- [modules.py:190-192](file:///d:/localProject/GTNM/model/modules.py#L190-L192) - Q, K, V 投影
- [modules.py:224-227](file:///d:/localProject/GTNM/model/modules.py#L224-L227) - FFN 层

---

### 步骤 4: 处理 Session 和 ConfigProto (30 分钟)

#### 4.1 `train.py` 中的 Session 配置 (第86-89行)

```python
# ❌ TF 1.x 写法
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

with tf.Session(config=gpu_config) as sess:
    sess.run(tf.global_variables_initializer())

# ✅ TF 2.x 兼容写法
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=gpu_config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
```

**或者** 更简洁的方式 (使用别名):

```python
# 在文件顶部已经设置了 tf = tf.compat.v1 的情况下
# 所有 tf.xxx 调用都自动映射到 compat.v1.xxx
# 所以实际上代码可以基本保持不变！

# 只需确保导入了:
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

#### 4.2 `test.py` 中的 Session 配置 (第36-37行)

同样处理:
```python
# 修改前
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# 修改后 (如果使用了 tf = tf.compat.v1 别名则无需改)
# 如果没有使用别名:
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
```

---

### 步骤 5: 其他细节调整 (30 分钟)

#### 5.1 Saver 和 Checkpoint (无需改动)

`tf.train.Saver()` 在 TF 2.x 中仍然可用，checkpoint 格式也兼容。

✅ **无需修改**

#### 5.2 Summary 相关 (无需改动)

```python
tf.summary.scalar('lr', lr)
tf.summary.merge_all()
tf.summary.FileWriter(hp.logdir, sess.graph)
```

这些在 `tf.compat.v1` 下都可以正常工作。

✅ **无需修改**

#### 5.3 Optimizer (model_invoked.py 第266行)

```python
# 当前代码已经使用了兼容写法 ✅
optimizer = tf.compat.v1.train.AdamOptimizer(lr)
```

**已经是正确的！无需修改。**

#### 5.4 Loss 函数 (model_invoked.py 第260行)

```python
# 当前代码
ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)

# ✅ TF 2.x 中这个函数仍然可用
# 但也可以改为标准名称 (功能相同):
ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
```

**建议**: 改为标准名称以避免未来警告，但不影响运行。

#### 5.5 global_step (model_invoked.py 第264行)

```python
# 当前代码
global_step = tf.train.get_or_create_global_step()

# ✅ 兼容写法
global_step = tf.compat.v1.train.get_or_create_global_step()
```

---

### 步骤 6: 完整的文件修改清单

#### 📄 文件 1: `model/modules.py`

**修改点汇总** (共 8 处):

| 行号 | 原代码 | 修改后 | 原因 |
|------|--------|--------|------|
| 48 | `tf.contrib.layers.xavier_initializer()` | `tf.keras.initializers.GlorotUniform()` | contrib 已移除 |
| 68 | `tf.contrib.layers.xavier_initializer()` | `tf.keras.initializers.GlorotUniform()` | contrib 已移除 |
| 142 | `tf.to_float(key_masks)` | `tf.cast(key_masks, tf.float32)` | to_float 已移除 |
| 190-192 | `tf.layers.dense(...)` | `tf.compat.v1.layers.dense(...)` | 消除警告 |
| 224-225 | `tf.layers.dense(...)` | `tf.compat.v1.layers.dense(...)` | 消除警告 |
| 113 | `tf.layers.dropout(...)` | `tf.compat.v1.layers.dropout(...)` | 消除警告 |
| 306 | `tf.to_float(outputs)` | `tf.cast(outputs, tf.float32)` | to_float 已移除 |

**额外添加** (文件开头):
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

---

#### 📄 文件 2: `model/model_invoked.py`

**修改点汇总** (共 11 处):

| 行号 | 原代码 | 修改后 | 原因 |
|------|--------|--------|------|
| 88 | `tf.layers.dropout(...)` | `tf.compat.v1.layers.dropout(...)` | 消除警告 |
| 104 | `tf.layers.dropout(...)` | `tf.compat.v1.layers.dropout(...)` | 消除警告 |
| 116 | `tf.layers.dropout(...)` | `tf.compat.v1.layers.dropout(...)` | 消除警告 |
| 186 | `tf.layers.dropout(...)` | `tf.compat.v1.layers.dropout(...)` | 消除警告 |
| 241 | `tf.to_int32(...)` | `tf.cast(..., tf.int32)` | to_int32 已移除 |
| 260 | `softmax_cross_entropy_with_logits_v2` | `softmax_cross_entropy_with_logits` | 标准化命名 |
| 261 | `tf.to_float(...)` | `tf.cast(..., tf.float32)` | to_float 已移除 |

**可选优化** (不影响运行):
- 第264行: `tf.train.get_or_create_global_step()` → `tf.compat.v1.train.get_or_create_global_step()`

**额外添加** (文件开头):
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

---

#### 📄 文件 3: `model/train.py`

**修改点汇总** (共 3 处):

| 行号 | 原代码 | 修改后 | 原因 |
|------|--------|--------|------|
| 86-87 | `tf.ConfigProto()` | `tf.compat.v1.ConfigProto()` | 明确兼容性 |
| 89 | `tf.Session(config=...)` | `tf.compat.v1.Session(config=...)` | 明确兼容性 |
| 90 | `tf.global_variables_initializer()` | `tf.compat.v1.global_variables_initializer()` | 明确兼容性 |

**额外添加** (第10行之后):
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

**注意**: 需要删除或注释掉原来的 `import tensorflow as tf` (第10行)，避免重复导入。

---

#### 📄 文件 4: `model/test.py`

**修改点汇总** (共 3 处):

| 行号 | 原代码 | 修改后 | 原因 |
|------|--------|--------|------|
| 36 | `tf.Session()` | `tf.compat.v1.Session()` | 明确兼容性 |
| 37 | `tf.global_variables_initializer()` | `tf.compat.v1.global_variables_initializer()` | 明确兼容性 |

**额外添加** (第7行之后):
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

---

#### 📄 文件 5: `model/utils.py`

**修改点汇总** (共 1 处):

| 行号 | 原代码 | 修改后 | 原因 |
|------|--------|--------|------|
| 133 | `tf.global_variables()` | `tf.compat.v1.global_variables()` | 明确兼容性 |

**额外添加** (文件开头):
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

---

#### 📄 文件 6: `gpu_init_test.py`

**修改点汇总** (共 5 处):

| 行号 | 原代码 | 修改后 | 原因 |
|------|--------|--------|------|
| 5 | `tf.ConfigProto()` | `tf.compat.v1.ConfigProto()` | 明确兼容性 |
| 10 | `tf.Session(config=config)` | `tf.compat.v1.Session(config=config)` | 明确兼容性 |

**额外添加** (第1行之后):
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

---

### 步骤 7: 测试验证 (45 分钟)

#### 7.1 基础环境测试

```bash
cd d:\localProject\GTNM
conda activate gtnm_tf2
python gpu_init_test.py
```

**预期输出**:
```
1. Import done
2. Creating session...
3. Session created!
4. Running matmul...
5. Result shape: (10, 10)
6. Done!
```

#### 7.2 模型构建测试

创建 `test_model_build.py`:

```python
# -*- coding: utf-8 -*-
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processing"))

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from model_invoked import Transformer
from extract_data_subword import *
from hparams import Hparams

print("=" * 60)
print("Testing Model Construction with TF 2.x Compatibility")
print("=" * 60)

hparams = Hparams()
hp = hparams.parser.parse_args([])

try:
    print("\n[1/3] Loading data...")
    data = localContext(
        hp.body_context_size,
        hp.doc_context_size,
        hp.project_context_size,
        hp.tgt_name_size,
        hp.sub_word_vocab_file,
        hp.doc_vocab_file,
        include_docstring=True,
        expr_max_len=1024,
        expr_max_num=30,
        datapath=hp.data_path
    )
    print("      ✅ Data loaded successfully")

    print("\n[2/3] Building model graph...")
    m = Transformer(hp)
    loss, train_op, global_step, summaries, preds = m.train()
    print("      ✅ Model graph built successfully")
    print(f"      - Loss node: {loss.name}")
    print(f"      - Train op: {train_op.name}")
    print(f"      - Global step: {global_step.name}")

    print("\n[3/3] Testing forward pass (dry run)...")
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=gpu_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Create dummy data
        import numpy as np
        feed_dict = {
            m.body_batch: np.random.randint(0, len(data.w2id), (hp.batch_size, hp.body_context_size)),
            m.pro_batch: np.random.randint(0, len(data.w2id), (hp.batch_size, hp.project_context_size)),
            m.doc_batch: np.random.randint(0, len(data.doc_w2id), (hp.batch_size, hp.doc_context_size)),
            m.invoked_batch: np.random.rand(hp.batch_size, hp.project_context_size).astype(np.float32),
            m.dec_inp_batch: np.random.randint(0, len(data.w2id), (hp.batch_size, data.tgt_name_len)),
            m.dec_tgt_batch: np.random.randint(0, len(data.w2id), (hp.batch_size, data.tgt_name_len))
        }

        _loss, _gs = sess.run([loss, global_step], feed_dict=feed_dict)
        print(f"      ✅ Forward pass successful!")
        print(f"      - Loss value: {_loss:.4f}")
        print(f"      - Global step: {_gs}")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! TF 2.x migration successful!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

运行测试:
```bash
python test_model_build.py
```

#### 7.3 完整训练流程测试

```bash
# 使用原始训练命令
python model/train.py --num_epochs 1 --batch_size 16
```

**预期结果**:
- 能够成功启动训练
- GPU 利用率正常 (可在任务管理器中查看)
- Loss 正常下降
- 无关键错误或崩溃

---

## 📊 工作量与时间估算

| 任务 | 预计时间 | 优先级 |
|------|----------|--------|
| 步骤 1: 环境搭建 | 30 分钟 | P0 (必须) |
| 步骤 2: 全局兼容性配置 | 15 分钟 | P0 (必须) |
| 步骤 3: API 替换 (modules.py) | 20 分钟 | P0 (必须) |
| 步骤 3: API 替换 (model_invoked.py) | 25 分钟 | P0 (必须) |
| 步骤 4: Session/Config 处理 | 15 分钟 | P0 (必须) |
| 步骤 5: 其他细节调整 | 15 分钟 | P1 (推荐) |
| 步骤 6: 文件修改执行 | 30 分钟 | P0 (必须) |
| 步骤 7: 测试验证 | 45 分钟 | P0 (必须) |
| **总计** | **~3 小时** | - |

---

## ✅ 成功标准

迁移成功的标志:

1. ✅ **环境安装成功**: `tensorflow==2.10.0` 安装完成，检测到 RTX 3090
2. ✅ **模型构建成功**: 无报错地创建 Transformer 计算图
3. ✅ **前向传播成功**: 能够执行一次 dummy 数据的前向传播并得到合理的 loss 值
4. ✅ **训练流程启动**: 能够至少运行 1 个 epoch 的训练
5. ✅ **GPU 加速生效**: 训练速度合理 (不应明显慢于原版)
6. ✅ **无关键错误**: 可以有 deprecation warnings，但不能有运行时错误

---

## ⚠️ 已知问题与解决方案

### 问题 1: Deprecation Warnings (警告信息)

**现象**: 运行时会出现大量警告如:
```
WARNING:tensorflow:From xxx: xx is deprecated and will be removed in a future version.
```

**解决方案**:
- ✅ 这些**不影响程序运行**，只是提示未来版本可能移除
- 可通过环境变量抑制:
  ```python
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 显示 FATAL only
  ```
- 或者在代码中忽略特定警告:
  ```python
  import warnings
  warnings.filterwarnings('ignore', category=DeprecationWarning)
  ```

### 问题 2: Windows GPU 支持限制

**现象**: TF 2.11+ 不再提供 Windows 原生 GPU 支持

**解决方案**:
- ✅ **推荐**: 使用 TF 2.10 (本方案采用的版本)
- ✅ **备选**: 使用 WSL2 + Linux 环境 + TF 2.12+
- ✅ **备选**: 使用 `tensorflow-cuda` 第三方包

### 问题 3: 性能差异

**预期**: 性能与原版相当或略有提升 (得益于 CUDA 11.x 优化)

**如果性能下降**:
1. 检查是否启用了 GPU:
   ```python
   print(tf.config.list_physical_devices('GPU'))
   ```
2. 确保 `allow_growth=True` 设置正确
3. 尝试启用 XLA 编译 (可选):
   ```python
   tf.config.optimizer.set_jit(True)
   ```

### 问题 4: Checkpoint 加载

**好消息**: TF 2.x 的 `tf.compat.v1.train.Saver` 完全兼容 TF 1.x 格式的 checkpoint！

**验证方法**:
```python
saver = tf.compat.v1.train.Saver()
ckpt = tf.compat.v1.train.latest_checkpoint('./saved')
if ckpt:
    saver.restore(sess, ckpt)
    print(f"✅ Successfully restored from {ckpt}")
else:
    print("ℹ️ No checkpoint found, training from scratch")
```

---

## 🚀 执行顺序总结

### Phase 1: 准备工作 (45分钟)
1. ✅ 创建 conda 环境 `gtnm_tf2`
2. ✅ 安装 TensorFlow 2.10 及依赖
3. ✅ 验证 GPU 检测和环境配置

### Phase 2: 代码修改 (90分钟)
4. ✅ 修改 `modules.py` (8处API替换)
5. ✅ 修改 `model_invoked.py` (11处API替换)
6. ✅ 修改 `train.py` (3处Session相关)
7. ✅ 修改 `test.py` (2处Session相关)
8. ✅ 修改 `utils.py` (1处变量获取)
9. ✅ 修改 `gpu_init_test.py` (2处Session相关)

### Phase 3: 测试验证 (45分钟)
10. ✅ 运行 `gpu_init_test.py` 验证基础环境
11. ✅ 运行 `test_model_build.py` 验证模型构建
12. ✅ 运行短时间训练 (1 epoch) 验证完整流程
13. ✅ 检查 GPU 利用率和训练速度

---

## 📝 后续可选优化 (非必须)

完成基础迁移后，如果时间和需求允许，可以考虑以下渐进式优化：

### 优化 1: 启用混合精度训练 (预计 30分钟)

```python
# 在 train.py 开头添加
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
```

**收益**: RTX 3090 训练速度提升 2-3倍

### 优化 2: 启用 XLA 编译 (预计 15分钟)

```python
# 在创建 Session 前
tf.config.optimizer.set_jit(True)
```

**收益**: 进一步加速计算图执行

### 优化 3: 升级到 Keras Model (方案B) (预计 6-8小时)

当项目稳定运行后，可以逐步将代码重构为标准的 Keras Model 架构，获得更好的可维护性和性能。

---

## 💡 总结

### 为什么选择方案A？

✅ **快速见效**: 2-3小时即可完成迁移
✅ **低风险**: 保持原有架构，最小化代码改动
✅ **立即可用**: 解决 RTX 3090 兼容性痛点
✅ **平滑过渡**: 为将来可能的方案B升级铺路

### 最终交付物

1. ✅ 完整的 TF 2.10 兼容代码 (6个文件)
2. ✅ 可复现的环境配置说明
3. ✅ 测试脚本和验证报告
4. ✅ 问题排查指南

---

## 🎯 准备好开始了吗？

如果您确认此计划，我将立即按照以下顺序执行：

1. **首先**修改所有 Python 文件，添加 TF 2.x 兼容性配置
2. **然后**逐一替换已废弃的 API 调用
3. **最后**创建测试脚本并验证迁移效果

整个过程预计 **2-3小时** 完成，您将拥有一个可以在 RTX 3090 上顺利运行的 TF 2.x 版本 GTNM 项目！
