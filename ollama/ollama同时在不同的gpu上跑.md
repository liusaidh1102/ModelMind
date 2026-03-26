# Ollama 双GPU部署与模型隔离问题总结
## 一、核心问题回顾
1. **`ollama serve` 运行模式**：直接执行是**前台运行**，关闭终端即停止；需通过系统服务/`nohup`实现后台运行。
2. **端口冲突**：Ollama 默认系统服务占用 `11434` 端口，导致自定义GPU绑定实例启动失败（`Exit 1`）。
3. **模型“消失”真相**：**未删除任何模型**，仅因**运行用户+模型目录不同**导致服务无法读取。
4. **多GPU模型隔离**：需启动独立Ollama实例，绑定不同GPU和端口，实现单GPU单模型运行。
5. **高请求安全性**：请求过多仅会排队、变慢、报OOM错误，**GPU不会损坏**，可通过限流保护服务。

## 二、关键原理：模型目录隔离（核心根源）
不同启动方式对应**不同模型存储路径**，互不通用：
| 启动方式 | 运行用户 | 模型存储路径 |
|----------|----------|--------------|
| `sudo systemctl start ollama`（系统服务） | `ollama` | `/usr/share/ollama/.ollama/models` |
| `sudo ollama serve`（手动sudo启动） | `root` | `/root/.ollama/models` |
| `ollama serve`（普通用户启动） | `hist` | `/home/hist/.ollama/models` |

✅ **原有模型全部存放在：`/usr/share/ollama/.ollama/models`**

## 三、最终解决方案：双GPU后台部署（可直接复用）
### 1. 前置准备（关闭冲突服务）
```bash
# 杀掉所有Ollama进程
sudo pkill -f ollama
# 关闭默认系统服务（防止端口冲突）
sudo systemctl stop ollama
sudo systemctl disable ollama
```

### 2. 启动双GPU后台实例（绑定原有模型）
```bash
# GPU0 绑定端口11434，加载原有模型目录
sudo CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
OLLAMA_MAX_QUEUE=5 \
nohup ollama serve > ollama-gpu0.log 2>&1 &

# GPU1 绑定端口11435，加载原有模型目录
sudo CUDA_VISIBLE_DEVICES=1 \
OLLAMA_HOST=0.0.0.0:11435 \
OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
OLLAMA_MAX_QUEUE=5 \
nohup ollama serve > ollama-gpu1.log 2>&1 &
```
- `CUDA_VISIBLE_DEVICES`：绑定指定GPU
- `OLLAMA_MODELS`：强制读取原有模型
- `OLLAMA_MAX_QUEUE=5`：限制请求队列，保护GPU

### 3. 后台运行指定模型
```bash
# GPU0 后台运行 qwen2.5vl:7b
sudo OLLAMA_HOST=http://localhost:11434 \
nohup ollama run qwen2.5vl:7b > run-qwen2.5vl-gpu0.log 2>&1 &

# GPU1 后台运行 qwen2.5:14b
sudo OLLAMA_HOST=http://localhost:11435 \
nohup ollama run qwen2.5:14b > run-qwen2.5-14b-gpu1.log 2>&1 &
```

### 4. 验证与管理
```bash
# 查看模型（必须加sudo）
sudo ollama list

# 查看GPU占用
nvidia-smi

# 查看运行日志
tail -f ollama-gpu0.log
tail -f run-qwen2.5vl-gpu0.log

# 停止所有服务
sudo pkill -f ollama
```

## 四、重要结论
1. **模型安全**：所有模型未丢失，仅目录隔离导致读取不到，指定`OLLAMA_MODELS`即可恢复。
2. **双GPU隔离**：通过多实例+不同端口，实现`GPU0跑视觉模型、GPU1跑文本模型`，互不干扰。
3. **服务稳定性**：后台运行+请求限流，高并发下仅排队/拒绝请求，无硬件损坏风险。
4. **核心命令**：手动启动必须绑定**原有模型目录**，否则看不到历史模型。