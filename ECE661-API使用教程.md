# ECE 661 课程 API / TPU 使用教程

> 来源：课程文档站 https://ece661-26fa-tpu.github.io （Canvas 公告里给的链接）。
> 课程 API 地址：`https://34-127-166-57.sslip.io/api/v1`（OpenAI 兼容）。

## 0. 先确认你手上是哪把 key

`ece661-key.json` 里的 `scope` 写的是 **`TPU workspace only`**，Yueqian 的邮件也说这把 key 不含 Gemini 额度。
**但实测（2026-10-23）这把 key 可以调 Gemini**：`/models` 列出了 18 个模型（包括 `gemini-4-argon`），
用 `gemini-2.5-flash-lite` 发一条请求，返回 HTTP 200。
它很可能用的是组里的 Gemini 额度，用之前先去 My usage / My group 页面看看余额。

key 不要提交到 GitHub，不要贴到聊天或截图里。这个文件夹以后要是变成 git 仓库，先把 `ece661-key.json` 加进 `.gitignore`。

## 1. 测试 key 能不能用

最简单的方法：打开 https://ece661-26fa-tpu.github.io → **My usage** → 粘贴 key。
页面会显示预算、已花费、能调用哪些模型、你的 TPU session。

命令行测试（Git Bash / WSL）：

```bash
export ECE661_API_KEY=$(python -c "import json;print(json.load(open('ece661-key.json'))['key'])")
curl https://34-127-166-57.sslip.io/api/v1/models -H "Authorization: Bearer $ECE661_API_KEY"
```

- 返回模型列表 → key 可以调 API。
- 返回 401/403 或预算错误 → 这把 key 没有 Gemini 额度（TPU-only），去换邮件里的那把 API key。
  注意：文档说预算为 0 的 key 也调不了 `/models`。

## 2. 调用 Gemini（Python）

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://34-127-166-57.sslip.io/api/v1",
    api_key=os.environ["ECE661_API_KEY"],
)
r = client.chat.completions.create(
    model="MODEL_NAME",   # 从 My usage 页面或 /models 里选，不要写死
    messages=[{"role": "user", "content": "Hello!"}],
)
print(r.choices[0].message.content)
```

可用接口：`/chat/completions`（支持 `stream=True`）、`/embeddings`、`/images/generations`、`/models`。
全组共用一个 Gemini 预算，用完了组里每个人都会同时报错。

## 3. 使用 TPU

`ece661` 命令行只支持 macOS / Linux / **WSL**。你是 Windows，要先装 WSL（`wsl --install`）。

安装（建议先下载下来看一眼脚本，再运行，不要直接 `curl | sh`）：

```bash
curl -fsSL https://34-127-166-57.sslip.io/broker/cli/install.sh -o install.sh
less install.sh
sh install.sh
```

登录并开机：

```bash
ssh-keygen -t ed25519                      # 没有 SSH key 才需要
ece661 login                               # 在隐藏提示符里粘贴 key
ece661 ssh-key add ~/.ssh/id_ed25519.pub
ece661 compute start --shape v6e-1 --hours 4   # 会先显示费用，输入 y 确认
ece661 compute list                        # 查看状态和 SSH 地址
ssh ece661-xxxxxxxx                        # 也可以用 VS Code Remote-SSH 连
ece661 compute stop <id>                   # 用完一定要关
```

> Yueqian 的邮件写的是旧的 pilot 命令（`ece661 login --key-file ...`、`ece661 workspace start/status/pause`）。
> 如果 `compute` 命令报错，再试邮件里的写法。两边说法不一样时以 TA 的回复为准。

### 存数据

- 机器到期就会**连硬盘一起删掉**。checkpoint 存到组共享 bucket `/mnt/group`（文档的说法），
  或者按邮件的说法：`~/project` + `ece661-checkpoint mark-ready`。
- 空闲 30 分钟自动删除（开机前 15 分钟不算）。跑长任务时，想让机器在空闲时也不被删：`touch /tmp/ece661-busy`，用完记得删掉。
- 跑长实验之前，先用小任务把“保存 → 停机 → 恢复”测一遍。

## 4. 能不能同时用多张 TPU / GPU？（2026-10-23 用你的 key 实测）

测试方法：调用 `GET /broker/v1/quote`，只检查请求能不能被批准，**不开机、不扣钱**。

| 规格 | 结果 |
|---|---|
| `v6e-1`（1 张 TPU） | ✅ 可以，$1.35/小时 |
| `v6e-4`（4 张 TPU） | ❌ "Your workspace allowance permits one v6e chip per session." |
| `v6e-8`（8 张 TPU） | ❌ 需要 TA 批准（去 Slack 问），而且你的额度只允许 1 张 |
| `l40s-1` / `h100-1` / `h100-8`（GPU） | ❌ HTTP 503 "GPU compute is not available right now. Ask a TA." |

你的账户（`GET /broker/v1/workspace`）：个人额度 $200，其中计算 $180，已用约 $1.23；
`max_chips = 1`，每个 session 最长 **8 小时**。

结论：
- **现在只能用 1 张 v6e TPU。** 要多卡（v6e-4/v6e-8）得 TA 开权限。
- **CLI 支持 GPU**（`ece661 compute start --gpu h100-1`，跑在 Nebius 上），但现在是关闭的，要找 TA。
- 这把 key 现在走的是 **个人 workspace 额度**（$200 / 8 小时 / 1 张卡），不是文档里写的组额度（$150 TPU / 168 小时）。
- 能不能同时开 2 个 session 没测（要真开机才知道，会花钱）。
- 需要更多资源：发邮件给 ece661-26fa-tas@duke.edu，写清楚要多少、用来干什么。

## 5. 遇到问题

课程 Slack 频道：https://ece661-2026fa.slack.com/archives/C0CA0JQJWTH
发 session ID 和报错信息，**不要发 key**。
