# Claude Code Agent Harness

教学级 Claude Code 编码代理（Coding Agent）的 Python 参考实现，复刻了 Claude Code CLI 的核心架构。适合学习 AI 编码代理的内部工作原理。

## 项目简介

本项目实现了一个交互式 CLI 编码代理，包含完整的工具调用循环、权限控制、上下文管理、后台任务调度、多代理协作等子系统。用户通过命令行输入自然语言任务描述，代理通过 Anthropic Claude API 进行工具调用，完成文件操作、命令执行、任务管理等工作。

## 技术栈

- **语言**: Python 3
- **LLM SDK**: Anthropic Python SDK (`anthropic>=0.25.0`)
- **可观测性**: LangSmith (tracing)
- **配置**: python-dotenv, PyYAML
- **版本控制**: Git (worktree 管理)

## 技术架构

```
claude-code-agent-harness/
├── agent_main.py          # 主入口：交互式 CLI + agent loop
├── tools.py               # 工具定义与原生处理器（30+ 工具）
├── log.py                 # 生产级结构化日志系统
├── dao/
│   └── anthropic_utils.py # Anthropic 客户端（LangSmith wrapped）
├── services/
│   └── agent_service.py   # 工具池组装、权限执行、Hook 编排
├── handlers/
│   ├── system_prompt.py   # 动态系统提示词构建器
│   ├── permission_system.py# 权限门控（default/plan/auto 三种模式）
│   ├── compact_context.py # 上下文压缩、内容持久化
│   ├── error_recovery.py  # API 错误恢复（退避、自动压缩）
│   ├── background_tasks.py# 后台任务线程管理
│   ├── cron_scheduler.py  # Cron 定时任务调度（持久/会话级）
│   ├── mcp_plugin.py      # MCP 协议插件系统（stdio 传输）
│   ├── skill_loading.py   # 技能文档加载
│   ├── memory_system.py   # 持久化记忆管理（跨会话）
│   ├── tasks_system.py    # 任务图管理（CRUD + 依赖）
│   ├── team_system.py     # 多代理协作（消息总线 + 队友线程）
│   ├── hook_system.py     # Hook 系统（PreToolUse/PostToolUse/SessionStart）
│   ├── todomanager.py     # 会话计划管理
│   └── worktrees_task.py  # Git Worktree 管理（并行任务隔离）
├── settings/
│   └── constant.py        # 全局配置常量
├── skills/                # 内置技能文档
└── .memory/               # 持久化记忆存储
```

### 核心子系统

| 子系统 | 说明 |
|--------|------|
| **Agent Loop** | 主循环：接收用户输入 → 构建系统提示 → 调用 Claude API → 执行工具 → 返回结果 |
| **Tool Pool** | 30+ 原生工具 + MCP 外部工具，统一权限门控 |
| **Permission Gate** | 三级模式：`default`（询问写入）/ `plan`（只读）/ `auto`（自动批准） |
| **Context Management** | 微压缩（micro-compact）+ 全量压缩（compact）+ 大输出持久化到磁盘 |
| **Error Recovery** | `max_tokens` 续写 + `prompt_too_long` 自动压缩 + 指数退避重试 |
| **Cron Scheduler** | 5 字段 cron 表达式，支持持久化存储，启动时检测遗漏任务 |
| **Teammate System** | 多代理协作：消息总线通信、任务认领、计划审批、关闭协议 |
| **Worktree Manager** | Git worktree 创建/进入/执行/删除，绑定任务 ID，事件审计 |
| **Hook System** | PreToolUse / PostToolUse / SessionStart 钩子，返回码控制阻塞/消息注入 |
| **MCP Plugin** | 加载 .claude-plugin/ 插件，通过 stdio MCP 协议连接外部工具服务器 |
| **Memory System** | 单文件 Markdown 格式，类型含 user/feedback/project/reference |

### 工具总览

原生工具 30+ 个，涵盖：文件操作（`read_file`/`write_file`/`edit_file`）、命令执行（`bash`/`background_run`）、任务管理（`task_create`/`task_update`/`task_list`/`claim_task`）、定时任务（`cron_create`/`cron_delete`/`cron_list`）、团队协作（`spawn_teammate`/`send_message`/`broadcast`/`plan_approval`）、Worktree（`worktree_create`/`worktree_run`/`worktree_remove`/`worktree_keep`）、记忆（`save_memory`）、技能（`load_skill`）、计划（`todo`）、上下文压缩（`compress`）等。

## 环境要求

- Python 3.10+
- Git
- Anthropic API Key（或兼容的 API 端点）

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd claude-code-agent-harness
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件：

```bash
ANTHROPIC_BASE_URL=https://api.anthropic.com   # API 端点
ANTHROPIC_API_KEY=sk-ant-xxx                    # 你的 API Key
MODEL_ID=claude-sonnet-4-6                      # 模型 ID
LOG_LEVEL=INFO                                   # 日志级别（可选）
```

关键环境变量说明：

| 变量 | 说明 | 必填 |
|------|------|------|
| `ANTHROPIC_BASE_URL` | Anthropic API 端点地址 | 是 |
| `ANTHROPIC_API_KEY` | API 密钥 | 是 |
| `MODEL_ID` | 使用的模型 ID | 是 |
| `LOG_LEVEL` | 日志级别 (DEBUG/INFO/WARNING/ERROR) | 否 |

### 4. 启动

```bash
python agent_main.py
```

启动后选择权限模式（`default` / `plan` / `auto`），进入交互式命令行 `s14 >>`。

## 交互命令

| 命令 | 说明 |
|------|------|
| 直接输入文本 | 发送给 AI 代理处理 |
| `/mode <模式>` | 切换权限模式 |
| `/tools` | 列出所有可用工具 |
| `/mcp` | 列出已连接的 MCP 服务器 |
| `/cron` | 查看定时任务列表 |
| `/team` | 查看队友状态 |
| `/tasks` | 查看任务面板 |
| `/inbox` | 查看收件箱消息 |
| `/prompt` | 打印当前系统提示词 |
| `/sections` | 查看系统提示词章节 |
| `q` / `exit` | 退出程序 |

## 配置说明

主要配置项在 `settings/constant.py`，包括：

- **权限模式**: `MODES = ("default", "plan", "auto")`
- **上下文限制**: `CONTEXT_LIMIT = 50000` 字符
- **压缩阈值**: `TOKEN_THRESHOLD = 50000` 字符
- **最大重试次数**: `MAX_RECOVERY_ATTEMPTS = 3`
- **退避延迟**: `BACKOFF_BASE_DELAY = 1.0s`, `BACKOFF_MAX_DELAY = 30.0s`
- **定时任务**: 7 天自动过期 (`AUTO_EXPIRY_DAYS = 7`)
- **Hook 超时**: `HOOK_TIMEOUT = 30s`

Hook 配置文件为项目根目录下的 `.hooks.json`，MCP 插件配置为 `.claude-plugin/plugin.json`。

## 项目特色

- **教学导向**: 代码结构清晰、注释丰富，每个子系统都可独立理解
- **生产级日志**: JSON 格式化器 + RotatingFileHandler + 日志上下文适配器
- **LangSmith 集成**: `agent_loop` 通过 `@traceable` 装饰器实现全链路追踪
- **完整的多代理协作**: 消息总线 + 任务认领 + 计划审批 + 关闭协议
- **Git Worktree 隔离**: 并行任务在独立 worktree 中执行，支持事件审计
