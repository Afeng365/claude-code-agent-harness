# production_logger.py
import os
import sys
import json
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional
from logging import LoggerAdapter


# ================= 1. 自定义 JSON 格式化器（结构化日志） =================
class JsonFormatter(logging.Formatter):
    """生产环境推荐的结构化日志输出，便于 ELK/Loki 等日志系统采集"""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }

        # 捕获异常堆栈
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # 注入自定义上下文字段（如 request_id, user_id, trace_id 等）
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in log_obj:
                log_obj[key] = value

        return json.dumps(log_obj, ensure_ascii=False)


# ================= 2. 日志上下文适配器 =================
class LogContext(LoggerAdapter):
    """用于在请求链路上透传上下文信息（如 request_id, tenant_id）"""

    def process(self, msg: str, kwargs: dict) -> tuple:
        # 将上下文注入到日志记录的 extra 中
        kwargs.setdefault("extra", {}).update(self.extra)
        return msg, kwargs


# ================= 3. 生产级日志初始化函数 =================
def setup_production_logger(
        app_name: str = "myapp",
        log_dir: str = "logs",
        level: str = "INFO",
        max_bytes: int = 50 * 1024 * 1024,  # 50MB/文件
        backup_count: int = 10,
        use_json: bool = False,
        console: bool = False
) -> logging.Logger:
    """初始化生产环境日志配置"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    datefmt = "%Y-%m-%d %H:%M:%S"
    text_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s"
    detail_fmt = "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,  # ⚠️ 生产关键：不要禁用第三方库日志
        "formatters": {
            "standard": {"format": text_fmt, "datefmt": datefmt},
            "detailed": {"format": detail_fmt, "datefmt": datefmt},
            "json": {"()": JsonFormatter, "datefmt": datefmt}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "json" if use_json else "standard",
                "stream": "ext://sys.stdout"
            },
            "file_info": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json" if use_json else "detailed",
                "filename": str(log_path / f"{app_name}.log"),
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "encoding": "utf-8",
                "delay": True  # 延迟创建文件，避免空跑占用 fd
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json" if use_json else "detailed",
                "filename": str(log_path / f"{app_name}_error.log"),
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "encoding": "utf-8",
                "delay": True
            }
        },
        "root": {
            "level": level,
            "handlers": ["console", "file_info", "file_error"] if console else ["file_info", "file_error"]
        }
    }

    # 应用配置
    logging.config.dictConfig(log_config)
    logging.captureWarnings(True)  # 捕获 warnings.warn 输出

    return logging.getLogger(app_name)


logging = setup_production_logger(app_name="agent",
                                  log_dir="logs",
                                  level=os.getenv("LOG_LEVEL", "INFO"),
                                  use_json=False  # 本地调试用文本，生产建议改为 True)
                                  )

# ================= 4. 使用示例 =================
if __name__ == "__main__":
    # 1. 初始化日志
    log = setup_production_logger(
        app_name="order_service",
        log_dir="logs",
        level=os.getenv("LOG_LEVEL", "INFO"),
        use_json=False  # 本地调试用文本，生产建议改为 True
    )

    #   2. 基础日志
    log.info("服务启动成功，版本: %s", "v2.1.0")
    log.warning("配置项未找到，使用默认值: %s", "timeout=30s")

    # 3. 异常日志（自动附加堆栈）
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("数据库连接失败，正在重试...")

        # 4. 上下文透传（推荐在 Web 框架中间件中注入 request_id）
    context = LogContext(log, {"request_id": "req_8f3a9c21", "user_id": 10045})
    context.info("开始处理订单", extra={"amount": 299.0, "currency": "CNY"})
    context.debug("调用支付网关", extra={"endpoint": "https://pay.example.com/v2"})

    # 5. 多模块共享同一配置
    other_log = logging.getLogger("order_service.cache")
    other_log.info("缓存预热完成，命中 keys: %d", 1240)
