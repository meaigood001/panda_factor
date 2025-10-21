# PandaFactor - iFlow CLI 指南

## 项目概述

PandaFactor 是一个高性能的量化因子计算和分析系统，专为金融数据分析、技术指标计算和因子构建而设计。该项目采用模块化架构，支持多种数据源集成，提供了因子生成、分析和可视化的完整解决方案。

### 核心技术栈
- **Python**: 主要编程语言
- **FastAPI**: Web API 框架
- **MongoDB**: 主要数据存储
- **Pandas/NumPy**: 数据处理和计算
- **Redis**: 缓存层
- **APScheduler**: 任务调度

### 项目架构
```
panda_factor/
├── panda_common/         # 公共组件和配置
├── panda_data/          # 数据访问层
├── panda_data_hub/      # 数据自动更新服务
├── panda_factor/        # 因子计算核心
├── panda_factor_server/ # REST API 服务器
├── panda_llm/           # 大模型集成服务
├── panda_web/           # 前端界面
└── server/              # 服务器配置
```

## 构建和运行

### 环境准备
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置数据库：
   - 修改 `panda_common/panda_common/config.yaml` 中的 MongoDB 连接信息
   - 确保 MongoDB 服务已启动

3. 开发环境设置（VSCode）：
```bash
# 在各个子模块目录下执行
cd panda_common && pip install -e .
cd ../panda_data && pip install -e .
cd ../panda_data_hub && pip install -e .
cd ../panda_factor && pip install -e .
cd ../panda_factor_server && pip install -e .
cd ../panda_llm && pip install -e .
```

### 启动服务

1. **启动主服务器**（包含所有API）：
```bash
python -m panda_factor_server
```
服务将在 `http://localhost:8111` 启动

2. **启动数据自动更新服务**：
```bash
python -m panda_data_hub
```

3. **单独启动LLM服务**：
```bash
python -m panda_llm
```

### 测试
```bash
pytest
```

## 开发约定

### 因子开发规范

#### Python模式
```python
from panda_factor.generate.factor_base import Factor

class CustomFactor(Factor):
    def calculate(self, factors):
        # 获取基础数据
        close = factors['close']
        volume = factors['volume']
        
        # 计算因子逻辑
        result = self.RANK((close / self.DELAY(close, 20)) - 1)
        
        return result  # 必须返回MultiIndex Series
```

#### 公式模式
```python
"RANK((CLOSE / DELAY(CLOSE, 20)) - 1) * STDDEV(VOLUME, 20)"
```

### 代码组织约定
- 每个模块都有独立的 `setup.py`
- 使用 `panda_common` 作为共享库
- 配置文件统一在 `panda_common/config.yaml`
- 日志使用 `loguru` 统一管理

### 数据库约定
- 主数据库：`panda`
- 集合命名：使用下划线分隔的小写命名
- 索引：在 `symbol` 和 `date` 字段上建立复合索引

## API 接口

### 因子相关接口
- `GET /api/v1/factors` - 获取因子列表
- `POST /api/v1/factor/calculate` - 计算因子
- `GET /api/v1/factor/{factor_name}` - 获取特定因子数据

### LLM接口
- `POST /llm/chat` - 聊天接口

### 前端界面
访问 `http://localhost:8111/factor` 查看Web界面

## 数据源配置

支持的数据源：
- **Tushare**: 需要配置 `TS_TOKEN`
- **RiceQuant**: 需要配置 `MUSER` 和 `MPASSWORD`
- **迅投(XT)**: 需要配置 `XT_TOKEN`
- **TQSDK**: 需要配置 `USER` 和 `PASSWORD`

在 `config.yaml` 中设置 `DATASOURCE` 来选择数据源。

## 常用命令

### 因子计算示例
```python
import panda_data

# 初始化
panda_data.init()

# 获取因子数据
factor = panda_data.get_factor_by_name(
    factor_name="VH03cc651", 
    start_date='20240320',
    end_date='20250325'
)
```

### 数据更新
数据更新会在每日20:00自动执行，确保：
- 股票基础数据更新
- 因子数据计算和更新
- 数据库索引优化

## 故障排除

### 常见问题
1. **MongoDB连接失败**
   - 检查 `config.yaml` 中的连接配置
   - 确认MongoDB服务已启动

2. **数据源认证失败**
   - 验证Token或账号密码是否正确
   - 检查网络连接

3. **因子计算错误**
   - 检查因子代码是否符合规范
   - 查看日志文件 `logs/data_cleaner.log`

### 日志查看
```bash
tail -f logs/data_cleaner.log
```

## 开发工具

### PyCharm配置
将以下文件夹标记为Sources Root：
- `panda_common`
- `panda_data`
- `panda_data_hub`
- `panda_factor`
- `panda_llm`
- `panda_factor_server`

### VSCode配置
确保Python解释器正确配置，并在各模块目录下执行 `pip install -e .`

## 部署说明

### Docker部署
```bash
# 构建镜像
docker build -f Dockerfile -t panda-factor .

# 运行容器
docker run -p 8111:8111 panda-factor
```

### 生产环境
- 使用MongoDB副本集模式
- 配置Redis缓存
- 设置日志轮转
- 配置监控和告警

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码
4. 发起Pull Request

## 许可证
本项目采用 GPLV3 许可证。