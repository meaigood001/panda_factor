# PandaFactor 大模型应用文档

## 概述

PandaFactor项目中的大模型(LLM)应用是一个独立的AI助手模块，专门设计用于量化因子开发辅助。该模块基于DeepSeek API，提供专业的因子开发咨询、代码编写和优化建议功能。

## 1. 系统架构

### 1.1 模块结构
```
panda_llm/
├── services/
│   ├── llm_service.py      # LLM核心服务类
│   ├── chat_service.py     # 聊天会话管理服务
│   └── mongodb.py          # MongoDB会话存储服务
├── routes/
│   └── chat_router.py      # FastAPI路由定义
├── models/
│   └── chat.py             # 数据模型定义
└── server.py               # FastAPI应用主文件
```

### 1.2 技术栈
- **后端框架**: FastAPI
- **LLM提供商**: DeepSeek API (OpenAI兼容)
- **数据库**: MongoDB (存储会话)
- **响应模式**: 支持流式和非流式响应
- **语言**: 强制中文回复

## 2. 核心功能

### 2.1 AI助手角色定位
- **专业领域**: 量化因子开发、编码和优化
- **语言支持**: 强制中文回复
- **知识范围**: 内置丰富的因子函数库和技术指标知识
- **辅助能力**: 因子代码编写、调试、优化建议

### 2.2 支持的因子开发模式

#### 公式模式
使用数学表达式与内置函数结合，适用于简单因子计算：
```python
# 简单动量因子
"RANK((CLOSE / DELAY(CLOSE, 20)) - 1)"

# 量价相关因子
"CORRELATION(CLOSE, VOLUME, 20)"

# 复合因子
"RANK((CLOSE / DELAY(CLOSE, 20)) - 1) * STDDEV((CLOSE / DELAY(CLOSE, 1)) - 1, 20)"
```

#### Python模式
自定义因子类继承Factor基类，适用于复杂因子逻辑：
```python
class MomentumFactor(Factor):
    def calculate(self, factors):
        close = factors['close']
        # 计算20日收益率
        returns = (close / self.DELAY(close, 20)) - 1
        return self.RANK(returns)

class ComplexFactor(Factor):
    def calculate(self, factors):
        close = factors['close']
        volume = factors['volume']
        
        # 计算收益率
        returns = (close / self.DELAY(close, 20)) - 1
        # 计算波动率
        volatility = self.STDDEV((close / self.DELAY(close, 1)) - 1, 20)
        # 计算成交量比率
        volume_ratio = volume / self.DELAY(volume, 1)
        # 组合信号
        result = self.RANK(returns) * volatility * (volume_ratio / self.SUM(volume_ratio, 10))
        return result
```

## 3. 内置因子函数库

### 3.1 基础数据因子
| 函数名 | 说明 | 参数 |
|--------|------|------|
| CLOSE | 收盘价 | - |
| OPEN | 开盘价 | - |
| HIGH | 最高价 | - |
| LOW | 最低价 | - |
| VOLUME | 成交量 | - |
| AMOUNT | 成交额 | - |
| TURNOVER | 换手率 | - |
| MARKET_CAP | 市值 | - |

### 3.2 基础计算函数
| 函数名 | 说明 | 参数 |
|--------|------|------|
| RANK(series) | 截面排名，归一化到[-0.5, 0.5] | series: 输入序列 |
| RETURNS(close, period=1) | 计算收益率 | close: 价格序列, period: 周期 |
| STDDEV(series, window=20) | 滚动标准差 | series: 输入序列, window: 窗口 |
| CORRELATION(series1, series2, window=20) | 滚动相关系数 | series1/2: 输入序列, window: 窗口 |
| IF(condition, true_value, false_value) | 条件选择 | condition: 条件, true/false: 值 |
| MIN(series1, series2) | 取最小值 | series1/2: 输入序列 |
| MAX(series1, series2) | 取最大值 | series1/2: 输入序列 |
| ABS(series) | 绝对值 | series: 输入序列 |
| LOG(series) | 自然对数 | series: 输入序列 |
| POWER(series, power) | 幂运算 | series: 输入序列, power: 幂 |

### 3.3 时间序列函数
| 函数名 | 说明 | 参数 |
|--------|------|------|
| DELAY(series, period=1) | 时间序列延迟 | series: 输入序列, period: 延迟周期 |
| SUM(series, window=20) | 滚动求和 | series: 输入序列, window: 窗口 |
| TS_MEAN(series, window=20) | 滚动平均 | series: 输入序列, window: 窗口 |
| TS_MIN(series, window=20) | 滚动最小值 | series: 输入序列, window: 窗口 |
| TS_MAX(series, window=20) | 滚动最大值 | series: 输入序列, window: 窗口 |
| TS_RANK(series, window=20) | 时间序列排名 | series: 输入序列, window: 窗口 |
| MA(series, window) | 简单移动平均 | series: 输入序列, window: 窗口 |
| EMA(series, window) | 指数移动平均 | series: 输入序列, window: 窗口 |
| SMA(series, window, M=1) | 平滑移动平均 | series: 输入序列, window: 窗口, M: 平滑参数 |
| WMA(series, window) | 加权移动平均 | series: 输入序列, window: 窗口 |

### 3.4 技术指标函数
| 函数名 | 说明 | 参数 |
|--------|------|------|
| MACD(close, SHORT=12, LONG=26, M=9) | MACD指标 | close: 价格, SHORT/LONG: 快慢线周期, M: 信号线周期 |
| KDJ(close, high, low, N=9, M1=3, M2=3) | KDJ指标 | close/high/low: 价格数据, N: 周期, M1/M2: 平滑参数 |
| RSI(close, N=24) | 相对强弱指数 | close: 价格, N: 周期 |
| BOLL(close, N=20, P=2) | 布林带 | close: 价格, N: 周期, P: 标准差倍数 |
| CCI(close, high, low, N=14) | 商品通道指数 | close/high/low: 价格数据, N: 周期 |
| ATR(close, high, low, N=20) | 平均真实波幅 | close/high/low: 价格数据, N: 周期 |

### 3.5 核心工具函数
| 函数名 | 说明 | 参数 |
|--------|------|------|
| RD(S, D=3) | 四舍五入到D位小数 | S: 输入值, D: 小数位数 |
| REF(S, N=1) | 序列整体下移N位 | S: 输入序列, N: 移动位数 |
| DIFF(S, N=1) | 计算差值 | S: 输入序列, N: 差分周期 |
| CROSS(S1, S2) | 检测上穿 | S1/S2: 输入序列 |
| FILTER(S, N) | 信号过滤，N周期内保留第一个信号 | S: 信号序列, N: 过滤周期 |

## 4. API接口

### 4.1 聊天接口
```http
POST /llm/chat
Content-Type: application/json

{
    "user_id": "用户ID",
    "message": "用户消息",
    "session_id": "会话ID（可选）"
}
```

**响应格式**（流式）：
```
data: {"content": "回复内容片段"}
data: {"content": "更多内容"}
data: [DONE]
```

### 4.2 获取会话列表
```http
GET /llm/chat/sessions?user_id=用户ID&limit=10
```

**响应格式**：
```json
{
    "sessions": [
        {
            "id": "会话ID",
            "user_id": "用户ID",
            "messages": [
                {
                    "role": "user/assistant",
                    "content": "消息内容",
                    "timestamp": "时间戳"
                }
            ],
            "created_at": "创建时间",
            "updated_at": "更新时间"
        }
    ]
}
```

## 5. 配置说明

### 5.1 环境配置
在 `panda_common/config.yaml` 中配置LLM相关参数：

```yaml
# OpenAI配置
LLM_API_KEY: "你的API密钥"
LLM_MODEL: "deepseek-chat"
LLM_BASE_URL: "https://api.deepseek.com/v1"
```

### 5.2 数据库配置
```yaml
# MongoDB配置
MONGO_USER: "panda"
MONGO_PASSWORD: "panda"
MONGO_URI: "127.0.0.1:27017"
MONGO_AUTH_DB: "admin"
MONGO_DB: "panda"
```

## 6. 部署和使用

### 6.1 安装依赖
```bash
cd panda_llm
pip install -r requirements.txt
```

### 6.2 启动服务
```bash
python -m panda_llm
```

服务默认运行在独立端口，提供RESTful API接口。

### 6.3 使用示例

#### Python调用示例
```python
import requests
import json

# 发送聊天请求
url = "http://localhost:8000/llm/chat"
data = {
    "user_id": "test_user",
    "message": "如何编写一个动量因子？",
    "session_id": None
}

response = requests.post(url, json=data, stream=True)
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if 'content' in data:
                print(data['content'], end='')
```

#### 前端JavaScript调用示例
```javascript
const eventSource = new EventSource('/llm/chat', {
    method: 'POST',
    body: JSON.stringify({
        user_id: 'user123',
        message: '帮我写一个RSI因子',
        session_id: 'session456'
    })
});

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.content) {
        console.log(data.content);
    }
};
```

## 7. 集成状态和发展方向

### 7.1 当前集成状态
**已实现功能：**
- ✅ 完整的AI聊天服务
- ✅ 专业的因子开发知识库
- ✅ 会话管理和历史记录
- ✅ 流式响应支持
- ✅ MongoDB数据持久化

**待集成功能：**
- ❌ LLM服务与主服务器的路由集成
- ❌ 前端界面的AI助手功能
- ❌ 因子计算过程中的实时AI辅助
- ❌ 因子分析报告的AI生成

### 7.2 潜在集成点
1. **因子生成界面集成**：在因子创建页面添加AI助手聊天窗口
2. **因子分析增强**：使用AI分析因子表现和提供改进建议
3. **智能调试**：在因子计算错误时提供AI诊断和修复建议
4. **知识库扩展**：基于用户反馈持续优化AI的因子开发知识

## 8. 最佳实践

### 8.1 提问技巧
- **明确需求**：具体描述因子逻辑和预期效果
- **提供上下文**：说明使用的数据源和时间范围
- **分步骤提问**：复杂问题可分解为多个简单问题
- **示例参考**：提供相关因子示例以便AI理解需求

### 8.2 因子开发建议
- **避免未来数据**：确保因子计算不使用未来信息
- **处理缺失值**：在因子计算中适当处理NaN值
- **标准化处理**：使用RANK等函数进行截面标准化
- **参数优化**：通过回测验证因子参数的有效性

### 8.3 性能优化
- **批量计算**：避免在循环中进行单个股票计算
- **向量化操作**：使用Pandas/Numpy的向量化函数
- **缓存机制**：对重复计算的中间结果进行缓存
- **并行处理**：利用多进程处理大规模因子计算

## 9. 故障排除

### 9.1 常见问题
**API调用失败**：
- 检查API密钥是否正确配置
- 确认网络连接和API服务可用性
- 验证请求格式和参数

**数据库连接问题**：
- 检查MongoDB服务状态
- 验证连接配置和认证信息
- 确认数据库权限设置

**响应质量问题**：
- 调整系统提示词以获得更精准的回答
- 限制回复长度避免内容截断
- 优化temperature参数控制回答创造性

### 9.2 日志查看
```bash
# 查看LLM服务日志
tail -f logs/llm_service.log

# 查看聊天会话日志
tail -f logs/chat_service.log
```

## 10. 总结

PandaFactor的大模型应用为量化因子开发提供了强大的AI辅助能力。通过专业的因子开发知识库和智能对话系统，用户可以快速获得因子编写、调试和优化的专业建议。随着功能的进一步集成，该模块将成为量化投资研究的重要工具，显著提升因子开发的效率和质量。

---

*本文档基于PandaFactor项目版本v1.0编写，如有更新请参考最新代码实现。*