# PandaFactor MongoDB 使用文档

## 概述

PandaFactor是一个高性能的量化因子计算和分析系统，使用MongoDB作为主要数据存储。本文档详细介绍了MongoDB在项目中的配置、架构设计、使用方式和性能优化策略。

## 1. 数据库配置

### 1.1 基础配置
MongoDB配置信息位于 `panda_common/panda_common/config.yaml`：

```yaml
# 数据库配置
MONGO_USER: "panda"
MONGO_PASSWORD: "panda"
MONGO_URI: "127.0.0.1:27017"
MONGO_AUTH_DB: "admin"
MONGO_DB: "panda"

# 数据库类型配置
MONGO_TYPE: "replica_set"  # single | replica_set | sharded
MONGO_REPLICA_SET: "rs0"
```

### 1.2 部署模式
- **单节点模式** (`single`): 适用于开发环境或小型应用
- **副本集模式** (`replica_set`): 提供高可用性和数据冗余
- **分片模式** (`sharded`): 用于大规模数据存储（待实现）

## 2. 数据库架构

### 2.1 主要集合结构

| 集合名称 | 用途 | 主要字段 |
|---------|------|---------|
| `stock_market` | 股票市场基础数据 | symbol, date, open, close, high, low, volume |
| `factor_base` | 基础因子数据 | symbol, date, 各种因子字段 |
| `user_factors` | 用户自定义因子定义 | user_id, factor_name, code, params |
| `user_factor_submissions` | 用户因子提交记录 | userId, factorId, factorDetails |
| `factor_{factor_name}_{user_id}` | 用户因子计算结果 | date, symbol, value |

### 2.2 数据模型特点
- **时间序列数据**: 按日期和股票代码组织数据
- **MultiIndex结构**: 使用 `(date, symbol)` 作为复合索引
- **用户隔离**: 通过用户ID隔离不同用户的因子数据

## 3. 数据库操作接口

### 3.1 DatabaseHandler类
`DatabaseHandler` 类提供统一的数据库操作接口，采用单例模式管理连接：

```python
from panda_common.handlers.database_handler import DatabaseHandler

# 初始化
db_handler = DatabaseHandler(config)

# 基础操作
db_handler.mongo_insert(db_name, collection_name, document)
db_handler.mongo_find(db_name, collection_name, query)
db_handler.mongo_update(db_name, collection_name, query, update)
db_handler.mongo_delete(db_name, collection_name, query)
```

### 3.2 核心方法说明

#### 查询操作
```python
# 查找多个文档
records = db_handler.mongo_find(
    db_name="panda",
    collection_name="stock_market",
    query={"date": {"$gte": start_date, "$lte": end_date}},
    projection={"symbol": 1, "close": 1, "_id": 0},
    sort=[("date", 1), ("symbol", 1)]
)

# 查找单个文档
record = db_handler.mongo_find_one(
    db_name="panda",
    collection_name="user_factors",
    query={"factor_name": factor_name}
)
```

#### 插入操作
```python
# 单个文档插入
insert_id = db_handler.mongo_insert(
    db_name="panda",
    collection_name="user_factors",
    document=factor_data
)

# 批量文档插入
insert_ids = db_handler.mongo_insert_many(
    db_name="panda",
    collection_name="factor_base",
    documents=factor_data_list
)
```

#### 聚合操作
```python
# 聚合查询
pipeline = [
    {"$match": {"date": {"$gte": start_date}}},
    {"$group": {"_id": "$symbol", "avg_close": {"$avg": "$close"}}}
]
results = db_handler.mongo_aggregate(
    db_name="panda",
    collection_name="stock_market",
    aggregation_pipeline=pipeline
)
```

## 4. 数据读取模式

### 4.1 FactorReader数据读取
`FactorReader` 类实现了高效的数据读取模式：

```python
class FactorReader:
    def __init__(self, config):
        self.db_handler = DatabaseHandler(config)
    
    def get_factor(self, symbols, factors, start_date, end_date):
        # 构建查询条件
        query = {"date": {"$gte": start_date, "$lte": end_date}}
        
        # 投影查询，只获取需要的字段
        projection = {field: 1 for field in ['date', 'symbol'] + factors}
        projection['_id'] = 0
        
        # 批量查询优化
        collection = self.db_handler.get_mongo_collection("panda", "factor_base")
        cursor = collection.find(query, projection).batch_size(100000)
        records = list(cursor)
        
        return pd.DataFrame(records)
```

### 4.2 查询优化策略
- **批量查询**: 使用 `batch_size(100000)` 优化大数据量查询
- **索引命中**: 在 `(symbol, date)` 复合索引上执行查询
- **投影查询**: 只查询需要的字段，减少数据传输
- **条件过滤**: 支持日期范围、股票代码、指数成分等条件过滤

## 5. 索引设计

### 5.1 核心索引
项目创建了以下关键索引：

```python
# stock_market集合索引
stock_market.create_index([("symbol", ASCENDING), ("date", ASCENDING)])  # 复合索引
stock_market.create_index([("date", ASCENDING)])  # 日期索引
stock_market.create_index([("symbol", ASCENDING)])  # 股票代码索引
```

### 5.2 索引使用原则
- **复合索引优先**: `(symbol, date)` 复合索引支持大部分查询模式
- **查询模式匹配**: 索引设计基于实际查询模式
- **索引监控**: 定期检查索引使用情况

## 6. 性能优化

### 6.1 连接管理
```python
# 单例模式连接池
class DatabaseHandler:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseHandler, cls).__new__(cls)
        return cls._instance
```

### 6.2 读写分离
```python
# 连接配置
self.mongo_client = pymongo.MongoClient(
    MONGO_URI,
    readPreference='secondaryPreferred',  # 优先从从节点读取
    w='majority',  # 写入多数节点确认
    retryWrites=True,  # 自动重试写操作
    socketTimeoutMS=30000,
    connectTimeoutMS=20000,
    serverSelectionTimeoutMS=30000
)
```

### 6.3 查询优化
- **批量处理**: 使用批量插入和查询
- **字段投影**: 只查询必要字段
- **分页查询**: 大数据量查询使用分页
- **索引提示**: 使用hint强制使用特定索引

## 7. 数据安全

### 7.1 连接安全
- **密码编码**: 特殊字符自动URL编码
- **认证数据库**: 明确指定认证数据库
- **连接超时**: 设置合理的连接和操作超时

### 7.2 数据保护
- **写入确认**: 使用 `w='majority'` 确保数据安全
- **重试机制**: 自动重试失败的写操作
- **错误处理**: 完善的异常处理机制

## 8. 监控和维护

### 8.1 性能监控
```python
# 查询性能监控
start_time = time.time()
records = db_handler.mongo_find(db_name, collection_name, query)
logger.info(f"Query took {time.time() - start_time:.3f} seconds")
```

### 8.2 索引维护
```python
# 查看索引使用情况
for index in collection.list_indexes():
    print(f"Index: {index['name']}, Key: {index['key']}")
```

### 8.3 数据清理
- **定期清理**: 清理过期的临时数据
- **索引重建**: 定期重建索引优化性能
- **数据归档**: 历史数据归档策略

## 9. 最佳实践

### 9.1 开发建议
1. **使用连接池**: 避免频繁创建和销毁连接
2. **合理设计索引**: 基于查询模式设计索引
3. **批量操作**: 优先使用批量插入和更新
4. **字段投影**: 只查询需要的字段
5. **异常处理**: 完善的错误处理和日志记录

### 9.2 部署建议
1. **副本集部署**: 生产环境使用副本集
2. **读写分离**: 配置读写分离提升性能
3. **监控告警**: 设置数据库监控和告警
4. **备份策略**: 定期备份数据
5. **容量规划**: 提前规划存储容量

## 10. 故障排除

### 10.1 常见问题
- **连接失败**: 检查网络和认证配置
- **查询慢**: 检查索引和查询条件
- **内存不足**: 调整批量大小和查询限制

### 10.2 调试工具
```python
# 查看执行计划
cursor = collection.find(query).explain()
print(cursor)

# 查看服务器状态
db.command("serverStatus")
```

## 总结

MongoDB在PandaFactor项目中扮演着核心数据存储的角色，通过合理的设计和优化，支撑了量化因子计算和分析的高性能需求。遵循本文档的最佳实践，可以确保数据库系统的稳定性和高性能。