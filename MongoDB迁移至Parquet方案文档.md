# PandaFactor MongoDB 迁移至 Parquet 方案文档

## 文档版本信息
- **版本**: 1.0
- **创建日期**: 2025-10-21
- **目标**: 将 PandaFactor 项目从 MongoDB 数据库存储迁移到 Parquet 文件存储

---

## 一、项目概述

### 1.1 当前架构
PandaFactor 目前使用 MongoDB 作为主要数据存储方案，主要存储：
- 股票市场行情数据 (stock_market)
- 期货市场行情数据 (future_market)
- 基础因子数据 (factor_base)
- 用户自定义因子定义 (user_factors)
- 用户因子提交记录 (user_factor_submissions)
- 因子计算结果 (factor_{factor_name}_{user_id})
- 因子分析结果 (factor_analysis_results)
- 后台任务状态 (tasks)

### 1.2 迁移目标
使用 Parquet 文件格式替代 MongoDB，实现：
- **更好的数据压缩**: Parquet 列式存储压缩比更高
- **更快的列式查询**: 适合因子计算场景的列式数据访问
- **更低的运维成本**: 无需维护数据库服务
- **更好的可移植性**: 文件系统存储更易于备份和迁移
- **与数据科学生态集成**: 与 pandas、dask、polars 等工具无缝对接

---

## 二、影响范围分析

### 2.1 核心模块影响

#### 2.1.1 panda_common 模块
**文件**: `panda_common/handlers/database_handler.py`

**当前功能**:
- MongoDB 连接管理 (单例模式)
- 基础 CRUD 操作封装
- 聚合查询
- 索引管理

**需要修改**:
- 重构为 `ParquetHandler` 类
- 实现文件系统管理
- 实现分区策略
- 实现元数据管理
- 实现并发读写控制

**修改内容**:
```python
# 当前接口
class DatabaseHandler:
    - mongo_insert(db_name, collection_name, document)
    - mongo_find(db_name, collection_name, query, projection, hint, sort)
    - mongo_update(db_name, collection_name, query, update)
    - mongo_delete(db_name, collection_name, query)
    - mongo_insert_many(db_name, collection_name, documents)
    - mongo_aggregate(db_name, collection_name, pipeline)
    - get_distinct_values(db_name, collection_name, field)

# 需要实现的新接口
class ParquetHandler:
    - insert(dataset_name, data)                    # 对应 mongo_insert
    - query(dataset_name, filters, columns, sort)    # 对应 mongo_find
    - update(dataset_name, filters, update_data)     # 对应 mongo_update
    - delete(dataset_name, filters)                  # 对应 mongo_delete
    - insert_batch(dataset_name, data_list)         # 对应 mongo_insert_many
    - aggregate(dataset_name, operations)            # 对应 mongo_aggregate
    - get_distinct_values(dataset_name, field)       # 对应 get_distinct_values
```

#### 2.1.2 panda_data 模块

**文件**: `panda_data/market_data/market_data_reader.py`

**当前功能**:
- 市场数据查询 (支持日期范围、股票代码、指标筛选)
- 分块并行查询 (ThreadPoolExecutor)
- 批量读取优化 (batch_size)
- 获取所有股票代码

**需要修改**:
- 使用 Parquet 分区读取替代 MongoDB 查询
- 利用 Parquet 的过滤下推 (filter pushdown)
- 保持分块并行读取逻辑
- 实现基于 Parquet 元数据的符号列表获取

**修改策略**:
```python
# 当前实现
def get_market_data(self, symbols, start_date, end_date, ...):
    collection.find(query, projection).batch_size(target_batch_size)

# Parquet 实现方案
def get_market_data(self, symbols, start_date, end_date, ...):
    # 使用 PyArrow/Pandas 读取分区 Parquet
    # 过滤条件: date >= start_date AND date <= end_date AND symbol IN (symbols)
    filters = [
        ('date', '>=', start_date),
        ('date', '<=', end_date),
        ('symbol', 'in', symbols)
    ]
    df = pd.read_parquet(
        path='data/stock_market',
        columns=fields,
        filters=filters
    )
```

**文件**: `panda_data/factor/factor_reader.py`

**当前功能**:
- 基础因子数据查询
- 自定义因子数据查询
- 因子代码动态执行
- 因子结果缓存

**需要修改**:
- 使用 Parquet 读取基础因子
- 使用 Parquet 读取缓存的自定义因子
- 用户因子定义需要迁移到 JSON/YAML 文件或轻量级数据库 (SQLite)

#### 2.1.3 panda_data_hub 模块

**文件**: `panda_data_hub/services/*_stock_market_clean_service.py`

**当前功能**:
- 从 Tushare/RiceQuant/XTQuant/TQSDK 获取数据
- 使用 `UpdateOne` 批量更新 MongoDB
- 创建集合和索引

**需要修改**:
- 改为写入 Parquet 文件
- 使用追加模式或分区覆盖模式
- 移除索引创建逻辑 (Parquet 不需要索引)

**修改策略**:
```python
# 当前实现
bulk_operations = [
    UpdateOne(
        {"symbol": row["symbol"], "date": row["date"]},
        {"$set": row_dict},
        upsert=True
    )
]
collection.bulk_write(bulk_operations)

# Parquet 实现方案
# 按日期分区存储
partition_path = f'data/stock_market/date={date_str}'
df.to_parquet(
    partition_path,
    engine='pyarrow',
    compression='snappy',
    partition_cols=['date'],
    existing_data_behavior='delete_matching'  # 覆盖同分区数据
)
```

**文件**: `panda_data_hub/utils/mongo_utils.py`

**当前功能**:
- 确保集合存在
- 创建复合索引 (symbol, date)

**需要修改**:
- 改为确保 Parquet 目录结构存在
- 创建必要的元数据文件
- 移除索引创建逻辑

#### 2.1.4 panda_factor 模块

**文件**: `panda_factor/generate/factor_data_handler.py`

**当前功能**:
- 并行获取多个基础因子数据
- 数据清洗和去重

**需要修改**:
- 使用 Parquet 并行读取
- 保持现有的 ThreadPoolExecutor 逻辑

**文件**: `panda_factor/analysis/factor_analysis.py`

**当前功能**:
- 因子分析结果存储 (图表、指标)

**需要修改**:
- 分析结果可以存储为 Parquet 或 JSON
- 图表数据建议使用 JSON 存储 (更适合嵌套结构)

#### 2.1.5 panda_factor_server 模块

**文件**: `panda_factor_server/services/user_factor_service.py`

**当前功能**:
- 用户因子列表查询 (分页、排序)
- 因子详情查询
- 因子创建、更新、删除

**需要修改**:
- 用户因子元数据建议迁移到 SQLite
- 因子计算结果存储为 Parquet
- 支持基于 Parquet 的分页查询

#### 2.1.6 panda_llm 模块

**文件**: `panda_llm/services/mongodb.py`

**当前功能**:
- LLM 对话历史存储

**需要修改**:
- 对话历史建议存储为 JSON 文件或 SQLite
- 不适合使用 Parquet (非时序、非列式分析场景)

### 2.2 配置文件修改

**文件**: `panda_common/panda_common/config.yaml`

**需要移除的配置**:
```yaml
MONGO_USER: "panda"
MONGO_PASSWORD: "panda"
MONGO_URI: "127.0.0.1:27017"
MONGO_AUTH_DB: "admin"
MONGO_DB: "panda"
MONGO_TYPE: "replica_set"
MONGO_REPLICA_SET: "rs0"
```

**需要添加的配置**:
```yaml
# Parquet 存储配置
DATA_ROOT_PATH: "C:/panda_data"  # Windows 示例
# DATA_ROOT_PATH: "/var/panda_data"  # Linux 示例

# 数据集路径
STOCK_MARKET_PATH: "${DATA_ROOT_PATH}/stock_market"
FUTURE_MARKET_PATH: "${DATA_ROOT_PATH}/future_market"
FACTOR_BASE_PATH: "${DATA_ROOT_PATH}/factor_base"
USER_FACTORS_PATH: "${DATA_ROOT_PATH}/user_factors"

# Parquet 引擎配置
PARQUET_ENGINE: "pyarrow"  # pyarrow 或 fastparquet
PARQUET_COMPRESSION: "snappy"  # snappy, gzip, zstd
PARQUET_ROW_GROUP_SIZE: 1000000  # 每个 row group 的行数

# 分区策略
PARTITION_COLS: ["date"]  # 按日期分区

# 元数据存储 (用于用户因子、任务等)
METADATA_DB_TYPE: "sqlite"  # sqlite 或 json
METADATA_DB_PATH: "${DATA_ROOT_PATH}/metadata.db"
```

---

## 三、数据存储方案设计

### 3.1 数据分类与存储策略

#### 3.1.1 时序市场数据 (推荐 Parquet)
**数据集**: stock_market, future_market, factor_base

**存储方案**:
```
data/
├── stock_market/
│   ├── date=20240101/
│   │   ├── part-0000.parquet
│   │   └── part-0001.parquet
│   ├── date=20240102/
│   │   └── part-0000.parquet
│   └── _metadata  # Parquet 元数据文件
│
├── future_market/
│   └── date=20240101/
│       └── part-0000.parquet
│
└── factor_base/
    └── date=20240101/
        └── part-0000.parquet
```

**分区策略**:
- 一级分区: `date` (日期分区，格式: YYYYMMDD)
- 可选二级分区: `symbol` (股票代码分区，适用于大规模数据)

**文件命名规范**:
- `part-{序号}.parquet`
- 每个文件大小建议: 128MB - 512MB

**索引策略**:
- Parquet 自带列统计信息 (min/max/null_count)
- 可使用 PyArrow Dataset 的索引功能
- 可选: 创建外部索引文件 (JSON 格式) 存储 symbol 列表

#### 3.1.2 用户因子计算结果 (推荐 Parquet)
**数据集**: factor_{factor_name}_{user_id}

**存储方案**:
```
data/
└── user_factors/
    ├── user_123/
    │   ├── momentum_factor/
    │   │   ├── date=20240101/
    │   │   │   └── part-0000.parquet
    │   │   └── _metadata
    │   └── volatility_factor/
    │       └── date=20240101/
    │           └── part-0000.parquet
    └── user_456/
        └── custom_factor/
            └── date=20240101/
                └── part-0000.parquet
```

#### 3.1.3 元数据和配置 (推荐 SQLite 或 JSON)
**数据集**: user_factors, user_factor_submissions, tasks

**SQLite 方案** (推荐):
```
data/
└── metadata.db  # SQLite 数据库
    ├── user_factors (表)
    ├── user_factor_submissions (表)
    └── tasks (表)
```

**表结构设计**:
```sql
-- 用户因子定义表
CREATE TABLE user_factors (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    code_type TEXT,  -- formula 或 python
    code TEXT,
    params TEXT,  -- JSON 格式
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    UNIQUE(user_id, factor_name)
);
CREATE INDEX idx_user_factors_user_id ON user_factors(user_id);
CREATE INDEX idx_user_factors_created_at ON user_factors(created_at);

-- 用户因子提交记录表
CREATE TABLE user_factor_submissions (
    id TEXT PRIMARY KEY,
    user_id INTEGER,
    factor_id TEXT,
    factor_details TEXT,  -- JSON 格式
    submitted_at TIMESTAMP
);
CREATE INDEX idx_submissions_user_id ON user_factor_submissions(user_id);

-- 任务状态表
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    factor_id TEXT,
    status TEXT,
    progress INTEGER,
    result TEXT,  -- JSON 格式
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);
```

**JSON 文件方案** (备选):
```
data/
└── metadata/
    ├── user_factors/
    │   ├── user_123_factor_001.json
    │   └── user_123_factor_002.json
    ├── submissions/
    │   └── submission_001.json
    └── tasks/
        └── task_001.json
```

#### 3.1.4 分析结果 (推荐混合存储)
**数据集**: factor_analysis_results

**存储方案**:
```
data/
└── analysis_results/
    ├── user_123/
    │   └── momentum_factor/
    │       ├── metadata.json  # 分析参数、指标
    │       ├── returns.parquet  # 收益率时序数据
    │       ├── ic_series.parquet  # IC 时序数据
    │       └── charts/
    │           ├── pct_chart.json
    │           ├── ic_chart.json
    │           └── ic_decay_chart.json
    └── user_456/
        └── custom_factor/
            └── ...
```

### 3.2 Parquet 文件格式配置

#### 3.2.1 推荐配置
```python
import pyarrow as pa
import pyarrow.parquet as pq

# Schema 定义示例 (stock_market)
schema = pa.schema([
    ('symbol', pa.string()),
    ('date', pa.int32()),  # YYYYMMDD 格式
    ('open', pa.float64()),
    ('close', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('volume', pa.float64()),
    ('amount', pa.float64()),
    ('market_cap', pa.float64()),
    ('turnover', pa.float64()),
    ('index_component', pa.string()),
    ('name', pa.string())
])

# 写入配置
pq.write_to_dataset(
    table,
    root_path='data/stock_market',
    partition_cols=['date'],
    compression='snappy',  # 压缩算法
    row_group_size=1000000,  # 每个 row group 100万行
    use_dictionary=True,  # 使用字典编码
    write_statistics=True,  # 写入统计信息
    version='2.6'  # Parquet 版本
)
```

#### 3.2.2 压缩算法选择
- **snappy**: 压缩速度快，压缩比中等 (推荐用于实时写入)
- **gzip**: 压缩比高，速度慢 (推荐用于归档数据)
- **zstd**: 平衡压缩比和速度 (推荐用于一般场景)

#### 3.2.3 读取性能优化
```python
# 使用 PyArrow Dataset API
import pyarrow.dataset as ds

dataset = ds.dataset(
    'data/stock_market',
    format='parquet',
    partitioning=ds.partitioning(
        pa.schema([('date', pa.int32())])
    )
)

# 过滤下推 (filter pushdown)
filtered = dataset.to_table(
    columns=['symbol', 'date', 'close'],
    filter=(
        (ds.field('date') >= 20240101) &
        (ds.field('date') <= 20241231) &
        (ds.field('symbol').isin(['000001.SZ', '600000.SH']))
    )
)

df = filtered.to_pandas()
```

---

## 四、详细修改方案

### 4.1 panda_common 模块修改

#### 4.1.1 新建 `ParquetHandler` 类
**文件路径**: `panda_common/handlers/parquet_handler.py`

**类设计**:
```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

class ParquetHandler:
    """Parquet 文件存储处理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config):
        if not hasattr(self, 'initialized'):
            self.config = config
            self.data_root = Path(config['DATA_ROOT_PATH'])
            self.engine = config.get('PARQUET_ENGINE', 'pyarrow')
            self.compression = config.get('PARQUET_COMPRESSION', 'snappy')
            self.row_group_size = config.get('PARQUET_ROW_GROUP_SIZE', 1000000)

            # 确保根目录存在
            self.data_root.mkdir(parents=True, exist_ok=True)

            # 初始化 SQLite (用于元数据)
            self.metadata_db = self._init_metadata_db()

            self.initialized = True

    def _get_dataset_path(self, dataset_name: str) -> Path:
        """获取数据集路径"""
        return self.data_root / dataset_name

    # ========== 基础操作 ==========

    def insert(self, dataset_name: str, data: Dict[str, Any]) -> str:
        """插入单条记录"""
        df = pd.DataFrame([data])
        return self.insert_batch(dataset_name, df)

    def insert_batch(self, dataset_name: str, data: pd.DataFrame) -> List[str]:
        """批量插入记录"""
        dataset_path = self._get_dataset_path(dataset_name)

        # 确定分区列
        partition_cols = self._get_partition_cols(dataset_name)

        # 写入 Parquet
        table = pa.Table.from_pandas(data)
        pq.write_to_dataset(
            table,
            root_path=str(dataset_path),
            partition_cols=partition_cols,
            compression=self.compression,
            row_group_size=self.row_group_size,
            existing_data_behavior='overwrite_or_ignore'
        )

        return [f"inserted_{i}" for i in range(len(data))]

    def query(
        self,
        dataset_name: str,
        filters: Optional[List] = None,
        columns: Optional[List[str]] = None,
        sort: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """查询数据"""
        dataset_path = self._get_dataset_path(dataset_name)

        if not dataset_path.exists():
            return pd.DataFrame()

        # 使用 PyArrow Dataset API
        dataset = ds.dataset(str(dataset_path), format='parquet')

        # 构建过滤器
        if filters:
            filter_expr = self._build_filter_expression(filters)
        else:
            filter_expr = None

        # 读取数据
        table = dataset.to_table(columns=columns, filter=filter_expr)
        df = table.to_pandas()

        # 排序
        if sort:
            sort_cols = [col for col, _ in sort]
            sort_ascending = [order == 1 for _, order in sort]
            df = df.sort_values(by=sort_cols, ascending=sort_ascending)

        return df

    def update(self, dataset_name: str, filters: Dict, update_data: Dict) -> int:
        """更新数据 (读取-修改-写入)"""
        # Parquet 是不可变的，需要读取、修改、重写
        df = self.query(dataset_name, filters=self._dict_to_filters(filters))

        if df.empty:
            return 0

        # 更新数据
        for key, value in update_data.items():
            df[key] = value

        # 重新写入
        self.insert_batch(dataset_name, df)

        return len(df)

    def delete(self, dataset_name: str, filters: Dict) -> int:
        """删除数据"""
        # 读取所有数据
        all_data = self.query(dataset_name)

        # 过滤要删除的数据
        delete_mask = self._apply_filters_to_df(all_data, filters)
        remaining_data = all_data[~delete_mask]

        # 重写数据集
        dataset_path = self._get_dataset_path(dataset_name)
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)

        if not remaining_data.empty:
            self.insert_batch(dataset_name, remaining_data)

        return delete_mask.sum()

    def get_distinct_values(self, dataset_name: str, field: str) -> List[Any]:
        """获取字段的唯一值"""
        df = self.query(dataset_name, columns=[field])
        return df[field].unique().tolist()

    # ========== 辅助方法 ==========

    def _build_filter_expression(self, filters: List) -> ds.Expression:
        """构建 PyArrow 过滤表达式"""
        # filters 格式: [('date', '>=', 20240101), ('symbol', 'in', ['000001.SZ'])]
        if not filters:
            return None

        expr_list = []
        for filter_item in filters:
            if len(filter_item) == 3:
                field, op, value = filter_item

                if op == '>=':
                    expr_list.append(ds.field(field) >= value)
                elif op == '<=':
                    expr_list.append(ds.field(field) <= value)
                elif op == '==':
                    expr_list.append(ds.field(field) == value)
                elif op == 'in':
                    expr_list.append(ds.field(field).isin(value))
                elif op == 'not_regex':
                    # Parquet 不支持正则，需要在读取后过滤
                    pass

        # 组合表达式
        if not expr_list:
            return None

        combined_expr = expr_list[0]
        for expr in expr_list[1:]:
            combined_expr = combined_expr & expr

        return combined_expr

    def _dict_to_filters(self, filter_dict: Dict) -> List:
        """将字典转换为过滤器列表"""
        filters = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == '$gte':
                        filters.append((key, '>=', val))
                    elif op == '$lte':
                        filters.append((key, '<=', val))
                    elif op == '$eq':
                        filters.append((key, '==', val))
                    elif op == '$in':
                        filters.append((key, 'in', val))
            else:
                filters.append((key, '==', value))

        return filters

    def _apply_filters_to_df(self, df: pd.DataFrame, filters: Dict) -> pd.Series:
        """应用过滤器到 DataFrame"""
        mask = pd.Series([True] * len(df), index=df.index)

        for key, value in filters.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == '$gte':
                        mask &= df[key] >= val
                    elif op == '$lte':
                        mask &= df[key] <= val
                    elif op == '$in':
                        mask &= df[key].isin(val)
            else:
                mask &= df[key] == value

        return mask

    def _get_partition_cols(self, dataset_name: str) -> List[str]:
        """获取数据集的分区列"""
        # 可以根据数据集名称返回不同的分区策略
        if dataset_name in ['stock_market', 'future_market', 'factor_base']:
            return ['date']
        return []

    def _init_metadata_db(self):
        """初始化元数据数据库 (SQLite)"""
        import sqlite3

        db_path = self.config.get('METADATA_DB_PATH',
                                   str(self.data_root / 'metadata.db'))

        conn = sqlite3.connect(db_path, check_same_thread=False)

        # 创建表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_factors (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                code_type TEXT,
                code TEXT,
                params TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                UNIQUE(user_id, factor_name)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_factor_submissions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                factor_id TEXT,
                factor_details TEXT,
                submitted_at TIMESTAMP
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                factor_id TEXT,
                status TEXT,
                progress INTEGER,
                result TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')

        conn.execute('CREATE INDEX IF NOT EXISTS idx_user_factors_user_id ON user_factors(user_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_submissions_user_id ON user_factor_submissions(user_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id)')

        conn.commit()

        return conn

    # ========== 元数据操作 (SQLite) ==========

    def metadata_insert(self, table_name: str, data: Dict) -> str:
        """插入元数据记录"""
        import json

        # 将 dict/list 类型的值转换为 JSON 字符串
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed_data[k] = json.dumps(v)
            else:
                processed_data[k] = v

        columns = ', '.join(processed_data.keys())
        placeholders = ', '.join(['?' for _ in processed_data])

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        cursor = self.metadata_db.execute(query, list(processed_data.values()))
        self.metadata_db.commit()

        return cursor.lastrowid

    def metadata_query(
        self,
        table_name: str,
        filters: Optional[Dict] = None,
        columns: Optional[List[str]] = None,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """查询元数据"""
        import json

        # 构建 SQL
        select_clause = ', '.join(columns) if columns else '*'
        query = f"SELECT {select_clause} FROM {table_name}"

        where_clauses = []
        params = []

        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    for op, val in value.items():
                        if op == '$ne':
                            where_clauses.append(f"{key} != ?")
                            params.append(val)
                else:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if sort:
            order_by = ', '.join([f"{col} {'ASC' if order == 1 else 'DESC'}"
                                  for col, order in sort])
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.metadata_db.execute(query, params)
        rows = cursor.fetchall()

        # 转换为字典列表
        column_names = [desc[0] for desc in cursor.description]
        results = []

        for row in rows:
            row_dict = dict(zip(column_names, row))

            # 尝试解析 JSON 字符串
            for key, value in row_dict.items():
                if isinstance(value, str) and value.startswith(('{', '[')):
                    try:
                        row_dict[key] = json.loads(value)
                    except:
                        pass

            results.append(row_dict)

        return results

    def metadata_update(self, table_name: str, filters: Dict, update_data: Dict) -> int:
        """更新元数据"""
        import json

        # 处理更新数据
        processed_data = {}
        for k, v in update_data.items():
            if isinstance(v, (dict, list)):
                processed_data[k] = json.dumps(v)
            else:
                processed_data[k] = v

        set_clause = ', '.join([f"{k} = ?" for k in processed_data.keys()])
        where_clause = ' AND '.join([f"{k} = ?" for k in filters.keys()])

        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

        params = list(processed_data.values()) + list(filters.values())

        cursor = self.metadata_db.execute(query, params)
        self.metadata_db.commit()

        return cursor.rowcount

    def metadata_delete(self, table_name: str, filters: Dict) -> int:
        """删除元数据"""
        where_clause = ' AND '.join([f"{k} = ?" for k in filters.keys()])
        query = f"DELETE FROM {table_name} WHERE {where_clause}"

        cursor = self.metadata_db.execute(query, list(filters.values()))
        self.metadata_db.commit()

        return cursor.rowcount
```

**主要方法对照表**:

| MongoDB 方法 | ParquetHandler 方法 | 说明 |
|-------------|-------------------|------|
| `mongo_insert()` | `insert()` | 插入单条数据到 Parquet |
| `mongo_insert_many()` | `insert_batch()` | 批量插入数据 |
| `mongo_find()` | `query()` | 查询数据 (支持过滤、投影、排序) |
| `mongo_find_one()` | `query()` + `iloc[0]` | 查询单条数据 |
| `mongo_update()` | `update()` | 更新数据 (读-改-写) |
| `mongo_delete()` | `delete()` | 删除数据 |
| `get_distinct_values()` | `get_distinct_values()` | 获取唯一值 |
| `mongo_aggregate()` | `query()` + pandas 操作 | 聚合查询 |

**元数据方法**:

| MongoDB 方法 | ParquetHandler 方法 | 说明 |
|-------------|-------------------|------|
| `mongo_insert()` (user_factors) | `metadata_insert()` | 插入元数据到 SQLite |
| `mongo_find()` (user_factors) | `metadata_query()` | 查询元数据 |
| `mongo_update()` (user_factors) | `metadata_update()` | 更新元数据 |
| `mongo_delete()` (user_factors) | `metadata_delete()` | 删除元数据 |

#### 4.1.2 修改配置加载
**文件路径**: `panda_common/config.py`

**修改内容**:
```python
# 添加 Parquet 配置加载
config['DATA_ROOT_PATH'] = config.get('DATA_ROOT_PATH', './panda_data')
config['PARQUET_ENGINE'] = config.get('PARQUET_ENGINE', 'pyarrow')
config['PARQUET_COMPRESSION'] = config.get('PARQUET_COMPRESSION', 'snappy')
config['PARQUET_ROW_GROUP_SIZE'] = config.get('PARQUET_ROW_GROUP_SIZE', 1000000)
config['METADATA_DB_TYPE'] = config.get('METADATA_DB_TYPE', 'sqlite')
config['METADATA_DB_PATH'] = config.get('METADATA_DB_PATH', './panda_data/metadata.db')
```

### 4.2 panda_data 模块修改

#### 4.2.1 MarketDataReader 改造
**文件路径**: `panda_data/market_data/market_data_reader.py`

**修改要点**:
1. 替换 `DatabaseHandler` 为 `ParquetHandler`
2. 将 MongoDB 查询改为 Parquet 读取
3. 保持分块并行读取逻辑
4. 利用 Parquet 过滤下推优化性能

**修改示例**:
```python
# 当前代码
class MarketDataReader:
    def __init__(self, config):
        self.db_handler = DatabaseHandler(config)

    def get_market_data(self, symbols, start_date, end_date, ...):
        collection = self.db_handler.get_mongo_collection(
            self.config["MONGO_DB"],
            "stock_market"
        )
        cursor = collection.find(query, projection).batch_size(target_batch_size)
        chunk_df = pd.DataFrame(list(cursor))

# 修改后代码
class MarketDataReader:
    def __init__(self, config):
        self.parquet_handler = ParquetHandler(config)

    def get_market_data(self, symbols, start_date, end_date, ...):
        # 构建过滤器
        filters = [
            ('date', '>=', start_date),
            ('date', '<=', end_date)
        ]

        if symbols:
            filters.append(('symbol', 'in', symbols))

        # 读取 Parquet
        df = self.parquet_handler.query(
            dataset_name='stock_market',
            filters=filters,
            columns=fields if fields else None
        )

        # 应用其他过滤逻辑 (indicator, st)
        if indicator != "000985":
            # ...

        return df
```

**性能优化建议**:
- 使用 `pyarrow.dataset` API 的过滤下推
- 读取时只选择需要的列
- 利用分区剪枝 (partition pruning)

#### 4.2.2 FactorReader 改造
**文件路径**: `panda_data/factor/factor_reader.py`

**修改要点**:
1. 用户因子元数据查询改为 SQLite
2. 因子数据查询改为 Parquet
3. 缓存的因子结果读取改为 Parquet

**修改示例**:
```python
# 当前代码
def get_custom_factor(self, factor_logger, user_id, factor_name, start_date, end_date):
    # 查询因子定义
    records = self.db_handler.mongo_find(
        self.config["MONGO_DB"],
        "user_factors",
        {"user_id": user_id, "factor_name": factor_name}
    )

    # 查询缓存的因子数据
    collection_name = f"factor_{factor_name}_{user_id}"
    if collection_name in self.db_handler.mongo_client[self.config["MONGO_DB"]].list_collection_names():
        records = self.db_handler.mongo_find(...)

# 修改后代码
def get_custom_factor(self, factor_logger, user_id, factor_name, start_date, end_date):
    # 从 SQLite 查询因子定义
    records = self.parquet_handler.metadata_query(
        table_name='user_factors',
        filters={'user_id': user_id, 'factor_name': factor_name}
    )

    # 查询缓存的因子数据
    factor_data_path = self.parquet_handler._get_dataset_path(
        f'user_factors/{user_id}/{factor_name}'
    )

    if factor_data_path.exists():
        df = self.parquet_handler.query(
            dataset_name=f'user_factors/{user_id}/{factor_name}',
            filters=[
                ('date', '>=', start_date),
                ('date', '<=', end_date)
            ]
        )
```

### 4.3 panda_data_hub 模块修改

#### 4.3.1 数据清洗服务改造
**文件路径**: `panda_data_hub/services/*_stock_market_clean_service.py`

**修改要点**:
1. 移除 MongoDB bulk_write 逻辑
2. 改为写入 Parquet 文件
3. 使用日期分区策略
4. 处理数据去重和更新

**修改示例**:
```python
# 当前代码
def clean_meta_market_data(self, date_str):
    # ... 获取数据 ...

    bulk_operations = [
        UpdateOne(
            {"symbol": row["symbol"], "date": row["date"]},
            {"$set": row_dict},
            upsert=True
        )
        for _, row in df_data.iterrows()
    ]

    collection.bulk_write(bulk_operations)

# 修改后代码
def clean_meta_market_data(self, date_str):
    # ... 获取数据 ...

    # 直接写入 Parquet (按日期分区)
    self.parquet_handler.insert_batch(
        dataset_name='stock_market',
        data=df_data
    )

    # 或者使用更精细的控制
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df_data)
    pq.write_to_dataset(
        table,
        root_path=str(self.parquet_handler.data_root / 'stock_market'),
        partition_cols=['date'],
        existing_data_behavior='delete_matching'  # 覆盖同分区数据
    )
```

#### 4.3.2 mongo_utils 改造
**文件路径**: `panda_data_hub/utils/mongo_utils.py`

**修改要点**:
- 移除索引创建逻辑
- 改为确保 Parquet 目录结构存在

**修改示例**:
```python
# 当前代码
def ensure_collection_and_indexes(table_name):
    db = DatabaseHandler(config).mongo_client[config["MONGO_DB"]]
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
    collection.create_index([('symbol', 1), ('date', 1)], ...)

# 修改后代码
def ensure_dataset_path(dataset_name):
    """确保数据集目录存在"""
    from pathlib import Path

    parquet_handler = ParquetHandler(config)
    dataset_path = parquet_handler._get_dataset_path(dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"确保数据集路径存在: {dataset_path}")
```

### 4.4 panda_factor 模块修改

#### 4.4.1 FactorDataHandler 改造
**文件路径**: `panda_factor/generate/factor_data_handler.py`

**修改要点**:
- 使用 Parquet 并行读取因子数据
- 保持 ThreadPoolExecutor 逻辑

**修改示例**:
```python
# 修改 fetch_factor 函数
def fetch_factor(factor_name: str, start_date: str, end_date: str,
                symbols: Optional[List[str]], data_provider) -> tuple:
    # data_provider 内部已经使用 ParquetHandler
    data = data_provider.get_factor_data(factor_name, start_date, end_date, symbols)
    # ... 其余逻辑不变 ...
```

#### 4.4.2 因子分析结果存储
**文件路径**: `panda_factor/analysis/factor_analysis.py`

**修改要点**:
- 分析指标存储到 JSON
- 时序数据存储到 Parquet
- 图表数据存储到 JSON

**修改示例**:
```python
def save_analysis_results(self, user_id, factor_name, results):
    import json
    from pathlib import Path

    # 创建结果目录
    result_dir = Path(self.config['DATA_ROOT_PATH']) / 'analysis_results' / user_id / factor_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存元数据和指标 (JSON)
    metadata = {
        'factor_name': factor_name,
        'user_id': user_id,
        'metrics': results['metrics'],
        'parameters': results['parameters'],
        'created_at': datetime.now().isoformat()
    }

    with open(result_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 保存时序数据 (Parquet)
    if 'returns' in results:
        results['returns'].to_parquet(result_dir / 'returns.parquet')

    if 'ic_series' in results:
        results['ic_series'].to_parquet(result_dir / 'ic_series.parquet')

    # 保存图表数据 (JSON)
    if 'charts' in results:
        charts_dir = result_dir / 'charts'
        charts_dir.mkdir(exist_ok=True)

        for chart_name, chart_data in results['charts'].items():
            with open(charts_dir / f'{chart_name}.json', 'w', encoding='utf-8') as f:
                json.dump(chart_data, f, ensure_ascii=False)
```

### 4.5 panda_factor_server 模块修改

#### 4.5.1 UserFactorService 改造
**文件路径**: `panda_factor_server/services/user_factor_service.py`

**修改要点**:
- 用户因子列表查询改为 SQLite
- 支持分页和排序
- 因子数据查询改为 Parquet

**修改示例**:
```python
# 当前代码
def get_user_factor_list(user_id, page, page_size, sort_field, sort_order):
    query = {"user_id": user_id}
    total = _db_handler.mongo_client["panda"]["user_factors"].count_documents(query)

    cursor = _db_handler.mongo_client["panda"]["user_factors"].find(query)
    cursor = cursor.sort(sort_field, DESCENDING if sort_order == "desc" else ASCENDING)
    cursor = cursor.skip((page - 1) * page_size).limit(page_size)

# 修改后代码
def get_user_factor_list(user_id, page, page_size, sort_field, sort_order):
    # 使用 SQLite 查询
    parquet_handler = ParquetHandler(config)

    # 获取总数
    all_factors = parquet_handler.metadata_query(
        table_name='user_factors',
        filters={'user_id': user_id}
    )
    total = len(all_factors)

    # 分页查询
    factors = parquet_handler.metadata_query(
        table_name='user_factors',
        filters={'user_id': user_id},
        sort=[(sort_field, -1 if sort_order == "desc" else 1)],
        limit=page_size
    )

    # 手动实现 skip (SQLite 不支持 OFFSET)
    offset = (page - 1) * page_size
    factors = factors[offset:offset + page_size]
```

### 4.6 panda_llm 模块修改

#### 4.6.1 对话历史存储
**文件路径**: `panda_llm/services/mongodb.py`

**修改建议**:
- 对话历史不适合 Parquet,建议使用 SQLite 或 JSON
- 创建 `conversation_history` 表

**修改示例**:
```python
# 重命名文件为 storage.py
# 使用 SQLite 存储对话历史

class ConversationStorage:
    def __init__(self, config):
        self.parquet_handler = ParquetHandler(config)
        self._init_tables()

    def _init_tables(self):
        self.parquet_handler.metadata_db.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP
            )
        ''')

    def save_message(self, user_id, conversation_id, role, content):
        import uuid
        from datetime import datetime

        self.parquet_handler.metadata_insert(
            'conversation_history',
            {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'conversation_id': conversation_id,
                'role': role,
                'content': content,
                'created_at': datetime.now()
            }
        )

    def get_conversation(self, conversation_id):
        return self.parquet_handler.metadata_query(
            'conversation_history',
            filters={'conversation_id': conversation_id},
            sort=[('created_at', 1)]
        )
```

---

## 五、数据迁移方案

### 5.1 MongoDB 数据导出

#### 5.1.1 导出市场数据
```python
#!/usr/bin/env python
"""
MongoDB 数据导出到 Parquet
"""
import pandas as pd
from pymongo import MongoClient
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime

def export_stock_market_to_parquet():
    # 连接 MongoDB
    client = MongoClient("mongodb://panda:panda@127.0.0.1:27017/admin")
    db = client['panda']
    collection = db['stock_market']

    # 获取所有唯一日期
    unique_dates = collection.distinct('date')
    print(f"找到 {len(unique_dates)} 个交易日")

    # 按日期分区导出
    for date in tqdm(unique_dates):
        # 查询该日期的所有数据
        cursor = collection.find({'date': date})
        data = list(cursor)

        if not data:
            continue

        # 转换为 DataFrame
        df = pd.DataFrame(data)
        df = df.drop(columns=['_id'], errors='ignore')

        # 写入 Parquet
        output_path = f'panda_data/stock_market/date={date}/part-0000.parquet'

        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 写入文件
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

    print("导出完成!")

def export_user_factors_to_sqlite():
    """导出用户因子元数据到 SQLite"""
    import sqlite3
    from bson import ObjectId
    import json

    # 连接 MongoDB
    client = MongoClient("mongodb://panda:panda@127.0.0.1:27017/admin")
    db = client['panda']
    collection = db['user_factors']

    # 连接 SQLite
    conn = sqlite3.connect('panda_data/metadata.db')

    # 查询所有因子
    factors = list(collection.find())

    for factor in tqdm(factors):
        # 转换 ObjectId 为字符串
        factor_id = str(factor['_id'])

        # 转换 params 为 JSON 字符串
        params = json.dumps(factor.get('params', {}))

        # 插入 SQLite
        conn.execute('''
            INSERT OR REPLACE INTO user_factors
            (id, user_id, factor_name, code_type, code, params, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            factor_id,
            factor.get('user_id'),
            factor.get('factor_name'),
            factor.get('code_type'),
            factor.get('code'),
            params,
            factor.get('created_at'),
            factor.get('updated_at')
        ))

    conn.commit()
    conn.close()

    print("用户因子元数据导出完成!")

if __name__ == '__main__':
    export_stock_market_to_parquet()
    export_user_factors_to_sqlite()
```

#### 5.1.2 批量导出脚本
```bash
#!/bin/bash
# export_all_data.sh

echo "开始导出 MongoDB 数据到 Parquet..."

# 导出 stock_market
python scripts/export_to_parquet.py --collection stock_market

# 导出 future_market
python scripts/export_to_parquet.py --collection future_market

# 导出 factor_base
python scripts/export_to_parquet.py --collection factor_base

# 导出用户因子元数据
python scripts/export_to_parquet.py --collection user_factors --target sqlite

# 导出因子提交记录
python scripts/export_to_parquet.py --collection user_factor_submissions --target sqlite

# 导出任务记录
python scripts/export_to_parquet.py --collection tasks --target sqlite

echo "数据导出完成!"
```

### 5.2 数据验证

```python
def validate_migration():
    """验证迁移后的数据完整性"""
    from pymongo import MongoClient
    import pandas as pd

    # 连接 MongoDB
    mongo_client = MongoClient("mongodb://panda:panda@127.0.0.1:27017/admin")
    mongo_db = mongo_client['panda']

    # 初始化 ParquetHandler
    parquet_handler = ParquetHandler(config)

    # 验证 stock_market
    mongo_count = mongo_db['stock_market'].count_documents({})
    parquet_df = parquet_handler.query('stock_market')
    parquet_count = len(parquet_df)

    print(f"MongoDB stock_market 记录数: {mongo_count}")
    print(f"Parquet stock_market 记录数: {parquet_count}")

    if mongo_count == parquet_count:
        print("✓ stock_market 数据验证通过")
    else:
        print("✗ stock_market 数据验证失败")

    # 验证用户因子
    mongo_factors = list(mongo_db['user_factors'].find())
    sqlite_factors = parquet_handler.metadata_query('user_factors')

    print(f"MongoDB user_factors 记录数: {len(mongo_factors)}")
    print(f"SQLite user_factors 记录数: {len(sqlite_factors)}")

    if len(mongo_factors) == len(sqlite_factors):
        print("✓ user_factors 数据验证通过")
    else:
        print("✗ user_factors 数据验证失败")
```

---

## 六、实施步骤与时间规划

### 6.1 阶段划分

#### 第一阶段:基础设施准备 (预计 3-5 天)
1. **Day 1-2**: 设计并实现 `ParquetHandler` 类
   - 实现基础 CRUD 操作
   - 实现 SQLite 元数据管理
   - 编写单元测试

2. **Day 3**: 修改配置系统
   - 更新 `config.yaml`
   - 添加 Parquet 配置项
   - 配置环境变量

3. **Day 4-5**: 数据迁移脚本开发
   - 编写 MongoDB 导出脚本
   - 编写数据验证脚本
   - 测试数据完整性

#### 第二阶段:核心模块改造 (预计 5-7 天)
1. **Day 6-7**: panda_data 模块改造
   - 改造 `MarketDataReader`
   - 改造 `FactorReader`
   - 单元测试

2. **Day 8-9**: panda_data_hub 模块改造
   - 改造数据清洗服务
   - 改造 `mongo_utils`
   - 集成测试

3. **Day 10-11**: panda_factor 模块改造
   - 改造 `FactorDataHandler`
   - 改造因子分析结果存储
   - 功能测试

4. **Day 12**: panda_factor_server 模块改造
   - 改造用户因子服务
   - API 测试

#### 第三阶段:测试与优化 (预计 3-4 天)
1. **Day 13-14**: 集成测试
   - 端到端功能测试
   - 性能压测
   - 数据一致性验证

2. **Day 15**: 性能优化
   - 查询性能调优
   - 分区策略优化
   - 缓存策略实施

3. **Day 16**: 文档更新
   - 更新 README
   - 更新 API 文档
   - 编写迁移指南

#### 第四阶段:上线与监控 (预计 2-3 天)
1. **Day 17**: 数据迁移
   - 备份 MongoDB 数据
   - 执行数据迁移
   - 数据验证

2. **Day 18-19**: 灰度发布
   - 部分模块切换到 Parquet
   - 监控性能和错误
   - 全量切换

### 6.2 风险控制

#### 6.2.1 回滚方案
- 保留 MongoDB 数据不删除
- `ParquetHandler` 可与 `DatabaseHandler` 并存
- 通过配置开关快速切换

#### 6.2.2 兼容性方案
```python
# 配置开关
STORAGE_BACKEND = "parquet"  # 或 "mongodb"

if STORAGE_BACKEND == "parquet":
    data_handler = ParquetHandler(config)
else:
    data_handler = DatabaseHandler(config)
```

---

## 七、性能预期

### 7.1 存储空间对比

| 数据类型 | MongoDB 存储 | Parquet 存储 (snappy) | 压缩比 |
|---------|-------------|---------------------|--------|
| stock_market (1年) | ~50 GB | ~15 GB | 3.3x |
| factor_base (1年) | ~30 GB | ~10 GB | 3.0x |
| user_factors (元数据) | ~100 MB | ~30 MB (SQLite) | 3.3x |

### 7.2 查询性能对比

| 查询类型 | MongoDB | Parquet | 提升 |
|---------|---------|---------|------|
| 单日全市场查询 | 2-3 秒 | 0.5-1 秒 | 2-3x |
| 日期范围查询 (1月) | 15-20 秒 | 3-5 秒 | 3-4x |
| 列式查询 (仅 close) | 10 秒 | 1-2 秒 | 5-10x |
| 股票代码过滤 | 3-5 秒 | 1-2 秒 | 2-3x |

### 7.3 写入性能对比

| 操作类型 | MongoDB | Parquet | 差异 |
|---------|---------|---------|------|
| 单日数据写入 | 5-10 秒 | 2-5 秒 | 快 1-2x |
| 批量更新 (覆盖) | 慢 (需 upsert) | 快 (直接覆盖分区) | 快 3-5x |
| 追加数据 | 快 | 稍慢 (需重写分区) | 慢 1.2-1.5x |

---

## 八、优缺点分析

### 8.1 迁移至 Parquet 的优势

1. **更低的存储成本**
   - 列式压缩,压缩比是 MongoDB 的 3-4 倍
   - 预计节省 60-70% 的存储空间

2. **更快的列式查询**
   - 因子计算场景主要是列式访问
   - 过滤下推和分区剪枝优化
   - 预计查询性能提升 2-5 倍

3. **更好的可移植性**
   - 文件系统存储,无需运行数据库服务
   - 易于备份、迁移和版本管理
   - 支持云存储 (S3, OSS 等)

4. **更好的生态集成**
   - 与 pandas、dask、polars 无缝对接
   - 支持 Spark、Hive 等大数据工具
   - 更易于数据分析和机器学习

5. **更低的运维成本**
   - 无需管理数据库实例
   - 无需担心索引维护
   - 更简单的扩展策略

### 8.2 迁移的劣势与挑战

1. **更新操作变慢**
   - Parquet 是不可变的,更新需要重写
   - 不适合频繁更新的场景
   - 解决方案: 使用日期分区,只重写当天数据

2. **事务支持弱**
   - Parquet 不支持 ACID 事务
   - 需要应用层保证一致性
   - 解决方案: 使用 Delta Lake 或 Hudi (可选)

3. **查询灵活性降低**
   - 不支持复杂的 JOIN 和聚�� (需 pandas 处理)
   - 不支持全文搜索和正则匹配
   - 解决方案: 元数据使用 SQLite

4. **学习成本**
   - 团队需要学习 PyArrow/Pandas
   - 需要理解分区策略
   - 解决方案: 提供培训和文档

5. **元数据管理复杂**
   - 用户因子、任务等非时序数据不适合 Parquet
   - 需要引入 SQLite 或 JSON
   - 解决方案: 混合存储策略

### 8.3 适用场景建议

**推荐使用 Parquet 的数据**:
- ✅ 股票市场行情数据 (stock_market)
- ✅ 期货市场行情数据 (future_market)
- ✅ 基础因子数据 (factor_base)
- ✅ 因子计算结果 (factor_{name}_{user_id})
- ✅ 因子分析时序数据 (returns, ic_series)

**推荐使用 SQLite 的数据**:
- ✅ 用户因子定义 (user_factors)
- ✅ 用户因子提交记录 (user_factor_submissions)
- ✅ 后台任务状态 (tasks)
- ✅ LLM 对话历史 (conversation_history)

**推荐使用 JSON 的数据**:
- ✅ 因子分析图表数据 (charts)
- ✅ 系统配置文件

---

## 九、依赖变更

### 9.1 需要添加的依赖

**requirements.txt 新增**:
```txt
# Parquet 支持
pyarrow>=14.0.0
fastparquet>=2023.10.0  # 可选

# 数据处理
pandas>=2.0.0
numpy>=1.24.0

# 元数据存储 (Python 自带 sqlite3,无需额外安装)
```

### 9.2 需要移除的依赖

**requirements.txt 移除**:
```txt
# 可以移除 (如果不再使用 MongoDB)
pymongo>=4.0.0
```

### 9.3 setup.py 修改

```python
# 每个模块的 setup.py 都需要更新依赖

# panda_common/setup.py
install_requires=[
    'pyarrow>=14.0.0',
    'pandas>=2.0.0',
    'pyyaml',
    'loguru',
    # 移除 'pymongo>=4.0.0'
]

# panda_data/setup.py
install_requires=[
    'panda_common',
    'pyarrow>=14.0.0',
    'pandas>=2.0.0',
]
```

---

## 十、测试计划

### 10.1 单元测试

#### 10.1.1 ParquetHandler 测试
```python
import pytest
from panda_common.handlers.parquet_handler import ParquetHandler

def test_insert_and_query():
    config = {...}
    handler = ParquetHandler(config)

    # 测试插入
    data = pd.DataFrame({
        'symbol': ['000001.SZ'],
        'date': [20240101],
        'close': [10.5]
    })

    handler.insert_batch('test_dataset', data)

    # 测试查询
    result = handler.query('test_dataset', filters=[('date', '==', 20240101)])

    assert len(result) == 1
    assert result.iloc[0]['close'] == 10.5

def test_update():
    # 测试更新操作
    pass

def test_delete():
    # 测试删除操作
    pass

def test_metadata_operations():
    # 测试 SQLite 元数据操作
    pass
```

### 10.2 集成测试

#### 10.2.1 数据完整性测试
```python
def test_market_data_integrity():
    """测试市场数据完整性"""
    # 1. 写入数据
    # 2. 读取数据
    # 3. 验证数据一致性
    pass

def test_factor_calculation():
    """测试因子计算流程"""
    # 1. 获取基础数据
    # 2. 计算因子
    # 3. 存储结果
    # 4. 读取验证
    pass
```

### 10.3 性能测试

```python
import time

def benchmark_query_performance():
    """性能基准测试"""
    # 测试不同数据量的查询性能
    test_cases = [
        {'days': 1, 'symbols': 5000},
        {'days': 30, 'symbols': 5000},
        {'days': 365, 'symbols': 5000}
    ]

    for case in test_cases:
        start = time.time()
        data = get_market_data(..., **case)
        elapsed = time.time() - start

        print(f"查询 {case['days']} 天, {case['symbols']} 股票, 耗时: {elapsed:.2f} 秒")
```

---

## 十一、文档更新清单

### 11.1 需要更新的文档

- [x] **CLAUDE.md**: 更新数据库架构说明
- [x] **MongoDB使用文档.md**: 重命名为 "Parquet存储使用文档.md"
- [x] **README.md**: 更新安装和配置说明
- [x] **快速开始指南.md**: 更新数据初始化步骤
- [x] **行情数据更新机制文档.md**: 更新数据存储方式说明

### 11.2 新增文档

- [ ] **Parquet存储使用文档.md**: Parquet 使用指南
- [ ] **数据迁移指南.md**: MongoDB 迁移到 Parquet 的详细步骤
- [ ] **性能优化指南.md**: Parquet 查询和存储优化

---

## 十二、总结与建议

### 12.1 核心建议

1. **分阶段实施**: 不要一次性全部迁移,先从读多写少的模块开始
2. **保留回滚能力**: MongoDB 数据保留 1-2 个月,确保可以快速回滚
3. **混合存储策略**: 时序数据用 Parquet,元数据用 SQLite
4. **性能监控**: 迁移后持续监控性能指标
5. **团队培训**: 确保团队理解 Parquet 的特性和最佳实践

### 12.2 可选增强方案

如果未来数据量进一步增长,可以考虑:

1. **使用 Delta Lake**: 提供 ACID 事务支持和时间旅行功能
2. **使用 Dask**: 支持超大数据集的并行计算
3. **使用 ClickHouse**: 作为 OLAP 数据库,提供更强的查询能力
4. **使用对象存储**: 将 Parquet 文件存储到 S3/OSS,降低成本

### 12.3 预期收益

- **存储成本**: 降低 60-70%
- **查询性能**: 提升 2-5 倍 (列式查询场景)
- **运维成本**: 降低 80% (无需管理数据库)
- **开发效率**: 与数据科学工具更好集成

---

## 附录

### 附录 A: Parquet 最佳实践

1. **分区策略**:
   - 一级分区: 日期 (date)
   - 二级分区: 可选,适用于超大数据集

2. **文件大小**:
   - 单个文件 128MB - 512MB 最佳
   - 避免过小文件 (< 10MB)

3. **压缩算法**:
   - 实时数据: snappy
   - 归档数据: gzip 或 zstd

4. **Row Group 大小**:
   - 推荐 100 万行
   - 内存充足可增加到 500 万行

### 附录 B: 相关工具和资源

- **PyArrow 文档**: https://arrow.apache.org/docs/python/
- **Pandas Parquet**: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
- **Parquet 格式规范**: https://parquet.apache.org/docs/
- **Delta Lake**: https://delta.io/

### 附录 C: 常见问题

**Q1: Parquet 如何处理数据更新?**
A: Parquet 是不可变的,更新需要重写分区。建议按日期分区,每日更新只重写当天数据。

**Q2: Parquet 支持事务吗?**
A: 原生 Parquet 不支持事务,可以使用 Delta Lake 或 Hudi 提供事务支持。

**Q3: 如何处理历史数据修正?**
A: 重写对应日期的分区文件。

**Q4: Parquet 查询性能优化要点?**
A: 1) 使用分区剪枝 2) 列裁剪 3) 过滤下推 4) 合理的文件大小。

---

**文档结束**
