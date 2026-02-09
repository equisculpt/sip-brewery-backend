# 🚀 PHASE 2: REAL-TIME INTELLIGENCE INFRASTRUCTURE

**Timeline:** Months 5-8  
**Status:** Architecture Design Complete  
**Objective:** Build real-time data processing infrastructure with knowledge graphs and vector databases

---

## 🎯 OBJECTIVES

### 1. Kafka + Flink Streaming Pipeline
- **Real-time data ingestion** from multiple sources
- **Stream processing** with Apache Flink
- **Event-driven architecture** for instant updates
- **Sub-second latency** for critical operations

### 2. Knowledge Graph (1M+ Nodes)
- **Neo4j graph database** for relationship modeling
- **Fund relationships**, holdings, correlations
- **User behavior patterns** and preferences
- **Market dynamics** and sector connections

### 3. Vector Database for Semantic Search
- **Pinecone/Weaviate** for similarity search
- **Fund recommendations** based on embeddings
- **Semantic query** understanding
- **Portfolio similarity** matching

### 4. Sub-Second Feature Computation
- **Real-time feature engineering**
- **Streaming aggregations**
- **Low-latency predictions**
- **Cache optimization**

---

## 📊 ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
├─────────────────────────────────────────────────────────────┤
│  Market Data │ User Actions │ Fund Updates │ News/Sentiment │
└──────┬────────────┬────────────┬────────────┬───────────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                   KAFKA CLUSTER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ market-  │  │  user-   │  │  fund-   │  │  news-   │   │
│  │  data    │  │ events   │  │ updates  │  │  feed    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└──────┬────────────┬────────────┬────────────┬───────────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                 APACHE FLINK CLUSTER                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Stream Processing Jobs                            │     │
│  │  • Feature Engineering                             │     │
│  │  • Aggregations (windowed, session)                │     │
│  │  • Pattern Detection (CEP)                         │     │
│  │  • Anomaly Detection                               │     │
│  └────────────────────────────────────────────────────┘     │
└──────┬────────────┬────────────┬────────────┬───────────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐
│  KNOWLEDGE   │ │   VECTOR     │ │    FEAST     │ │  REDIS  │
│    GRAPH     │ │  DATABASE    │ │   FEATURE    │ │  CACHE  │
│   (Neo4j)    │ │ (Pinecone)   │ │    STORE     │ │         │
└──────────────┘ └──────────────┘ └──────────────┘ └─────────┘
       │            │            │            │
       └────────────┴────────────┴────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │   ML INFERENCE LAYER    │
       │  (Phase 1 Models)       │
       └─────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │   REST API / WebSocket  │
       │   (Node.js Backend)     │
       └─────────────────────────┘
```

---

## 🔧 COMPONENT SPECIFICATIONS

### 1. Kafka Streaming Infrastructure

#### Topics Design
```yaml
market-data:
  partitions: 12
  replication: 3
  retention: 7 days
  compression: snappy
  
user-events:
  partitions: 24
  replication: 3
  retention: 30 days
  compression: lz4
  
fund-updates:
  partitions: 6
  replication: 3
  retention: 90 days
  compression: snappy
  
portfolio-changes:
  partitions: 12
  replication: 3
  retention: 365 days
  compression: gzip
```

#### Producers
- **Market Data Producer**: Ingests real-time market data
- **User Event Producer**: Captures all user actions
- **Fund Update Producer**: Tracks fund NAV, holdings changes
- **News Producer**: Sentiment analysis from news feeds

#### Consumers
- **Flink Consumer**: Stream processing
- **Feature Store Consumer**: Real-time feature updates
- **Analytics Consumer**: Metrics and monitoring
- **Backup Consumer**: Data archival

---

### 2. Apache Flink Processing

#### Stream Processing Jobs

**Job 1: Real-Time Feature Engineering**
```java
DataStream<UserEvent> userEvents = env
    .addSource(new FlinkKafkaConsumer<>("user-events", schema, props))
    .keyBy(event -> event.getUserId())
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new FeatureAggregator());
```

**Job 2: Market Anomaly Detection**
```java
DataStream<MarketData> marketData = env
    .addSource(new FlinkKafkaConsumer<>("market-data", schema, props))
    .keyBy(data -> data.getSymbol())
    .flatMap(new AnomalyDetector())
    .filter(anomaly -> anomaly.getSeverity() > 0.8);
```

**Job 3: Portfolio Risk Monitoring**
```java
DataStream<PortfolioUpdate> portfolios = env
    .addSource(new FlinkKafkaConsumer<>("portfolio-changes", schema, props))
    .keyBy(portfolio -> portfolio.getUserId())
    .process(new RiskCalculator());
```

**Job 4: Pattern Detection (CEP)**
```java
Pattern<UserEvent, ?> pattern = Pattern.<UserEvent>begin("start")
    .where(evt -> evt.getAction().equals("VIEW_FUND"))
    .followedBy("middle")
    .where(evt -> evt.getAction().equals("ADD_TO_WATCHLIST"))
    .followedBy("end")
    .where(evt -> evt.getAction().equals("BUY"))
    .within(Time.hours(24));
```

---

### 3. Knowledge Graph (Neo4j)

#### Node Types
```cypher
// Fund nodes
CREATE (f:Fund {
  id: 'FUND001',
  name: 'ABC Equity Fund',
  category: 'EQUITY',
  aum: 5000000000,
  expense_ratio: 0.015
})

// Stock nodes
CREATE (s:Stock {
  id: 'RELIANCE',
  name: 'Reliance Industries',
  sector: 'ENERGY',
  market_cap: 1500000000000
})

// Sector nodes
CREATE (sec:Sector {
  id: 'ENERGY',
  name: 'Energy Sector'
})

// User nodes
CREATE (u:User {
  id: 'USER123',
  risk_profile: 'MODERATE',
  age: 35
})

// AMC nodes
CREATE (amc:AMC {
  id: 'AMC001',
  name: 'ABC Asset Management'
})
```

#### Relationship Types
```cypher
// Fund holdings
CREATE (f:Fund)-[:HOLDS {weight: 0.15}]->(s:Stock)

// Fund correlations
CREATE (f1:Fund)-[:CORRELATED_WITH {correlation: 0.85}]->(f2:Fund)

// User investments
CREATE (u:User)-[:INVESTED_IN {amount: 100000, date: '2026-01-01'}]->(f:Fund)

// Fund management
CREATE (amc:AMC)-[:MANAGES]->(f:Fund)

// Sector membership
CREATE (f:Fund)-[:BELONGS_TO]->(sec:Sector)

// Similar users
CREATE (u1:User)-[:SIMILAR_TO {similarity: 0.92}]->(u2:User)
```

#### Graph Queries
```cypher
// Find similar funds based on holdings
MATCH (f1:Fund)-[:HOLDS]->(s:Stock)<-[:HOLDS]-(f2:Fund)
WHERE f1.id = 'FUND001' AND f1 <> f2
WITH f2, COUNT(s) AS common_stocks
ORDER BY common_stocks DESC
LIMIT 10
RETURN f2

// Find funds in user's risk profile
MATCH (u:User {id: 'USER123'})
MATCH (f:Fund)
WHERE f.risk_level = u.risk_profile
RETURN f

// Detect portfolio concentration risk
MATCH (u:User)-[inv:INVESTED_IN]->(f:Fund)-[:BELONGS_TO]->(sec:Sector)
WITH u, sec, SUM(inv.amount) AS sector_exposure
WHERE sector_exposure > 0.35 * u.total_portfolio_value
RETURN u, sec, sector_exposure
```

---

### 4. Vector Database (Pinecone)

#### Index Configuration
```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index for fund embeddings
pinecone.create_index(
    "fund-embeddings",
    dimension=768,  # BERT embedding size
    metric="cosine",
    pods=4,
    replicas=2,
    pod_type="p1.x1"
)

# Create index for user profiles
pinecone.create_index(
    "user-profiles",
    dimension=512,
    metric="cosine",
    pods=2,
    replicas=2
)
```

#### Embedding Generation
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

def generate_fund_embedding(fund_description):
    inputs = tokenizer(fund_description, return_tensors="pt", 
                      padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embedding.flatten()

# Generate and store embeddings
fund_description = "Large cap equity fund focusing on technology sector..."
embedding = generate_fund_embedding(fund_description)

index = pinecone.Index("fund-embeddings")
index.upsert([("FUND001", embedding.tolist(), {"category": "EQUITY"})])
```

#### Similarity Search
```python
# Find similar funds
query_embedding = generate_fund_embedding("Technology focused growth fund")
results = index.query(
    vector=query_embedding.tolist(),
    top_k=10,
    include_metadata=True
)

for match in results['matches']:
    print(f"Fund: {match['id']}, Score: {match['score']}")
```

---

### 5. Sub-Second Feature Computation

#### Real-Time Feature Pipeline
```python
from feast import FeatureStore
from kafka import KafkaConsumer
import json

store = FeatureStore(repo_path="feature_repo")

consumer = KafkaConsumer(
    'user-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    event = message.value
    user_id = event['user_id']
    
    # Compute features in real-time
    features = {
        'user_id': user_id,
        'last_action': event['action'],
        'action_count_1h': get_action_count(user_id, hours=1),
        'portfolio_value': get_portfolio_value(user_id),
        'risk_score': calculate_risk_score(user_id)
    }
    
    # Write to feature store (< 100ms)
    store.write_to_online_store(
        feature_view_name="user_realtime_features",
        df=pd.DataFrame([features])
    )
```

---

## 🚀 IMPLEMENTATION PLAN

### Week 1-2: Kafka Setup
- [ ] Install Kafka cluster (3 brokers)
- [ ] Configure topics and partitions
- [ ] Set up producers for data sources
- [ ] Implement consumers
- [ ] Test throughput (target: 100K msgs/sec)

### Week 3-4: Flink Deployment
- [ ] Deploy Flink cluster (JobManager + TaskManagers)
- [ ] Implement feature engineering job
- [ ] Implement anomaly detection job
- [ ] Implement pattern detection (CEP)
- [ ] Set up checkpointing and state management

### Week 5-6: Knowledge Graph
- [ ] Deploy Neo4j cluster
- [ ] Design graph schema
- [ ] Import initial data (funds, stocks, sectors)
- [ ] Create relationship mappings
- [ ] Implement graph queries
- [ ] Build graph update pipeline

### Week 7-8: Vector Database
- [ ] Set up Pinecone/Weaviate
- [ ] Generate fund embeddings
- [ ] Generate user profile embeddings
- [ ] Implement similarity search
- [ ] Build recommendation engine
- [ ] Integrate with API

---

## 📈 PERFORMANCE TARGETS

| Metric | Target | Measurement |
|--------|--------|-------------|
| Kafka Throughput | > 100K msgs/sec | Producer rate |
| Flink Latency | < 100ms (p99) | Event-to-output |
| Feature Computation | < 50ms | End-to-end |
| Graph Query | < 200ms | Complex queries |
| Vector Search | < 100ms | Top-10 results |
| Overall Latency | < 500ms | User request to response |

---

## 💰 INFRASTRUCTURE COSTS (Monthly)

| Component | Specification | Cost |
|-----------|--------------|------|
| Kafka Cluster | 3 brokers, 2TB storage | $500 |
| Flink Cluster | 4 TaskManagers, 16 cores | $800 |
| Neo4j | Enterprise, 1M nodes | $1,200 |
| Pinecone | 4 pods, 1M vectors | $700 |
| Redis Cluster | 32GB memory | $300 |
| **Total** | | **$3,500/month** |

---

## 🔐 SECURITY & COMPLIANCE

- **Data Encryption**: TLS for all inter-service communication
- **Access Control**: RBAC for all databases
- **Audit Logging**: All data access logged
- **Data Retention**: Configurable per topic/collection
- **Privacy**: PII anonymization in streams

---

## 📊 MONITORING & ALERTING

### Metrics to Track
- Kafka lag per consumer group
- Flink job backpressure
- Neo4j query performance
- Pinecone index size and latency
- Feature freshness

### Alerts
- Kafka lag > 10K messages
- Flink job failure
- Graph query > 1s
- Vector search > 500ms
- Feature staleness > 5 minutes

---

**Phase 2 Status:** Architecture Complete, Ready for Implementation  
**Next:** Begin Kafka cluster setup and producer implementation
