use anyhow::Result;
use dubbo::{
    codegen::async_trait, config::{protocol::{Protocol, ProtocolConfig}, registry::RegistryConfig, service::ServiceConfig}, status::DubboError
};
use ordered_float::OrderedFloat;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde::{Deserialize, Serialize};
use serde_json::to_string;
use slab::Slab;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs::{File, OpenOptions},
    io::Write,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::order_match::{interfaces::interfaces::MatchingService, order_shard::order_shard::ShardManager, structs::structs::{OrderEntry, OrderKind, OrderRequest, OrderResponse, PriceLevel, Side, Trade}};

pub struct MatchingServiceImpl {
    managers: HashMap<String, Arc<ShardManager>>,
}

impl MatchingServiceImpl {
    pub fn new(
        markets: Vec<String>,
        kafka_brokers: &str,
        kafka_topic: &str,
        wal_dir: &str,
    ) -> Self {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", kafka_brokers)
            .create()
            .expect("producer");
        let mut managers = HashMap::new();
        for m in markets {
            let mgr = Arc::new(ShardManager::new(
                &m,
                producer.clone(),
                kafka_topic,
                wal_dir,
            ));
            mgr.start_with(1);
            managers.insert(m.clone(), mgr);
        }
        MatchingServiceImpl { managers }
    }
}

#[async_trait]
impl MatchingService for MatchingServiceImpl {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError> {
        if let Some(mgr) = self.managers.get(&req.market) {
            if let Err(e) = mgr.route(req.clone()) {
                return Ok(OrderResponse {
                    success: false,
                    message: format!("route err: {}", e),
                    trades: vec![],
                    remaining: req.quantity,
                });
            }
            Ok(OrderResponse {
                success: true,
                message: "accepted".to_string(),
                trades: vec![],
                remaining: req.quantity,
            })
        } else {
            Ok(OrderResponse {
                success: false,
                message: "unknown market".to_string(),
                trades: vec![],
                remaining: req.quantity,
            })
        }
    }
}

pub fn start_rpc(markets: Vec<String>, kafka_brokers: &str, kafka_topic: &str, wal_dir: &str) {
    let registry_config = RegistryConfig {
        protocol: "zookeeper".to_string(),
        address: "127.0.0.1:2181".to_string(),
        ..Default::default()
    };
    let protocal = Protocol {
        name: "dubbo".to_string(),
        port: 20086.to_string(),
        ..Default::default()
    };
    let protocol_config = ProtocolConfig::default();
    protocol_config.insert("dubbo".to_string(), protocal);
    let svc = MatchingServiceImpl::new(markets, kafka_brokers, kafka_topic, wal_dir);
    let service_config = ServiceConfig::new("org.dex.matching.MatchingService", svc)
        .with_group("/dev/metawebthree")
        .with_protocol("dubbo");
    // @TDOO fix dubbo api error
    dubbo::protocol::triple::TRIPLE_SERVICES
        .with_registry(registry_config)
        .with_protocol(protocol_config)
        .with_service(service_config)
        .serve();
}

// #[tokio::main]
async fn start() {
    let markets = vec![
        "BTC/USDT".to_string(),
        "ETH/USDT".to_string(),
        "DOGE/USDT".to_string(),
    ];
    start_rpc(markets, "localhost:9092", "dex-trades", "/tmp");
}
