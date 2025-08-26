use anyhow::Result;
use dubbo::{
    codegen::async_trait,
    config::{
        protocol::{Protocol, ProtocolConfig},
        provider::ProviderConfig,
        registry::RegistryConfig,
        service::ServiceConfig,
        RootConfig,
    },
    status::DubboError,
    Dubbo,
};
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer};
use std::{
    collections::{HashMap},
    sync::{Arc}
};

use crate::order_match::{
    interfaces::interfaces::MatchingService,
    order_shard::order_shard::ShardManager,
    structs::structs::{
        OrderRequest, OrderResponse,
    },
};

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
    let consumer = MatchingServiceImpl::new(markets, kafka_brokers, kafka_topic, wal_dir);
    // register_server(consumer);
    let mut registry_map = HashMap::<String, RegistryConfig>::new();
    registry_map.insert(
        "zookeeper".to_string(),
        RegistryConfig {
            protocol: "zookeeper".to_string(),
            address: "127.0.0.1:2181".to_string(),
        },
    );
    let protocal = Protocol {
        name: "dubbo".to_string(),
        port: 20086.to_string(),
        ..Default::default()
    };
    let mut protocol_config = ProtocolConfig::default();
    protocol_config.insert("dubbo".to_string(), protocal);
    let mut service_map = HashMap::new();
    service_map.insert(
        "MatchingService".to_string(),
        ServiceConfig {
            interface: "com.metawebthree.common.rpc.interfaces.MatchingService".to_string(),
            version: "".to_string(),
            tag: "".to_string(),
            group: "/dev/metawebthree".to_string(),
            protocol: "dubbo".to_string(),
        },
    );
    let mut root_config = RootConfig::new();
    root_config.registries = registry_map;
    root_config.protocols = protocol_config;
    root_config.provider = ProviderConfig::new().with_services(service_map);
    // @TDOO fix dubbo api error
    Dubbo::new()
        .with_config(match root_config.load() {
            Ok(config) => config,
            Err(_err) => panic!("err: {:?}", _err), // response was droped
        })
        .start();
}

pub async fn start() {
    let markets = vec![
        "BTC/USDT".to_string(),
        "ETH/USDT".to_string(),
        "DOGE/USDT".to_string(),
    ];
    start_rpc(markets, "localhost:9092", "dex-trades", "/tmp");
}
