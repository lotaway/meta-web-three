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
    sync::{Arc},
    env,
};

use crate::order_match::{
    interfaces::interfaces::MatchingService,
    order_shard::order_shard::ShardManager,
    structs::structs::{
        OrderRequest, OrderResponse,
    },
};
use crate::config::AppConfig;

pub struct MatchingServiceImpl {
    managers: HashMap<String, Arc<ShardManager>>,
}

impl MatchingServiceImpl {
    pub fn new(config: &AppConfig) -> Self {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka.brokers)
            .set("request.timeout.ms", config.kafka.producer_timeout_ms.to_string())
            .set("retries", config.kafka.producer_retry_count.to_string())
            .create()
            .expect("Failed to create Kafka producer");
        
        let mut managers = HashMap::new();
        for market in &config.markets.markets {
            let mgr = Arc::new(ShardManager::new(
                market,
                producer.clone(),
                &config.kafka.topic,
                &config.storage.wal_dir,
            ));
            mgr.start_with(1);
            managers.insert(market.clone(), mgr);
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

pub fn start_rpc(config: &AppConfig) {
    let _consumer = MatchingServiceImpl::new(config);
    // register_server(consumer);
    let mut registry_map = HashMap::<String, RegistryConfig>::new();
    registry_map.insert(
        "zookeeper".to_string(),
        RegistryConfig {
            protocol: "zookeeper".to_string(),
            address: config.dubbo.registry_address.clone(),
        },
    );
    let protocal = Protocol {
        name: "dubbo".to_string(),
        port: config.dubbo.port.to_string(),
        ..Default::default()
    };
    let mut protocol_config = ProtocolConfig::default();
    protocol_config.insert("dubbo".to_string(), protocal);
    let mut service_map = HashMap::new();
    service_map.insert(
        "MatchingService".to_string(),
        ServiceConfig {
            interface: "com.metawebthree.common.rpc.interfaces.MatchingService".to_string(),
            version: config.dubbo.version.clone(),
            tag: "".to_string(),
            group: config.dubbo.group.clone(),
            protocol: "dubbo".to_string(),
        },
    );
    let mut root_config = RootConfig::new();
    root_config.registries = registry_map;
    root_config.protocols = protocol_config;
    root_config.provider = ProviderConfig::new().with_services(service_map);
    // @TDOO fix dubbo api error
    let mut dubbo = Dubbo::new()
        .with_config(match root_config.load() {
            Ok(config) => config,
            Err(_err) => panic!("err: {:?}", _err), // response was droped
        });
    let _dubbo = dubbo.start();
}

pub async fn start() {
    // 加载配置
    let config = AppConfig::load().unwrap_or_else(|e| {
        eprintln!("Failed to load configuration: {}", e);
        eprintln!("Using default configuration");
        AppConfig::default()
    });

    // 设置日志级别
    unsafe {
        env::set_var("RUST_LOG", &config.app.log_level);
    }
    env_logger::init();

    log::info!("Starting {} with configuration:", config.app.name);
    log::info!("  Kafka brokers: {}", config.kafka.brokers);
    log::info!("  Kafka topic: {}", config.kafka.topic);
    log::info!("  Dubbo port: {}", config.dubbo.port);
    log::info!("  Markets: {:?}", config.markets.markets);

    start_rpc(&config);
}
