use dubbo::{codegen::async_trait, config::{protocol::ProtocolConfig, registry::RegistryConfig, service::ServiceConfig}, status::DubboError};
use serde::{Deserialize, Serialize};

pub fn start_rpc() {
    let registry_config = RegistryConfig {
        protocol: "zookeeper".to_string(),
        address: "127.0.0.1:2181".to_string(),
        ..Default::default()
    };

    let protocol_config = ProtocolConfig {
        name: "dubbo".to_string(),
        port: 20086,
        ..Default::default()
    };
    // protocol_config.insert("registry", {
    //     name: "dubbo".to_string(),
    //     port: 20086,
    //     ..Default::default()
    // });

    let service_config = ServiceConfig {
        interface: "org.dex.matching.MatchingService".to_string(),
        protocol: "dubbo".to_string(),
        group: "/dev/metawebthree".to_string(),
        ..Default::default()
    };
    dubbo::init()
        .with_registry(registry_config)
        .with_protocol(protocol_config)
        .with_service(service_config)
        .serve();
}

// 使用与Java相同的hessian2序列化
pub struct OrderRequest {
    pub order_id: String,
    pub price: f64,
    // ...其他字段
}

#[derive(Serialize, Deserialize)]
pub struct OrderResponse {
    pub success: bool,
    // ...其他字段
}

// 实现Dubbo服务
#[derive(Default)]
pub struct MatchingServiceImpl;

trait MatchingService {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError>;
}

#[async_trait]
impl MatchingService for MatchingServiceImpl {
    async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError> {
        // 撮合引擎逻辑
    }
}