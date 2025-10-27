pub mod interfaces {
    use dubbo::{codegen::async_trait, status::DubboError};

    use crate::order_match::structs::structs::{OrderRequest, OrderResponse};


    // #[dubbo::service]
    #[async_trait]
    pub trait OrderMatchService {
        async fn match_order(&self, req: OrderRequest) -> Result<OrderResponse, DubboError>;
    }
}
