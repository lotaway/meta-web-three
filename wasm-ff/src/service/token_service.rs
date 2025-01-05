
use ethers::prelude::*;
use std::sync::Arc;
use tokio::time::{self, Duration};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TokenService {
    provider: Arc<Provider<Ws>>,
    token_contract_address: Address,
    token_contract_abi: Vec<u8>,
};

#[wasm_bindgen]
impl TokenService {

    pub fn new(rpc_url: &str, token_contract_address: &str, token_contract_abi: &str) -> Self {
        Self {
            provider: Arc::new(
                Provider::<Ws>::connect(rpc_url)
                .await
                .map_err(|e| JsValue::from_str(&format!("Failed to connect: {:?}", e)))?
            ),
            token_contract_address: token_contract_address.parse().unwrap(),
            token_contract_abi: include_bytes!(token_contract_abi),
        }
    }

    #[wasm_bindgen]
    pub async fn start_service(&self, listener: fn(from: Address, to: Address, value: U256)) -> Result<(), JsValue> {
    let client = Arc::new(self.provider.clone());
    let contract_abi = include_bytes!(self.token_contract_abi);
    let contract = Contract::from_json(client.clone(), self.token_contract_address, contract_abi)
        .map_err(|e| JsValue::from_str(&format!("Failed to load ABI: {:?}", e)))?;

    tokio::spawn(async move {
        let filter = contract.event::<(Address, U256)>("Transfer").unwrap().filter;
        let mut stream = client.subscribe_logs(&filter).await.unwrap();

        while let Some(log) = stream.next().await {
            if let Ok(decoded) = contract.decode_log::<(Address, Address, U256)>("Transfer", &log) {
                let (from, to, value) = decoded;
                log::info!("Transfer Event: from: {:?}, to: {:?}, value: {:?}", from, to, value);
                listener(from, to, value);
            }
        }
    });

    Ok(())
}

#[wasm_bindgen]
async fn get_holders(&self) {
    let logs = contract
        .event::<(Address, U256)>("Transfer")
        .unwrap()
        .query()
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to query logs: {:?}", e)))?;
    let holders: Vec<Address> = logs.into_iter().map(|log| log.0).collect();
    holders
}

#[wasm_bindgen]
async fn start_task(handler: fn(), delay: Option<u64>) {
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(delay.unwrap_or(60)));
        loop {
            interval.tick().await;
            handler();
        }
    })
}

#[wasm_bindgen]
fn get_balances(user_address: Vec<String>, _token_addresses: Vec<String>) {
    let client_clone = client.clone();
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(60));
        let token_addresses = _token_addresses
            .iter()
            .map(|addr| addr.parse::<Address>().unwrap())
            .collect::<Vec<_>>();

        loop {
            interval.tick().await;

            for holder in &user_address {
                for token in &token_addresses {
                    if let Ok(balance) = contract
                        .method::<_, U256>("balanceOf", (*holder,))
                        .unwrap()
                        .call()
                        .await
                    {
                        log::info!("Holder {:?} has {:?} tokens of {:?}", holder, balance, token);
                    }
                }
            }
        }
    });
}
}