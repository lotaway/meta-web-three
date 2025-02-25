use ethers::{
    providers::{Provider, Http},
    types::{Address, U256, Bytes},
    contract::{Contract, MultiCall},
};
use std::sync::Arc;

pub struct ChainService {
    pub pre_fix: String,
    provider: Arc<Provider<Http>>,
    multicall: MultiCall<Provider<Http>>,
}

impl ChainService {
    pub async fn new(rpc_url: &str, _pre_fix: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let provider = Provider::<Http>::try_from(rpc_url)?;
        let provider = Arc::new(provider);
        
        // 创建multicall合约实例
        let multicall = MultiCall::new(
            provider.clone(),
            Some(Address::from_str("0xcA11bde05977b3631167028862bE2a173976CA11")?), // Multicall3地址
        )?;

        Ok(ChainService { 
            pre_fix: _pre_fix.unwrap_or("").to_string(),
            provider,
            multicall,
        })
    }

    pub async fn batch_call<T>(&self, calls: Vec<T>) -> Result<Vec<Bytes>, Box<dyn std::error::Error>> 
    where
        T: Into<Bytes> + Send + Sync,
    {
        let results = self.multicall
            .aggregate_calls(calls)
            .await?;

        Ok(results)
    }

    pub async fn batch_get_balances(&self, token: Address, addresses: Vec<Address>) -> Result<Vec<U256>, Box<dyn std::error::Error>> {
        let contract = Contract::new(
            token,
            include_bytes!("../../genereated/contract/ERC20_ABI.json"),
            self.provider.clone(),
        )?;

        let calls: Vec<_> = addresses
            .iter()
            .map(|&address| contract.method::<_, U256>("balanceOf", address))
            .collect::<Result<_, _>>()?;

        let results = self.multicall
            .aggregate_calls(calls)
            .await?;

        Ok(results)
    }
}

