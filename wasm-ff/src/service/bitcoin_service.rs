use std::future;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen_futures::JsFuture;
use web_sys::RequestMode;

#[wasm_bindgen]
pub struct UnisatService {
    host: String,
    prev_fix: String,
    version: i8,
}

impl UnisatService {
    fn new() -> Self {
        UnisatService {
            host: String::from("https://mempool.space"),
            prev_fix: String::from("/testnet/api"),
            version: 1,
        }
    }

    fn resolve_url(&self, url: &str, method: &str) -> String {
        if method != "POST" {
            return format!("{}{}{}", self.host, self.prev_fix, url);
        }
        format!("{}{}{}{}{}", self.host, self.prev_fix, "/v", self.version, url)
    }

    async fn fetch_api(&self, url: &str, method: &str, body: Option<JsValue>) -> Result<JsValue, JsValue> {
        let resolved_url = self.resolve_url(url, method);
        let mut opts = web_sys::RequestInit::new();
        opts.method(method);
        opts.mode(RequestMode::Cors);
        if let Some(body_data) = body {
            opts.body(Some(&body_data));
        }
        let request = web_sys::Request::new_with_str_and_init(&resolved_url, &opts)?;
        let window = web_sys::window().unwrap();
        JsFuture::from(window.fetch_with_request(&request)).await
    }

    pub async fn get_fee(&self) -> Result<JsValue, JsValue> {
        self.fetch_api("/fees/recommended", "GET", None).await
    }

    pub async fn send_tx(&self, tx_data: JsValue) -> Result<JsValue, JsValue> {
        self.fetch_api("/tx", "POST", Some(tx_data)).await
    }

    pub async fn get_block(&self, block_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/block/{}", block_id), "GET", None).await
    }

    pub async fn get_tx(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}", tx_id), "GET", None).await
    }

    pub async fn get_mempool(&self) -> Result<JsValue, JsValue> {
        self.fetch_api("/mempool", "GET", None).await
    }

    pub async fn get_address(&self, address: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/address/{}", address), "GET", None).await
    }

    pub async fn get_chain_head(&self) -> Result<JsValue, JsValue> {
        self.fetch_api("/blocks/tip/height", "GET", None).await
    }

    // 添加其他接口的方法
    pub async fn get_block_status(&self, block_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/block/{}/status", block_id), "GET", None).await
    }

    pub async fn get_block_transactions(&self, block_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/block/{}/txs", block_id), "GET", None).await
    }

    pub async fn get_block_transactions_with_index(&self, block_id: &str, tx_index: usize) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/block/{}/txs/{}", block_id, tx_index), "GET", None).await
    }

    pub async fn get_tx_status(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/status", tx_id), "GET", None).await
    }

    pub async fn get_tx_hex(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/hex", tx_id), "GET", None).await
    }

    pub async fn get_tx_raw(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/raw", tx_id), "GET", None).await
    }

    pub async fn get_tx_merkleblock_proof(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/merkleblock-proof", tx_id), "GET", None).await
    }

    pub async fn get_tx_merkle_proof(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/merkle-proof", tx_id), "GET", None).await
    }

    pub async fn get_mempool_recent(&self) -> Result<JsValue, JsValue> {
        self.fetch_api("/mempool/recent", "GET", None).await
    }

    pub async fn get_block_tip_hash(&self) -> Result<JsValue, JsValue> {
        self.fetch_api("/blocks/tip/hash", "GET", None).await
    }

    pub async fn get_blocks_at_height(&self, height: u32) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/block-height/{}", height), "GET", None).await
    }

    pub async fn get_outspend(&self, tx_id: &str, vout: usize) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/outspend/{}", tx_id, vout), "GET", None).await
    }

    pub async fn get_outspends(&self, tx_id: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/tx/{}/outspends", tx_id), "GET", None).await
    }

    pub async fn get_address_tx(&self, address: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/address/{}/txs", address), "GET", None).await
    }

    pub async fn get_address_tx_chain(&self, address: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/address/{}/txs/chain", address), "GET", None).await
    }

    pub async fn get_address_tx_mempool(&self, address: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/address/{}/txs/mempool", address), "GET", None).await
    }

    pub async fn get_scripthash(&self, scripthash: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/scripthash/{}/txs", scripthash), "GET", None).await
    }

    pub async fn get_scripthash_tx_chain(&self, scripthash: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/scripthash/{}/txs/chain", scripthash), "GET", None).await
    }

    pub async fn get_scripthash_tx_mempool(&self, scripthash: &str) -> Result<JsValue, JsValue> {
        self.fetch_api(&format!("/scripthash/{}/txs/mempool", scripthash), "GET", None).await
    }
}