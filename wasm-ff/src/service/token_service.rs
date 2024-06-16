use std::future;
use wasm_bindgen::JsValue;

pub struct TokenService {
    host: String,
    prev_fix: String,
    version: i8,
}

impl TokenService {
    fn new() -> Self {
        TokenService {
            host: String::from("https://mempool.space"),
            prev_fix: String::from("/testnet/api/"),
            version: 1,
        }
    }
    
    fn resolve_url(&self, url: &str) -> String {
        return format!("{}{}{}{}", self.host, self.prev_fix, self.version, url)
    }

    fn get_fee(&self) -> JsValue {
        future::Future::new(window::fetch(self.resolve_url("/fees/recommended")))
    }
}