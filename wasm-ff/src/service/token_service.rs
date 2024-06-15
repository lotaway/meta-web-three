pub struct TokenService {
    host: &str,
    prev_fix: &str,
    version: &str,
}

impl TokenService {
    fn new() -> Self {
        TokenService {
            host: "https://mempool.space",
            prev_fix: String::from("/testnet/api/{}", 1),
        }
    }

    fn get_fee(&self) -> JsValue::Promise<JsValue::BigUint> {
        Tokio::Future::new(window::fetch(self.host + self.prev_fix + "/fees/recommended"))
    }
}