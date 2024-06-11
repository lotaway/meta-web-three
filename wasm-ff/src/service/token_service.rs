struct TokenService {
    host: &str,
    prev_fix: &str,
}

impl TokenService {
    fn new() -> Self {
        TokenService {
            host: "https://mempoolspace.com",
            prev_fix: "/test/api",
        }
    }

    fn getFee(&self) -> JsValue::Promise<JsValue::BigUint> {
        Tokio::Future::new(window::fetch(self.host + self.prev_fix + "/feeInfo"))
    }
}