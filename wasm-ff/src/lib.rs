mod oauth_utils;

extern crate wasm_bindgen;

use wasm_bindgen::prelude::*;
use crate::oauth_utils::{generate, get_base_oauth1_map, hashmap_to_query_string};
use std::borrow::BorrowMut;
use js_sys::Date;

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn date_now() -> u64;
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}

#[wasm_bindgen]
pub fn twitter_signature(
    method: &str,
    url: &str,
    key: &str,
    oauth_callback: &str,
) -> String {
    let consumer_secret = key.clone();
    let mut parameters = get_base_oauth1_map(key, Option::Some((Date::now().round() / 1000.0) as u64));
    parameters.insert(String::from("oauth_callback"), oauth_callback.to_string());
    let signature = generate(
        method,
        url,
        parameters.borrow_mut(),
        consumer_secret,
    );
    parameters.insert(String::from("signature"), signature);
    hashmap_to_query_string(&parameters)
}

#[wasm_bindgen]
pub fn twitter_signature2(
    method: &str,
    url: &str,
    key: &str,
    oauth_callback: &str,
) -> JsValue {
    let consumer_secret = key.clone();
    let mut parameters = get_base_oauth1_map(key, Option::Some((Date::now().round() / 1000.0) as u64));
    parameters.insert("oauth_callback", oauth_callback.to_string());
    let signature = generate(
        method,
        url,
        parameters.borrow_mut(),
        consumer_secret,
    );
    parameters.insert("signature", signature);
    let json = serde_json::to_string(&parameters).unwrap();
    JsValue::from(json)
}

#[wasm_bindgen]
pub fn p2tr_transaction(network: &str) {
    let mut p2tr_transaction = P2trTransaction::new();
    p2tr_transaction.generate_address(Network::from_str(network).unwrap()).transaction()
}

#[wasm_bindgen]
pub fn psbt_generate(public_key: &str, pub_script: &str) -> Result<JsValue, JsValue> {
    let val = JsValue::from(public_key);
    Ok(val)
}

#[wasm_bindgen(start)]
pub async fn main() -> Result<(), JsValue> {
    log("Wasm main running");
    Ok(())
}