mod oauth_utils;
mod tetris;
mod bitcoin_utils;

extern crate wasm_bindgen;

use wasm_bindgen::prelude::*;
use crate::oauth_utils::{generate, get_base_oauth1_map, hashmap_to_query_string};
use std::borrow::BorrowMut;
use std::collections::HashSet;
use js_sys::Date;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};
use crate::bitcoin_utils::{BitcoinNetwork, P2trTransaction};
use crate::tetris::{Shape, Tetris};

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
    parameters.insert("oauth_callback", oauth_callback.to_string());
    let signature = generate(
        method,
        url,
        parameters.borrow_mut(),
        consumer_secret,
    );
    parameters.insert("signature", signature);
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
    p2tr_transaction.generate_address(BitcoinNetwork::from_str(network).unwrap().as_network()).build_transaction("0")
}

#[wasm_bindgen]
pub fn psbt_generate(public_key: &str, pub_script: &str) -> Result<JsValue, JsValue> {
    let val = JsValue::from(public_key);
    Ok(val)
}

pub fn game() {
    let mut pos_set = HashSet::<Vec<usize>>::new();
    pos_set.insert(vec! {0, 0});
    let mut shape = Shape { positions: pos_set };
    let start_pos = shape.positions.take(1).unwrap();
    let tetris = Tetris::new(start_pos);
}

pub async fn get_chains(page_index: i32, page_size: i32) -> Result<JsValue, JsValue> {
    let mut host = String::from("https://www.bitsat.ai");
    let mut options = RequestInit::new();
    options.method("GET");
    let api_url = format!("{}/bitsat/api/bridge/chain/list?pageSize=100", host);
    let request = Request::new_with_str_and_init(api_url.as_ref(), options.as_ref())?;
    request.headers().set("Accept", "application/vnd.github.v3+json")?;
    let window = web_sys::window().unwrap();
    let response_val = JsFuture::from(window.fetch_with_request(&request)).await?;
    assert!(response_val.is_instance_of::<Response>());
    let response = response_val.dyn_into::<Response>()?;
    let json = JsFuture::from(response.json()?).await?;
    Ok(json)
}

#[wasm_bindgen(start)]
pub async fn main() -> Result<(), JsValue> {
    log("Wasm main running");
    Ok(())
}