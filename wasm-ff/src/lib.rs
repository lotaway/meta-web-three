mod utils;
mod tetris;
mod service;
mod excel;

extern crate wasm_bindgen;

use wasm_bindgen::prelude::*;
use std::collections::HashSet;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};
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

pub fn game() {
    let mut pos_set = HashSet::<Vec<usize>>::new();
    pos_set.insert(vec! {0, 0});
    let mut shape = Shape { positions: pos_set };
    let start_pos = shape.positions.iter().find(|e| true).unwrap().clone();
    let tetris = Tetris::new(start_pos);
}

pub async fn get_chains(page_index: i32, page_size: i32) -> Result<JsValue, JsValue> {
    let mut host = String::from("https://www.bitsat.ai");
    let mut options = RequestInit::new();
    options.set_method("GET");
    let api_url = format!("{}/bitsat/api/bridge/chain/list?pageIndex={}&pageSize={}", host, page_index, page_size);
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