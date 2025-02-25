use crate::log;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use hmac::{Hmac, Mac};
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CryptoUtils;

#[wasm_bindgen]
impl CryptoUtils {
    
    pub fn twitter_signature(method: &str, url: &str, key: &str, oauth_callback: &str) -> String {
        let consumer_secret = key.clone();
        let mut parameters =
            Self::get_base_oauth1_map(key, Option::Some((js_sys::Date::now().round() / 1000.0) as u64));
        parameters.insert("oauth_callback", oauth_callback.to_string());
        let signature = Self::generate(method, url, &mut parameters, consumer_secret);
        parameters.insert("signature", signature);
        Self::hashmap_to_query_string(&parameters)
    }

    pub fn twitter_signature2(method: &str, url: &str, key: &str, oauth_callback: &str) -> JsValue {
        let consumer_secret = key.clone();
        let mut parameters =
            Self::get_base_oauth1_map(key, Option::Some((js_sys::Date::now().round() / 1000.0) as u64));
        parameters.insert("oauth_callback", oauth_callback.to_string());
        let signature = Self::generate(method, url, parameters.borrow_mut(), consumer_secret);
        parameters.insert("signature", signature);
        let json = serde_json::to_string(&parameters).unwrap();
        JsValue::from(json)
    }

    fn get_base_oauth1_map(key: &str, timestamp: Option<u64>) -> HashMap<&str, String> {
        let mut parameters = HashMap::new();
        parameters.insert("oauth_consumer_key", key.to_string());
        parameters.insert("oauth_signature_method", "HMAC-SHA1".to_string());
        parameters.insert(
            "oauth_timestamp",
            timestamp
                .or_else(|| {
                    Option::Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    )
                })
                .unwrap()
                .to_string(),
        );
        parameters.insert("oauth_nonce", Uuid::new_v4().to_string());
        parameters.insert("oauth_version", "1.0".to_string());
        parameters
    }

    fn generate(
        method: &str,
        url: &str,
        parameters: &mut HashMap<&str, String>,
        consumer_secret: &str,
    ) -> String {
        let mut sorted_params: Vec<_> = parameters.iter().collect();
        sorted_params.sort_by(|a, b| a.0.cmp(b.0));
        let base_string = Self::generate_base_string(method, url, &sorted_params);
        log(format!("base_string:{}", base_string).as_str());
        Self::hmac_sha1(base_string.as_bytes(), consumer_secret.as_bytes())
    }

    fn generate_base_string(
        method: &str,
        url: &str,
        sorted_params: &Vec<(&&str, &String)>,
    ) -> String {
        let params: Vec<String> = sorted_params
            .iter()
            .map(|(key, value)| format!("{}={}", key, Self::encode(value)))
            .collect();
        let params_str = params.join("&");
        format!(
            "{}&{}&{}",
            method.to_uppercase(),
            Self::encode(url),
            Self::encode(params_str.as_ref())
        )
    }

    pub fn hmac_sha1(data: &[u8], key: &[u8]) -> String {
        let mut mac = Hmac::<Sha1>::new_from_slice(key).expect("HMAC can take key of any size");
        mac.update(data);
        BASE64.encode(mac.finalize().into_bytes())
    }

    pub fn encode(value: &str) -> String {
        utf8_percent_encode(value, ENCODE_URI_FRAGMENT).to_string()
    }

    fn hashmap_to_query_string(parameters: &HashMap<&str, String>) -> String {
        parameters
            .iter()
            .map(|(key, value)| format!("{}={}", Self::encode(key), Self::encode(value)))
            .collect::<Vec<String>>()
            .join("&")
    }

    #[wasm_bindgen]
    pub fn encrypt_common_request(input: &str) -> js_sys::Array {
        let timestamp = js_sys::Date::now() as i64;
        let input_with_timestamp = format!("{}{}", input, timestamp);
        let bytes = input_with_timestamp.as_bytes();
        let base64_encoded = BASE64.encode(bytes);
        let mut hasher = Sha256::new();
        hasher.update(base64_encoded.as_bytes());
        let encrypted = format!("{:x}", hasher.finalize());
        let result = js_sys::Array::new();
        result.push(&JsValue::from_str(&encrypted));
        result.push(&JsValue::from_f64(timestamp as f64));
        result
    }

    #[wasm_bindgen]
    pub fn decrypt_common_request(encrypted: &str, original: &str, timestamp: f64) -> bool {
        let input_with_timestamp = format!("{}{}", original, timestamp as i64);
        let bytes = input_with_timestamp.as_bytes();
        let base64_encoded = BASE64.encode(bytes);

        let mut hasher = Sha256::new();
        hasher.update(base64_encoded.as_bytes());
        let verify_hash = format!("{:x}", hasher.finalize());

        encrypted == verify_hash
    }
}

// 定义基于 NON_ALPHANUMERIC，但排除了点号(.)和其他不想编码的字符
const ENCODE_URI_FRAGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'<')
    .add(b'>')
    .add(b'`') // 这些字符是非字母数字但在 URI 组件中通常不被编码
    .add(b'#')
    .add(b'?')
    .add(b'{')
    .add(b'}')
    .add(b';')
    .add(b'/')
    .add(b':')
    .add(b'@')
    .add(b'=')
    .add(b'&')
    .add(b'$')
    .add(b',')
    .add(b'[')
    .add(b']')
    .add(b'+')
    .add(b'!')
    .add(b'*')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'%');

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        assert_eq!(CryptoUtils::encode("1.0"), "1.0");
        assert_eq!(CryptoUtils::encode("HMAC-SHA1"), "HMAC-SHA1");
        assert_eq!(
            CryptoUtils::encode("http://tsp.nat300.top/airdrop/bindAccount"),
            "http%3A%2F%2Ftsp.nat300.top%2Fairdrop%2FbindAccount"
        );
    }

    #[test]
    fn test_generate() {
        let key = "OxhqcUXNEaQUtMMreqvRdYl38";
        let oauth_callback = "http://tsp.nat300.top/airdrop/bindAccount";
        let mut parameters = CryptoUtils::get_base_oauth1_map(key, Option::None);
        parameters.insert("oauth_callback", String::from(oauth_callback));
        let method = "POST";
        let url = "https://api.twitter.com/oauth/request_token";
        let consumer_secret = key.clone();

        let signature = CryptoUtils::generate(method, url, &mut parameters, consumer_secret);
        // assert_eq!(signature, "vLIA5PXGId2uCyDXYyp90hkg9G8=");
        parameters.insert("signature", signature);
        let full_query_str = CryptoUtils::hashmap_to_query_string(&parameters);
        println!("{}", full_query_str.clone());
        assert_eq!(full_query_str.starts_with(url), false)
    }

    #[test]
    fn test_crypto_utils() {
        let input = "Hello, World!";
        let result = CryptoUtils::encrypt_common_request(input);
        let encrypted = result.get(0).as_string().unwrap();
        let timestamp = result.get(1).as_f64().unwrap();
        assert!(CryptoUtils::decrypt_common_request(
            &encrypted, input, timestamp
        ));
        assert!(!CryptoUtils::decrypt_common_request(
            &encrypted,
            "Wrong input",
            timestamp
        ));
        assert!(!CryptoUtils::decrypt_common_request(
            &encrypted,
            input,
            timestamp + 1.0
        ));
    }
}
