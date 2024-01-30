use hmac::{Hmac, Mac};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use sha1::Sha1;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::{Uuid};
use crate::log;

pub fn get_base_oauth1_map(key: &str, timestamp: Option<u64>) -> HashMap<String, String> {
    let mut parameters = HashMap::new();
    parameters.insert("oauth_consumer_key".to_string(), key.to_string());
    parameters.insert(
        "oauth_signature_method".to_string(),
        "HMAC-SHA1".to_string(),
    );
    parameters.insert(
        "oauth_timestamp".to_string(),
        timestamp
            .or_else(|| {
                Option::Some(SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs())
            })
            .unwrap()
            .to_string(),
    );
    parameters.insert("oauth_nonce".to_string(), Uuid::new_v4().to_string());
    parameters.insert("oauth_version".to_string(), "1.0".to_string());
    parameters
}

pub fn generate(
    method: &str,
    url: &str,
    parameters: &mut HashMap<String, String>,
    consumer_secret: &str,
) -> String {
    let mut sorted_params: Vec<_> = parameters.iter().collect();
    sorted_params.sort_by(|a, b| a.0.cmp(b.0));
    let base_string = generate_base_string(method, url, &sorted_params);
    hmac_sha1(base_string.as_bytes(), consumer_secret.as_bytes())
}

pub fn generate_base_string(
    method: &str,
    url: &str,
    sorted_params: &Vec<(&String, &String)>,
) -> String {
    let params: Vec<String> = sorted_params
        .iter()
        .map(|(key, value)| format!("{}={}", key, encode(value)))
        .collect();
    let params_str = params.join("&");
    format!(
        "{}&{}&{}",
        method.to_uppercase(),
        encode(url),
        encode(&params_str)
    )
}

pub fn hmac_sha1(data: &[u8], key: &[u8]) -> String {
    let mut mac = Hmac::<Sha1>::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    base64::encode(mac.finalize().into_bytes())
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

pub fn encode(value: &str) -> String {
    utf8_percent_encode(value, ENCODE_URI_FRAGMENT).to_string()
}

pub fn hashmap_to_query_string(parameters: &HashMap<String, String>) -> String {
    parameters.iter()
        .map(|(key, value)| format!("{}={}", encode(key), encode(value)))
        .collect::<Vec<String>>()
        .join("&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        assert_eq!(encode("1.0"), "1.0");
        assert_eq!(encode("HMAC-SHA1"), "HMAC-SHA1");
        assert_eq!(encode("http://tsp.nat300.top/airdrop/bindAccount"), "http%3A%2F%2Ftsp.nat300.top%2Fairdrop%2FbindAccount");
    }

    #[test]
    fn test_generate() {
        let key = "OxhqcUXNEaQUtMMreqvRdYl38";
        let oauth_callback = "http://tsp.nat300.top/airdrop/bindAccount";
        let mut parameters = get_base_oauth1_map(key, Option::None);
        parameters.insert(String::from("oauth_callback"), String::from(oauth_callback));
        let method = "POST";
        let url = "https://api.twitter.com/oauth/request_token";
        let consumer_secret = key.clone();

        let signature = generate(method, url, &mut parameters, consumer_secret);
        // assert_eq!(signature, "vLIA5PXGId2uCyDXYyp90hkg9G8=");
        parameters.insert(String::from("signature"), signature);
        let full_query_str = hashmap_to_query_string(&parameters);
        println!("{}", full_query_str.clone());
        assert_eq!(full_query_str.starts_with(url), false)
    }
}
