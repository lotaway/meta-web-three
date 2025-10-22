use anyhow::Result;
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub app: AppSettings,
    pub kafka: KafkaSettings,
    pub dubbo: DubboSettings,
    pub storage: StorageSettings,
    pub markets: MarketsSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppSettings {
    pub name: String,
    pub log_level: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KafkaSettings {
    pub brokers: String,
    pub topic: String,
    pub producer_timeout_ms: u64,
    pub producer_retry_count: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DubboSettings {
    pub port: u16,
    pub registry_address: String,
    pub group: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StorageSettings {
    pub wal_dir: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MarketsSettings {
    pub markets: Vec<String>,
}

impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok();

        let config = Config::builder()
            .add_source(File::with_name("config").required(false))
            .add_source(
                Environment::with_prefix("ORDER_MATCH")
                    .prefix_separator("_")
                    .separator("_")
            )
            .add_source(
                Environment::default()
                    .try_parsing(true)
                    .separator("_")
            )
            .build()?;

        config.try_deserialize()
    }

    pub fn load_with_env_prefix(prefix: &str) -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok();

        let config = Config::builder()
            .add_source(File::with_name("config").required(false))
            .add_source(
                Environment::with_prefix(prefix)
                    .prefix_separator("_")
                    .separator("_")
            )
            .build()?;

        config.try_deserialize()
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app: AppSettings {
                name: "order-match".to_string(),
                log_level: "info".to_string(),
            },
            kafka: KafkaSettings {
                brokers: "localhost:9092".to_string(),
                topic: "dex-trades".to_string(),
                producer_timeout_ms: 5000,
                producer_retry_count: 3,
            },
            dubbo: DubboSettings {
                port: 20086,
                registry_address: "127.0.0.1:2181".to_string(),
                group: "/dev/metawebthree".to_string(),
                version: "".to_string(),
            },
            storage: StorageSettings {
                wal_dir: "/tmp".to_string(),
            },
            markets: MarketsSettings {
                markets: vec![
                    "BTC/USDT".to_string(),
                    "ETH/USDT".to_string(),
                    "DOGE/USDT".to_string(),
                ],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AppConfig::default();
        assert_eq!(config.app.name, "order-match");
        assert_eq!(config.kafka.brokers, "localhost:9092");
        assert_eq!(config.dubbo.port, 20086);
    }

    #[test]
    fn test_config_load() {
        if std::path::Path::new("config.toml").exists() {
            let config = AppConfig::load();
            assert!(config.is_ok());
        }
    }
}
