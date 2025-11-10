use crate::config::AppConfig;

mod config;
mod order_match;
mod generated;

#[tokio::main]
async fn main() {
    println!("Order Book Program is running!");
    match AppConfig::load() {
        Ok(config) => {
            println!("Config loaded: {:#?}", config);
        }
        Err(e) => {
            println!("COnfig loading failed: {:#?}", e);
        }
    }
    order_match::order_match::start().await;
}
