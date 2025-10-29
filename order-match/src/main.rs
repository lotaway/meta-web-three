mod config;
mod order_match;
mod generated;

#[tokio::main]
async fn main() {
    println!("Order Book Program is running!");
    order_match::order_match::start().await;
}
