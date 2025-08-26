mod order_match;

#[tokio::main]
async fn main() {
    println!("Order Book Program is running!");
    order_match::order_match::start().await;
}
