mod order_match;

#[tokio::main]
fn main() {
    println!("Order Book Program is running!");
    start().await;
}
