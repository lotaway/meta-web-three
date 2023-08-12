use std::sync::Arc;
use std::thread;
use std::time::Duration;
use block_chain::BlockChain;
use crate::async_task::{get_future_result, start_future_task, TFutureTask};

mod utils;
mod matcher;
mod single_linked_list;
mod async_task;

#[derive(Debug)]
struct LinkSummonProxy {}

enum LinkSummonProxyError {
    InitFail
}

impl LinkSummonProxy {
    pub fn new() -> Self {
        Self {}
    }

    pub fn arc_new() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl TFutureTask<Result<BlockChain, LinkSummonProxyError>> for LinkSummonProxy {
    fn start(&mut self) -> Result<BlockChain, LinkSummonProxyError> {
        let mut block_chain = BlockChain::new();
        if block_chain.init().is_err() {
            return Err(LinkSummonProxyError::InitFail);
        }
        dbg!("{:?}", &block_chain);
        Ok(block_chain)
    }
}

unsafe impl Send for LinkSummonProxy {}
unsafe impl Sync for LinkSummonProxy {}

// #[tokio::main]
fn main() {
    let public_config_str = utils::get_config_file(".\\config\\public.json");
    dbg!("{}", public_config_str);
    println!("Start main");
    let block_chain_proxy = LinkSummonProxy::new();
    let future = start_future_task(&block_chain_proxy);
    // println!("{}", block_chain_proxy);
    dbg!("{:?}", get_future_result(future));
    thread::sleep(Duration::from_secs(2));
    println!("End main");
}