use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc};
use block_chain::{Block, BlockChain};
use crate::async_task::{get_future_result, start_future_task, TFutureTask};
use tg_bot::run;

mod utils;
mod matcher;
mod single_linked_list;
mod async_task;

#[derive(Debug)]
struct LinkSummonProxy {
    block_chain: Option<Arc<Mutex<BlockChain>>>,
}

enum LinkSummonProxyError {
    InitFail
}

impl LinkSummonProxy {
    pub fn new() -> Self {
        Self {
            block_chain: Option::None
        }
    }

    pub async fn arc_new() -> Arc<Self> {
        Arc::new(Self::new())
    }

    pub fn start(&self) -> Self {
        let future = start_future_task(&self);
        // println!("{}", &self);
        get_future_result(future)
    }

    pub async fn set_online(&mut self) {
        // 创建一个HashMap来存储连接的节点
        let peers = Arc::new(Mutex::new(HashMap::new()));
        // 创建一个用于消息传递的通道
        let (tx, mut rx) = mpsc::channel::<Block>(32);
        tokio::task::spawn(async move {
            while let Some(block_from_other) = rx.recv().await {
                let mut block_chain = self.block_chain.as_mut().expect("Can't found block chain to lock").lock();
                block_chain.unwrap().add_block(block_from_other)
            }
        });
        // 启动一个任务来监听传入的连接
        let listener = tokio::net::TcpListener::bind("127.0.0.1:10100").await.unwrap();
        println!("Listening for incoming connections...");
        while let Result::Ok((socket, _)) = listener.accept().await {
            let peers_clone = Arc::clone(&peers);
            // 启动一个任务来处理每个连接
            tokio::task::spawn(async move {
                let (mut reader, mut writer) = socket.into_split();
                // 读取来自连接的区块数据
                let mut buffer = [0u8; 1024];
                let n = reader.read(&mut buffer).await.unwrap();
                let data_string = String::from_utf8_lossy(&buffer[..n]).to_string();
                let data_result = serde_json::from_str(&data_string);
                match data_result {
                    Result::Ok(block_data) => {
                        // 将区块添加到区块链中
                        tx.send(block_data.unwrap()).await.unwrap();
                        // 将连接的写端存储到peers中
                        let mut peers = peers_clone.lock().unwrap();
                        peers.insert(socket.peer_addr().unwrap(), writer);
                    }
                    Result::Err(e) => println!("Error in convert received data: ${e}"),
                }
            });
        }
        // 启动一个任务来处理区块链数据的同步
        let peers_clone = Arc::clone(&peers);
        tokio::task::spawn(async move {
            let block_chain_guard = self.block_chain.as_ref().expect("Can't found block chain to lock").lock().unwrap();
            let peers_guard = peers_clone.lock().unwrap();
            // 向每个连接的节点发送最新的区块链数据
            for mut writer in peers_guard.values() {
                for block in &block_chain_guard.blocks {
                    writer.write_all(serde_json::to_string(&block).unwrap().as_bytes()).await.unwrap();
                }
            }
        });
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
    }
}

impl TFutureTask<Result<&Arc<Mutex<BlockChain>>, LinkSummonProxyError>> for LinkSummonProxy {
    fn start(&mut self) -> Result<&Arc<Mutex<BlockChain>>, LinkSummonProxyError> {
        self.block_chain = Option::Some(Arc::new(Mutex::new(BlockChain::new())));
        let block_chain = self.block_chain.as_ref().expect("Can't found block chain to return");
        if block_chain.lock().unwrap().init().is_err() {
            return Err(LinkSummonProxyError::InitFail);
        }
        dbg!("{:?}", &self.block_chain);
        Ok(&block_chain)
    }
}

unsafe impl Send for LinkSummonProxy {}

unsafe impl Sync for LinkSummonProxy {}

#[tokio::main]
async fn main() {
    println!("Start main");
    let public_config_str = utils::get_config_file(".\\config\\public.json");
    dbg!("{}", public_config_str);
    let future = async_task::tokio_spawn();
    let mut block_chain_proxy = LinkSummonProxy::new();
    let result = block_chain_proxy.start();
    dbg!("{:?}", &result);
    if result.ok() {
        block_chain_proxy.set_online().await;
    }
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    println!("Start run bot");
    run().await;
    println!("End run bot");
    println!("End main");
}