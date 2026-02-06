use block_chain::{Block, BlockChain};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

mod async_task;
mod gui;
mod matcher;
mod utils;

#[derive(Debug, Clone)]
struct LinkSummonProxy {
    block_chain: Option<Arc<Mutex<BlockChain>>>,
    node_state: gui::NodeState,
}

impl LinkSummonProxy {
    pub fn new() -> Self {
        Self {
            block_chain: Option::None,
            node_state: gui::NodeState::new(),
        }
    }

    pub async fn start(&mut self) -> Result<(), String> {
        let block_chain = Arc::new(Mutex::new(BlockChain::new()));

        let block_chain_clone = block_chain.clone();
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                if let Err(e) = block_chain_clone.lock().await.init() {
                    eprintln!("Failed to initialize blockchain");
                }
            });
        })
        .await
        .map_err(|e| format!("Spawn error: {:?}", e))?;

        self.block_chain = Some(block_chain);
        Ok(())
    }

    pub async fn set_online(&mut self, node_state: gui::NodeState) {
        let block_chain = match &self.block_chain {
            Some(bc) => bc.clone(),
            None => return,
        };

        let peers: Arc<Mutex<HashMap<std::net::SocketAddr, tokio::net::tcp::OwnedWriteHalf>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let (tx, mut rx) = mpsc::channel::<Block>(32);

        let bc_for_rx = block_chain.clone();
        tokio::task::spawn(async move {
            while let Some(block_from_other) = rx.recv().await {
                let mut block_chain_guard = bc_for_rx.lock().await;
                let _ = block_chain_guard.add_block(block_from_other);
            }
        });

        let listener = tokio::net::TcpListener::bind("127.0.0.1:10100")
            .await
            .unwrap();
        println!("Listening for incoming connections...");

        let peers_clone = peers.clone();
        let node_state_clone = node_state.clone();
        let tx_clone = tx.clone();

        loop {
            if let Ok((socket, _)) = listener.accept().await {
                let peer_addr = match socket.peer_addr() {
                    Ok(addr) => addr,
                    Err(e) => {
                        eprintln!("Failed to get peer address: {:?}", e);
                        continue;
                    }
                };

                let peers_inner = peers_clone.clone();
                let node_state_inner = node_state_clone.clone();
                let tx_inner = tx_clone.clone();

                tokio::task::spawn(async move {
                    let (reader, writer) = socket.into_split();
                    let mut reader = reader;
                    let mut buffer = [0u8; 1024];
                    if let Ok(n) = reader.read(&mut buffer).await {
                        let data_string = String::from_utf8_lossy(&buffer[..n]).to_string();
                        if let Ok(block_data) = serde_json::from_str::<Option<Block>>(&data_string)
                        {
                            if let Some(block) = block_data {
                                let _ = tx_inner.send(block).await;
                            }
                        }
                    }

                    let mut peers_guard = peers_inner.lock().await;
                    peers_guard.insert(peer_addr, writer);

                    node_state_inner.add_peer(peer_addr);

                    println!("New peer connected: {}", peer_addr);
                    println!("Total peers: {}", peers_guard.len());
                });
            }
        }
    }
}

use tokio::sync::mpsc;

fn main() {
    println!("Start main");
    let public_config_str = utils::utils::get_config_file("public.json");
    dbg!("{}", public_config_str);

    // 创建共享的节点状态
    let node_state = gui::NodeState::new();

    // 在单独线程运行区块链服务
    let node_state_for_blockchain = node_state.clone();
    thread::spawn(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut proxy = LinkSummonProxy::new();
            if proxy.start().await.is_ok() {
                println!("Blockchain initialized successfully");
                proxy.set_online(node_state_for_blockchain).await;
            } else {
                eprintln!("Failed to start blockchain");
            }
        });
    });

    // 在主线程运行 GUI（macOS 要求）
    println!("Starting GUI...");
    if let Err(e) = gui::run_gui(node_state) {
        eprintln!("GUI error: {:?}", e);
    }
}
