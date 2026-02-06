use eframe::egui;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct NodeState {
    pub peer_count: Arc<Mutex<usize>>,
    pub peer_addresses: Arc<Mutex<Vec<SocketAddr>>>,
}

impl NodeState {
    pub fn new() -> Self {
        Self {
            peer_count: Arc::new(Mutex::new(0)),
            peer_addresses: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn set_peer_count(&self, count: usize) {
        let mut guard = self.peer_count.lock().unwrap();
        *guard = count;
    }

    pub fn set_peer_addresses(&self, addresses: Vec<SocketAddr>) {
        let mut guard = self.peer_addresses.lock().unwrap();
        *guard = addresses;
    }

    pub fn add_peer(&self, addr: SocketAddr) {
        let mut count_guard = self.peer_count.lock().unwrap();
        let mut addrs_guard = self.peer_addresses.lock().unwrap();
        *count_guard += 1;
        addrs_guard.push(addr);
    }

    pub fn get_peer_count(&self) -> usize {
        let guard = self.peer_count.lock().unwrap();
        *guard
    }

    pub fn get_peer_addresses(&self) -> Vec<SocketAddr> {
        let guard = self.peer_addresses.lock().unwrap();
        guard.clone()
    }
}

pub struct NodeGuiApp {
    node_state: NodeState,
}

impl NodeGuiApp {
    pub fn new(node_state: NodeState) -> Self {
        Self { node_state }
    }
}

impl eframe::App for NodeGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Block Chain Node Monitor");
            ui.separator();

            let peer_count = self.node_state.get_peer_count();
            ui.label(format!("Connected Nodes: {}", peer_count));

            ui.separator();
            ui.heading("Node Addresses:");

            let addresses = self.node_state.get_peer_addresses();
            for (index, addr) in addresses.iter().enumerate() {
                ui.label(format!("{}. {}", index + 1, addr));
            }

            // 刷新界面
            ui.separator();
            ui.label("Auto-refresh every frame");
        });
    }
}

pub fn run_gui(node_state: NodeState) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Block Chain Node Monitor",
        options,
        Box::new(|_cc| Ok(Box::new(NodeGuiApp::new(node_state)))),
    )
}
