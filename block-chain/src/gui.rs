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

    pub fn add_peer(&self, addr: SocketAddr) {
        if let Ok(mut count) = self.peer_count.lock() {
            *count += 1;
        }
        if let Ok(mut addrs) = self.peer_addresses.lock() {
            addrs.push(addr);
        }
    }

    pub fn get_peer_count(&self) -> usize {
        self.peer_count.lock().map(|c| *c).unwrap_or(0)
    }

    pub fn get_peer_addresses(&self) -> Vec<SocketAddr> {
        self.peer_addresses.lock().map(|a| a.clone()).unwrap_or_default()
    }
}

fn show_node_stats(ui: &mut egui::Ui, peer_count: usize) {
    ui.heading("📊 Node Statistics");
    ui.separator();
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label("Connected Nodes:");
            ui.heading(format!("{}", peer_count));
        });
    });
    ui.add_space(10.0);
}

fn show_address_list(ui: &mut egui::Ui, addresses: &[SocketAddr]) {
    ui.heading("🌐 Node Addresses");
    ui.separator();
    egui::ScrollArea::vertical()
        .max_height(200.0)
        .show(ui, |ui| {
            for (index, addr) in addresses.iter().enumerate() {
                ui.label(format!("{}. {}", index + 1, addr));
            }
        });
    ui.add_space(10.0);
}

fn show_status(ui: &mut egui::Ui, running: bool) {
    ui.horizontal(|ui| {
        let color = if running { egui::Color32::GREEN } else { egui::Color32::YELLOW };
        ui.colored_label(color, "●");
        ui.label(if running { "Running" } else { "Paused" });
    });
}

fn show_controls(ui: &mut egui::Ui, running: &mut bool) {
    ui.heading("🎮 Controls");
    ui.separator();
    if ui.button(if *running { "⏸ Pause" } else { "▶ Resume" }).clicked() {
        *running = !*running;
    }
    ui.add_space(10.0);
}

fn show_system_info(ui: &mut egui::Ui) {
    ui.heading("ℹ️ System Info");
    ui.separator();
    ui.label("egui Version: 0.29");
    ui.label("Refresh Rate: 60 FPS");
    ui.add_space(10.0);
}

fn show_dashboard(ui: &mut egui::Ui, node_state: &NodeState, running: &mut bool) {
    ui.heading("⛓️ Block Chain Node Monitor");
    ui.add_space(10.0);
    show_node_stats(ui, node_state.get_peer_count());
    ui.columns(2, |columns| {
        show_address_list(&mut columns[0], &node_state.get_peer_addresses());
        let right = &mut columns[1];
        show_system_info(right);
        show_controls(right, running);
        show_status(right, *running);
    });
}

pub struct NodeGuiApp {
    node_state: NodeState,
    running: bool,
}

impl NodeGuiApp {
    pub fn new(node_state: NodeState) -> Self {
        Self {
            node_state,
            running: true,
        }
    }
}

impl eframe::App for NodeGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        egui::CentralPanel::default().show(ctx, |ui| {
            show_dashboard(ui, &self.node_state, &mut self.running);
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

