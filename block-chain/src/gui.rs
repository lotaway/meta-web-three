use eframe::egui;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

// ==================== 数据层 ====================

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
        let mut count_guard = self.peer_count.lock().unwrap();
        let mut addrs_guard = self.peer_addresses.lock().unwrap();
        *count_guard += 1;
        addrs_guard.push(addr);
    }

    pub fn get_peer_count(&self) -> usize {
        *self.peer_count.lock().unwrap()
    }

    pub fn get_peer_addresses(&self) -> Vec<SocketAddr> {
        self.peer_addresses.lock().unwrap().clone()
    }
}

// ==================== 组件层 ====================

/// 组件1: 节点统计卡片
fn show_node_stats(ui: &mut egui::Ui, peer_count: usize) {
    ui.heading("📊 Node Statistics");
    ui.separator();

    // 使用 Group 代替 Frame::card
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label("Connected Nodes:");
            ui.heading(format!("{}", peer_count));
        });
    });

    ui.add_space(10.0);
}

/// 组件2: 节点地址列表（带滚动条）
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

/// 组件3: 状态指示器
fn show_status(ui: &mut egui::Ui, status: &str, color: egui::Color32) {
    ui.horizontal(|ui| {
        ui.colored_label(color, "●");
        ui.label(status);
    });
}

/// 组件4: 按钮面板（返回交互事件）
fn show_controls(ui: &mut egui::Ui, running: &mut bool) -> bool {
    ui.heading("🎮 Controls");
    ui.separator();

    let clicked = ui
        .button(if *running { "⏸ Pause" } else { "▶ Resume" })
        .clicked();
    if clicked {
        *running = !*running;
    }

    ui.add_space(10.0);
    clicked
}

/// 组件5: 系统信息
fn show_system_info(ui: &mut egui::Ui) {
    ui.heading("ℹ️ System Info");
    ui.separator();

    ui.label("egui Version: 0.29");
    ui.label("Refresh Rate: 60 FPS");
    ui.label("Auto-refresh: Enabled");

    ui.add_space(10.0);
}

// ==================== 布局组合层 ====================

/// 左侧面板组合
fn show_left_panel(ui: &mut egui::Ui, peer_count: usize, addresses: &[SocketAddr]) {
    show_node_stats(ui, peer_count);
    show_address_list(ui, addresses);
}

/// 右侧面板组合  
fn show_right_panel(ui: &mut egui::Ui, running: &mut bool) {
    show_system_info(ui);
    let _ = show_controls(ui, running);
    show_status(
        ui,
        if *running { "Running" } else { "Paused" },
        if *running {
            egui::Color32::GREEN
        } else {
            egui::Color32::YELLOW
        },
    );
}

/// 分栏布局页面
fn show_dashboard(ui: &mut egui::Ui, node_state: &NodeState, running: &mut bool) {
    ui.heading("⛓️ Block Chain Node Monitor");
    ui.add_space(10.0);

    // 顶部统计
    show_node_stats(ui, node_state.get_peer_count());

    // 分栏布局 - 使用 ui.columns
    ui.columns(2, |columns| {
        // 左侧列
        show_address_list(&mut columns[0], &node_state.get_peer_addresses());

        // 右侧列
        show_right_panel(&mut columns[1], running);
    });
}

// ==================== 应用层 ====================

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
        // 请求每帧重绘（实时更新）
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
