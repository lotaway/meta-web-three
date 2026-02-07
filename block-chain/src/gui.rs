use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct NodeState {
    pub peer_count: Arc<Mutex<usize>>,
    pub peer_addresses: Arc<Mutex<Vec<SocketAddr>>>,
}

#[derive(Clone)]
enum Page {
    Monitor,
    NodeList,
    NodeDetail(SocketAddr),
}

impl NodeState {
    pub fn new() -> Self {
        Self {
            peer_count: Arc::new(Mutex::new(1)),
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
        self.peer_addresses
            .lock()
            .map(|a| a.clone())
            .unwrap_or_default()
    }
}

fn show_status(ui: &mut egui::Ui, running: bool) {
    ui.horizontal(|ui| {
        let color = if running {
            egui::Color32::GREEN
        } else {
            egui::Color32::YELLOW
        };
        ui.colored_label(color, "●");
        ui.label(if running { "Running" } else { "Paused" });
    });
}

fn show_controls(ui: &mut egui::Ui, running: &mut bool) {
    ui.heading("🎮 Controls");
    ui.separator();
    if ui
        .button(if *running { "⏸ Pause" } else { "▶ Resume" })
        .clicked()
    {
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

fn show_monitor(ui: &mut egui::Ui, node_state: &NodeState, running: &mut bool) {
    ui.heading("⛓️ Block Chain Node Monitor");
    ui.add_space(10.0);
    ui.heading("📊 Node Statistics");
    ui.separator();
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label("Connected Nodes:");
            ui.heading(format!("{}", node_state.get_peer_count()));
        });
    });
    ui.add_space(10.0);

    ui.columns(2, |columns| {
        let right = &mut columns[1];
        show_system_info(right);
        show_controls(right, running);
        show_status(right, *running);
    });
}

fn show_node_detail(ui: &mut egui::Ui, addr: &SocketAddr) {
    ui.heading("🔍 Node Detail");
    ui.add_space(10.0);
    ui.separator();
    ui.label("Address:");
    ui.heading(format!("{}", addr));
    ui.add_space(10.0);
    ui.label("Status: Connected");
    ui.label("Latency: 12ms");
    ui.label("Last Seen: Just now");
}

pub struct NodeGuiApp {
    node_state: NodeState,
    running: bool,
    stack: Vec<Page>,
    anim: f32,
}

impl NodeGuiApp {
    pub fn new(node_state: NodeState) -> Self {
        Self {
            node_state,
            running: true,
            stack: vec![Page::Monitor],
            anim: 0.0,
        }
    }
}

impl eframe::App for NodeGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        let target = (self.stack.len() as f32) - 1.0;
        self.anim = ctx.animate_value_with_time(egui::Id::new("nav"), target, 0.25);

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.stack.len() > 1 {
                ui.horizontal(|ui| {
                    if ui.button("← Back").clicked() {
                        self.stack.pop();
                    }
                });
                ui.add_space(10.0);
            }

            ui.horizontal(|ui| {
                if ui
                    .selectable_label(
                        matches!(self.stack.last(), Some(Page::Monitor)),
                        "📊 Monitor",
                    )
                    .clicked()
                {
                    if !matches!(self.stack.last(), Some(Page::Monitor)) {
                        self.stack.push(Page::Monitor);
                    }
                }
                if ui
                    .selectable_label(
                        matches!(self.stack.last(), Some(Page::NodeList)),
                        "🌐 Nodes",
                    )
                    .clicked()
                {
                    if !matches!(self.stack.last(), Some(Page::NodeList)) {
                        self.stack.push(Page::NodeList);
                    }
                }
            });

            ui.separator();

            let rect = ui.available_rect_before_wrap();
            let width = rect.width();

            for (i, page) in self.stack.iter().enumerate() {
                let offset = (i as f32 - self.anim) * width;
                ui.allocate_ui_at_rect(rect.translate(egui::vec2(offset, 0.0)), |ui| match page {
                    Page::Monitor => show_monitor(ui, &self.node_state, &mut self.running),
                    Page::NodeList => {
                        ui.heading("🌐 Node List");
                        ui.add_space(10.0);
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.label("Total Nodes:");
                            ui.heading(format!("{}", self.node_state.get_peer_count()));
                            ui.add_space(10.0);
                            ui.separator();
                            for addr in self.node_state.get_peer_addresses() {
                                ui.horizontal(|ui| {
                                    if ui.button("📍").clicked() {
                                        self.stack.push(Page::NodeDetail(addr));
                                    }
                                    ui.label(format!("{}", addr));
                                });
                            }
                        });
                    }
                    Page::NodeDetail(addr) => show_node_detail(ui, addr),
                });
            }
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

impl NodeState {
    pub fn new() -> Self {
        Self {
            peer_count: Arc::new(Mutex::new(1)),
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
        self.peer_addresses
            .lock()
            .map(|a| a.clone())
            .unwrap_or_default()
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
        let color = if running {
            egui::Color32::GREEN
        } else {
            egui::Color32::YELLOW
        };
        ui.colored_label(color, "●");
        ui.label(if running { "Running" } else { "Paused" });
    });
}

fn show_controls(ui: &mut egui::Ui, running: &mut bool) {
    ui.heading("🎮 Controls");
    ui.separator();
    if ui
        .button(if *running { "⏸ Pause" } else { "▶ Resume" })
        .clicked()
    {
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

pub struct NodeGuiApp {
    node_state: NodeState,
    running: bool,
    tab: usize,
    anim: f32,
}

impl NodeGuiApp {
    pub fn new(node_state: NodeState) -> Self {
        Self {
            node_state,
            running: true,
            tab: 0,
            anim: 0.0,
        }
    }
}

impl eframe::App for NodeGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        let target = self.tab as f32;
        self.anim = ctx.animate_value_with_time(egui::Id::new("tab"), target, 0.25);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.tab == 0, "📊 Monitor").clicked() {
                    self.tab = 0;
                }
                if ui.selectable_label(self.tab == 1, "🌐 Nodes").clicked() {
                    self.tab = 1;
                }
            });

            ui.separator();

            let rect = ui.available_rect_before_wrap();
            let width = rect.width();

            for (i, show_page) in [
                |ui| show_monitor(ui, &self.node_state, &mut self.running),
                |ui| show_nodes(ui, &self.node_state),
            ]
            .iter()
            .enumerate()
            {
                let offset = (i as f32 - self.anim) * width;
                ui.allocate_ui_at_rect(rect.translate(egui::vec2(offset, 0.0)), |ui| show_page(ui));
            }
        });
    }
}

fn show_monitor(ui: &mut egui::Ui, node_state: &NodeState, running: &mut bool) {
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

fn show_nodes(ui: &mut egui::Ui, node_state: &NodeState) {
    ui.heading("🌐 Node List");
    ui.add_space(10.0);
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Total Nodes:");
        ui.heading(format!("{}", node_state.get_peer_count()));
        ui.add_space(10.0);
        show_address_list(ui, &node_state.get_peer_addresses());
    });
}

pub fn run_gui(node_state: NodeState) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Block Chain Node Monitor",
        options,
        Box::new(|_cc| Ok(Box::new(NodeGuiApp::new(node_state)))),
    )
}
