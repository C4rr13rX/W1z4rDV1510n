//! W1z4rD Node — desktop dashboard.
//!
//! Shows node identity, service health, cluster membership, neural fabric
//! metrics, and system resources.  No chess, no training-specific content —
//! that belongs to the scripts that use the API.

use egui::{Color32, FontId, RichText, Ui, Vec2};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

// ── Shared state (written by background tasks, read by GUI thread) ─────────────

#[derive(Default, Clone)]
pub struct NodeDashboard {
    // Identity
    pub node_id:     String,
    pub cluster_id:  Option<String>,
    pub started_at:  Option<Instant>,

    // Services
    pub api_neuro_port:   u16,
    pub api_node_port:    u16,
    pub cluster_port:     u16,
    pub api_neuro_up:     bool,
    pub api_node_up:      bool,
    pub neuro_requests:   u64,
    pub neuro_errors:     u64,

    // Cluster
    pub cluster_role:   String,    // "Standalone" | "Coordinator" | "Worker"
    pub cluster_peers:  Vec<PeerInfo>,
    pub ring_slots:     usize,
    pub pending_otp:    Option<String>,

    // Neural fabric
    pub neuro_pools:       usize,
    pub neuro_labels:      usize,
    pub neuro_episodic:    usize,
    pub neuro_observations: u64,

    // Resources
    pub cpu_pct:       f32,
    pub ram_used_gb:   f32,
    pub ram_total_gb:  f32,

    // Log
    pub log_lines: VecDeque<String>,
}

#[derive(Default, Clone)]
pub struct PeerInfo {
    pub short_id:  String,
    pub addr:      String,
    pub cores:     u32,
    pub os:        String,
    pub is_coord:  bool,
}

impl NodeDashboard {
    pub fn push_log(&mut self, line: impl Into<String>) {
        self.log_lines.push_back(line.into());
        while self.log_lines.len() > 200 {
            self.log_lines.pop_front();
        }
    }

    fn uptime(&self) -> String {
        let Some(start) = self.started_at else { return "—".into() };
        let s = start.elapsed().as_secs();
        if s < 60      { format!("{s}s") }
        else if s < 3600 { format!("{}m {}s", s/60, s%60) }
        else           { format!("{}h {}m", s/3600, (s%3600)/60) }
    }
}

// ── eframe App ────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Tab { Node, Cluster, Neural, Resources, Log }

pub struct NodeApp {
    pub state: Arc<Mutex<NodeDashboard>>,
    tab:       Tab,
    otp_input: String,
    coord_input: String,
}

impl NodeApp {
    pub fn new(state: Arc<Mutex<NodeDashboard>>) -> Self {
        Self {
            state,
            tab:         Tab::Node,
            otp_input:   String::new(),
            coord_input: String::new(),
        }
    }
}

impl eframe::App for NodeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint at ~4 Hz so stats feel live without burning CPU.
        ctx.request_repaint_after(Duration::from_millis(250));

        let snap = self.state.lock().unwrap().clone();

        // ── Top bar ───────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("topbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new("⬡ W1z4rD").color(Color32::from_rgb(120, 180, 255))
                );
                ui.separator();
                tab_btn(ui, &mut self.tab, Tab::Node,      "Node");
                tab_btn(ui, &mut self.tab, Tab::Cluster,   "Cluster");
                tab_btn(ui, &mut self.tab, Tab::Neural,    "Neural");
                tab_btn(ui, &mut self.tab, Tab::Resources, "Resources");
                tab_btn(ui, &mut self.tab, Tab::Log,       "Log");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (role_color, role_label) = match snap.cluster_role.as_str() {
                        "Coordinator" => (Color32::from_rgb(100, 220, 100), "● Coordinator"),
                        "Worker"      => (Color32::from_rgb(100, 180, 255), "● Worker"),
                        _             => (Color32::GRAY, "● Standalone"),
                    };
                    ui.label(RichText::new(role_label).color(role_color).small());
                    ui.separator();
                    ui.label(RichText::new(format!("up {}", snap.uptime())).small().color(Color32::GRAY));
                });
            });
        });

        // ── Log strip at bottom ────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("logbar").min_height(90.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("LOG").small().color(Color32::DARK_GRAY));
            });
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for line in snap.log_lines.iter().rev().take(20).collect::<Vec<_>>().into_iter().rev() {
                    ui.label(RichText::new(line).monospace().small());
                }
            });
        });

        // ── Central panel (selected tab) ───────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.tab {
                Tab::Node      => self.tab_node(ui, &snap),
                Tab::Cluster   => self.tab_cluster(ui, &snap),
                Tab::Neural    => self.tab_neural(ui, &snap),
                Tab::Resources => self.tab_resources(ui, &snap),
                Tab::Log       => self.tab_log(ui, &snap),
            }
        });
    }
}

// ── Tab implementations ────────────────────────────────────────────────────────

impl NodeApp {
    fn tab_node(&self, ui: &mut Ui, snap: &NodeDashboard) {
        egui::Grid::new("node_grid")
            .num_columns(2)
            .spacing([20.0, 6.0])
            .show(ui, |ui| {
                kv(ui, "Node ID",    &snap.node_id);
                kv(ui, "Uptime",     &snap.uptime());
                kv(ui, "Role",       &snap.cluster_role);
                if let Some(cid) = &snap.cluster_id {
                    kv(ui, "Cluster", cid);
                }
                ui.end_row();
                ui.label(RichText::new("Services").strong());
                ui.end_row();

                service_row(ui, "Neuro API", snap.api_neuro_port, snap.api_neuro_up,
                    &format!("{} req  {} err", snap.neuro_requests, snap.neuro_errors));
                service_row(ui, "Node  API", snap.api_node_port, snap.api_node_up, "");
                service_row(ui, "Cluster  ", snap.cluster_port, true, "SIGIL :51611");
            });
    }

    fn tab_cluster(&mut self, ui: &mut Ui, snap: &NodeDashboard) {
        ui.horizontal(|ui| {
            ui.strong("Peers");
            ui.label(RichText::new(format!("({} nodes, {} ring slots)",
                snap.cluster_peers.len() + 1, snap.ring_slots)).small().color(Color32::GRAY));
        });
        ui.separator();

        egui::Grid::new("peers")
            .num_columns(4)
            .spacing([16.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label(RichText::new("ID").strong());
                ui.label(RichText::new("Address").strong());
                ui.label(RichText::new("CPU").strong());
                ui.label(RichText::new("OS").strong());
                ui.end_row();
                for p in &snap.cluster_peers {
                    let label = if p.is_coord {
                        RichText::new(&p.short_id).color(Color32::from_rgb(100, 220, 100))
                    } else {
                        RichText::new(&p.short_id)
                    };
                    ui.label(label);
                    ui.label(&p.addr);
                    ui.label(format!("{} cores", p.cores));
                    ui.label(&p.os);
                    ui.end_row();
                }
            });

        ui.add_space(12.0);
        ui.separator();
        ui.strong("OTP — invite a new node");
        ui.add_space(4.0);
        if let Some(otp) = &snap.pending_otp {
            ui.horizontal(|ui| {
                ui.label("Active OTP:");
                ui.label(RichText::new(otp).monospace().color(Color32::from_rgb(255, 220, 80)));
            });
        }

        ui.add_space(8.0);
        ui.strong("Join existing cluster");
        ui.horizontal(|ui| {
            ui.label("Coordinator addr:");
            ui.text_edit_singleline(&mut self.coord_input);
            ui.label("OTP:");
            ui.text_edit_singleline(&mut self.otp_input);
        });
        ui.label(RichText::new("Use: w1z4rd cluster-join --coordinator <addr> --otp <OTP>").small().color(Color32::GRAY));
    }

    fn tab_neural(&self, ui: &mut Ui, snap: &NodeDashboard) {
        egui::Grid::new("neuro_grid")
            .num_columns(2)
            .spacing([20.0, 6.0])
            .show(ui, |ui| {
                kv(ui, "Pools (shards)",    &snap.neuro_pools.to_string());
                kv(ui, "Unique labels",     &snap.neuro_labels.to_string());
                kv(ui, "Episodic memory",   &format!("{} episodes", snap.neuro_episodic));
                kv(ui, "Total observations",&snap.neuro_observations.to_string());
            });
        ui.add_space(8.0);
        ui.label(RichText::new("Streams and activations are reported by connected scripts via /neuro/train and /neuro/activate.").small().color(Color32::GRAY));
    }

    fn tab_resources(&self, ui: &mut Ui, snap: &NodeDashboard) {
        ui.label(RichText::new("CPU").strong());
        let cpu = snap.cpu_pct / 100.0;
        ui.add(egui::ProgressBar::new(cpu)
            .text(format!("{:.1}%", snap.cpu_pct))
            .fill(bar_color(cpu)));

        ui.add_space(8.0);
        ui.label(RichText::new("RAM").strong());
        let ram = if snap.ram_total_gb > 0.0 { snap.ram_used_gb / snap.ram_total_gb } else { 0.0 };
        ui.add(egui::ProgressBar::new(ram)
            .text(format!("{:.1} / {:.1} GB", snap.ram_used_gb, snap.ram_total_gb))
            .fill(bar_color(ram)));
    }

    fn tab_log(&self, ui: &mut Ui, snap: &NodeDashboard) {
        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
            for line in &snap.log_lines {
                ui.label(RichText::new(line).monospace().small());
            }
        });
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn tab_btn(ui: &mut Ui, current: &mut Tab, target: Tab, label: &str) {
    if ui.selectable_label(*current == target, label).clicked() {
        *current = target;
    }
}

fn kv(ui: &mut Ui, key: &str, val: &str) {
    ui.label(RichText::new(key).color(Color32::GRAY));
    ui.label(RichText::new(val).monospace());
    ui.end_row();
}

fn service_row(ui: &mut Ui, name: &str, port: u16, up: bool, extra: &str) {
    ui.label(name);
    let dot = if up {
        RichText::new("●").color(Color32::from_rgb(80, 200, 80))
    } else {
        RichText::new("○").color(Color32::GRAY)
    };
    ui.label(dot);
    ui.label(RichText::new(format!(":{port}")).monospace());
    ui.label(RichText::new(extra).small().color(Color32::GRAY));
    ui.end_row();
}

fn bar_color(pct: f32) -> Color32 {
    if pct < 0.6      { Color32::from_rgb(60, 180, 60) }
    else if pct < 0.85 { Color32::from_rgb(220, 180, 40) }
    else               { Color32::from_rgb(220, 60, 60) }
}

// ── Resource polling (runs in background) ────────────────────────────────────

pub fn spawn_resource_poller(state: Arc<Mutex<NodeDashboard>>) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let (cpu, ram_used, ram_total) = sample_resources();
            if let Ok(mut s) = state.lock() {
                s.cpu_pct      = cpu;
                s.ram_used_gb  = ram_used;
                s.ram_total_gb = ram_total;
            }
        }
    });
}

fn sample_resources() -> (f32, f32, f32) {
    // Best-effort cross-platform resource sampling.
    let ram_total = ram_total_gb();
    let ram_used  = ram_used_gb();
    let cpu       = cpu_pct();
    (cpu, ram_used, ram_total)
}

fn ram_total_gb() -> f32 {
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb) = line.split_whitespace().nth(1) {
                    return kb.parse::<f32>().unwrap_or(0.0) / (1024.0 * 1024.0);
                }
            }
        }
    }
    0.0
}

fn ram_used_gb() -> f32 {
    #[cfg(target_os = "linux")]
    {
        let mut total = 0f32;
        let mut avail = 0f32;
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(v) = line.split_whitespace().nth(1) { total = v.parse().unwrap_or(0.0); }
                }
                if line.starts_with("MemAvailable:") {
                    if let Some(v) = line.split_whitespace().nth(1) { avail = v.parse().unwrap_or(0.0); }
                }
            }
        }
        return (total - avail) / (1024.0 * 1024.0);
    }
    #[allow(unreachable_code)]
    0.0
}

fn cpu_pct() -> f32 {
    // Lightweight: read /proc/stat twice with a short sleep (Linux).
    // On other platforms returns 0 for now.
    #[cfg(target_os = "linux")]
    {
        fn read_idle() -> Option<(u64, u64)> {
            let s = std::fs::read_to_string("/proc/stat").ok()?;
            let line = s.lines().next()?;
            let vals: Vec<u64> = line.split_whitespace().skip(1)
                .filter_map(|v| v.parse().ok()).collect();
            if vals.len() < 4 { return None; }
            let idle  = vals[3];
            let total: u64 = vals.iter().sum();
            Some((idle, total))
        }
        let s1 = read_idle();
        std::thread::sleep(Duration::from_millis(200));
        let s2 = read_idle();
        if let (Some((i1, t1)), Some((i2, t2))) = (s1, s2) {
            let dt = (t2 - t1) as f32;
            let di = (i2 - i1) as f32;
            if dt > 0.0 { return (1.0 - di / dt) * 100.0; }
        }
    }
    0.0
}
