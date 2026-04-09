//! W1z4rD Dashboard — desktop GUI front-end.
//!
//! Connects to a running w1z4rd node (default http://localhost:8090 and
//! http://localhost:8080) and displays its live state.
//!
//! Usage:
//!   w1z4rd-dashboard                          # connects to localhost defaults
//!   w1z4rd-dashboard --node http://10.0.0.5:8090 --api http://10.0.0.5:8080

use eframe::egui::{self, Color32, RichText, Ui};
use serde::Deserialize;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Cfg {
    node_url: String,
    api_url:  String,
}

impl Cfg {
    fn from_args() -> Self {
        let mut node_url = "http://localhost:8090".to_string();
        let mut api_url  = "http://localhost:8080".to_string();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--node" => { if let Some(v) = args.next() { node_url = v; } }
                "--api"  => { if let Some(v) = args.next() { api_url  = v; } }
                "--help" | "-h" => {
                    println!("Usage: w1z4rd-dashboard [--node URL] [--api URL]");
                    std::process::exit(0);
                }
                _ => {}
            }
        }
        Self { node_url, api_url }
    }
}

// ── Data pulled from the node APIs ───────────────────────────────────────────

#[derive(Default, Clone)]
struct NodeState {
    // Connection
    node_url:        String,
    api_url:         String,
    node_reachable:  bool,
    api_reachable:   bool,
    last_poll:       Option<Instant>,

    // Node identity (from /health or /neuro/snapshot)
    node_id:         String,
    uptime_secs:     u64,
    api_status:      String,    // "ok" | "starting" | "degraded"
    total_requests:  u64,
    completed_req:   u64,

    // Neuro fabric (from /neuro/snapshot)
    neuro_pools:     usize,
    neuro_labels:    usize,
    neuro_streams:   Vec<String>,

    // Cluster (from /cluster/status if available)
    cluster_nodes:   Vec<ClusterPeer>,
    cluster_role:    String,
    cluster_id:      String,
    ring_slots:      usize,

    // Log
    log:             VecDeque<String>,
}

impl NodeState {
    fn push_log(&mut self, s: impl Into<String>) {
        self.log.push_back(s.into());
        while self.log.len() > 300 { self.log.pop_front(); }
    }
}

#[derive(Default, Clone)]
struct ClusterPeer {
    short_id: String,
    addr:     String,
    os:       String,
    cores:    u32,
    is_coord: bool,
}

// ── API response shapes (minimal) ─────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct HealthResp {
    #[serde(default)] status:              String,
    #[serde(default)] uptime_secs:         u64,
    #[serde(default)] total_requests:      u64,
    #[serde(default)] completed_requests:  u64,
}

#[derive(Deserialize, Default)]
struct NeuroSnap {
    #[serde(default)] pools:   Vec<serde_json::Value>,
    #[serde(default)] streams: Vec<String>,
}

#[derive(Deserialize, Default)]
struct NodeHealth {
    #[serde(default)] node_id: String,
}

// ── Poller ────────────────────────────────────────────────────────────────────

fn spawn_poller(state: Arc<Mutex<NodeState>>) {
    std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .expect("HTTP client");
        loop {
            std::thread::sleep(Duration::from_secs(2));

            let (node_url, api_url) = {
                let s = state.lock().unwrap();
                (s.node_url.clone(), s.api_url.clone())
            };

            // ── Neuro API healthz ─────────────────────────────────────────────
            let api_ok;
            match client.get(format!("{api_url}/healthz")).send()
                .and_then(|r| r.json::<HealthResp>())
            {
                Ok(h) => {
                    api_ok = true;
                    let mut s = state.lock().unwrap();
                    s.api_reachable    = true;
                    s.api_status       = h.status;
                    s.uptime_secs      = h.uptime_secs;
                    s.total_requests   = h.total_requests;
                    s.completed_req    = h.completed_requests;
                }
                Err(e) => {
                    api_ok = false;
                    let mut s = state.lock().unwrap();
                    s.api_reachable = false;
                    s.push_log(format!("API unreachable: {e}"));
                }
            }

            // ── Neuro snapshot ────────────────────────────────────────────────
            if api_ok {
                if let Ok(snap) = client.get(format!("{api_url}/neuro/snapshot")).send()
                    .and_then(|r| r.json::<NeuroSnap>())
                {
                    let mut s = state.lock().unwrap();
                    s.neuro_pools   = snap.pools.len();
                    s.neuro_streams = snap.streams;
                    // Count unique labels across all pools.
                    s.neuro_labels = snap.pools.iter()
                        .filter_map(|p| p.get("label_count").and_then(|v| v.as_u64()))
                        .sum::<u64>() as usize;
                }
            }

            // ── Node API health ───────────────────────────────────────────────
            match client.get(format!("{node_url}/health")).send()
                .and_then(|r| r.json::<serde_json::Value>())
            {
                Ok(v) => {
                    let mut s = state.lock().unwrap();
                    s.node_reachable = true;
                    if let Some(id) = v.get("node_id").and_then(|x| x.as_str()) {
                        s.node_id = id.to_string();
                    }
                    s.last_poll = Some(Instant::now());
                }
                Err(e) => {
                    let mut s = state.lock().unwrap();
                    s.node_reachable = false;
                    s.push_log(format!("Node unreachable: {e}"));
                }
            }

            // ── Cluster status (best-effort) ──────────────────────────────────
            if let Ok(v) = client.get(format!("{node_url}/cluster/status")).send()
                .and_then(|r| r.json::<serde_json::Value>())
            {
                let mut s = state.lock().unwrap();
                if let Some(nodes) = v.get("nodes").and_then(|n| n.as_array()) {
                    s.cluster_nodes = nodes.iter().map(|n| ClusterPeer {
                        short_id: n["id"].as_str().unwrap_or("?").chars().take(8).collect(),
                        addr:     n["addr"].as_str().unwrap_or("?").to_string(),
                        os:       n["capabilities"]["os"].as_str().unwrap_or("?").to_string(),
                        cores:    n["capabilities"]["cpu_cores"].as_u64().unwrap_or(0) as u32,
                        is_coord: n["is_coordinator"].as_bool().unwrap_or(false),
                    }).collect();
                }
                s.cluster_id   = v["cluster_id"].as_str().unwrap_or("").to_string();
                s.cluster_role = v["role"].as_str().unwrap_or("Standalone").to_string();
                s.ring_slots   = v["ring_size"].as_u64().unwrap_or(0) as usize;
            }
        }
    });
}

// ── eframe App ────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Tab { Overview, Cluster, Neural, Log }

/// Mutable control state for the Cluster tab (held on Dashboard, not polled state).
struct ClusterControls {
    coord_input: String,
    otp_input:   String,
    status_msg:  String,
}

impl Default for ClusterControls {
    fn default() -> Self {
        Self {
            coord_input: "192.168.1.84:51611".to_string(),
            otp_input:   String::new(),
            status_msg:  String::new(),
        }
    }
}

struct Dashboard {
    state:   Arc<Mutex<NodeState>>,
    tab:     Tab,
    cluster: ClusterControls,
}

impl Dashboard {
    fn new(state: Arc<Mutex<NodeState>>) -> Self {
        Self { state, tab: Tab::Overview, cluster: ClusterControls::default() }
    }
}

impl eframe::App for Dashboard {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(500));
        let snap = self.state.lock().unwrap().clone();

        // ── Top bar ───────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("⬡ W1z4rD").color(Color32::from_rgb(120, 180, 255)));
                ui.separator();
                tab_btn(ui, &mut self.tab, Tab::Overview, "Overview");
                tab_btn(ui, &mut self.tab, Tab::Cluster,  "Cluster");
                tab_btn(ui, &mut self.tab, Tab::Neural,   "Neural");
                tab_btn(ui, &mut self.tab, Tab::Log,      "Log");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dot = |up: bool| if up {
                        RichText::new("●").color(Color32::from_rgb(80, 200, 80))
                    } else {
                        RichText::new("○").color(Color32::GRAY)
                    };
                    ui.label(dot(snap.api_reachable));  ui.label(RichText::new(":8080 API").small());
                    ui.separator();
                    ui.label(dot(snap.node_reachable)); ui.label(RichText::new(":8090 Node").small());
                });
            });
        });

        // ── Log bar ───────────────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("log_bar").min_height(70.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for line in snap.log.iter().rev().take(8).collect::<Vec<_>>().into_iter().rev() {
                    ui.label(RichText::new(line).monospace().small().color(Color32::GRAY));
                }
            });
        });

        // ── Central panel ─────────────────────────────────────────────────────
        let node_url = snap.node_url.clone();
        let state_arc = self.state.clone();
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.tab {
                Tab::Overview => tab_overview(ui, &snap),
                Tab::Cluster  => tab_cluster(ui, &snap, &mut self.cluster, &node_url, state_arc),
                Tab::Neural   => tab_neural(ui, &snap),
                Tab::Log      => tab_log(ui, &snap),
            }
        });
    }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

fn tab_overview(ui: &mut Ui, s: &NodeState) {
    egui::Grid::new("ov").num_columns(2).spacing([20.0, 5.0]).show(ui, |ui| {
        kv(ui, "Node ID",     if s.node_id.is_empty() { "connecting…" } else { &s.node_id });
        kv(ui, "API status",  &s.api_status);
        kv(ui, "Uptime",      &fmt_uptime(s.uptime_secs));
        kv(ui, "Requests",    &format!("{} total  {} completed", s.total_requests, s.completed_req));
        kv(ui, "Node URL",    &s.node_url);
        kv(ui, "API URL",     &s.api_url);
        if !s.cluster_id.is_empty() {
            kv(ui, "Cluster", &s.cluster_id);
            kv(ui, "Role",    &s.cluster_role);
        }
    });
}

fn tab_cluster(
    ui: &mut Ui,
    s: &NodeState,
    ctrl: &mut ClusterControls,
    node_url: &str,
    state_arc: Arc<Mutex<NodeState>>,
) {
    // ── Control panel ─────────────────────────────────────────────────────────
    let is_standalone = s.cluster_nodes.is_empty();
    let is_coordinator = s.cluster_role == "coordinator";

    if is_standalone {
        ui.strong("Cluster");
        ui.label(RichText::new("This node is running standalone.").color(Color32::GRAY));
        ui.add_space(6.0);

        ui.collapsing("Init new cluster (this machine becomes coordinator)", |ui| {
            if ui.button("  Init Cluster  ").clicked() {
                let url = format!("{node_url}/cluster/init");
                let state2 = state_arc.clone();
                ctrl.status_msg = "Initialising…".to_string();
                std::thread::spawn(move || {
                    let client = reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(10))
                        .build().unwrap();
                    match client.post(&url).json(&serde_json::json!({})).send()
                        .and_then(|r| r.json::<serde_json::Value>())
                    {
                        Ok(v) => {
                            let msg = if let Some(otp) = v["otp"].as_str() {
                                format!("Cluster started. OTP: {}  (cluster {})", otp, &v["cluster_id"].as_str().unwrap_or("?")[..8])
                            } else {
                                format!("Error: {}", v["error"].as_str().unwrap_or("unknown"))
                            };
                            state2.lock().unwrap().push_log(msg);
                        }
                        Err(e) => { state2.lock().unwrap().push_log(format!("Init failed: {e}")); }
                    }
                });
            }
        });

        ui.add_space(6.0);
        ui.collapsing("Join existing cluster", |ui| {
            ui.horizontal(|ui| {
                ui.label("Coordinator:");
                ui.text_edit_singleline(&mut ctrl.coord_input);
            });
            ui.horizontal(|ui| {
                ui.label("OTP:        ");
                ui.text_edit_singleline(&mut ctrl.otp_input);
            });
            ui.add_space(4.0);
            let can_join = !ctrl.coord_input.is_empty() && !ctrl.otp_input.is_empty();
            ui.add_enabled_ui(can_join, |ui| {
                if ui.button("  Join Cluster  ").clicked() {
                    let url = format!("{node_url}/cluster/join");
                    let body = serde_json::json!({
                        "coordinator": ctrl.coord_input,
                        "otp": ctrl.otp_input,
                    });
                    let state2 = state_arc.clone();
                    ctrl.status_msg = "Joining…".to_string();
                    ctrl.otp_input.clear();
                    std::thread::spawn(move || {
                        let client = reqwest::blocking::Client::builder()
                            .timeout(std::time::Duration::from_secs(15))
                            .build().unwrap();
                        match client.post(&url).json(&body).send()
                            .and_then(|r| r.json::<serde_json::Value>())
                        {
                            Ok(v) => {
                                let msg = if v["status"].as_str() == Some("ok") {
                                    format!("Joined cluster {} ({} nodes)",
                                        &v["cluster_id"].as_str().unwrap_or("?")[..8],
                                        v["node_count"].as_u64().unwrap_or(0))
                                } else {
                                    format!("Join failed: {}", v["error"].as_str().unwrap_or("unknown"))
                                };
                                state2.lock().unwrap().push_log(msg);
                            }
                            Err(e) => { state2.lock().unwrap().push_log(format!("Join failed: {e}")); }
                        }
                    });
                }
            });
        });
    } else if is_coordinator {
        // ── Coordinator controls ──────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.strong("Cluster Coordinator");
            ui.label(RichText::new(format!("({} nodes, {} ring slots)", s.cluster_nodes.len(), s.ring_slots)).small().color(Color32::GRAY));
        });
        ui.add_space(4.0);
        if ui.button("Generate new OTP (for next joiner)").clicked() {
            let url = format!("{node_url}/cluster/otp");
            let state2 = state_arc.clone();
            std::thread::spawn(move || {
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(5))
                    .build().unwrap();
                match client.post(&url).send()
                    .and_then(|r| r.json::<serde_json::Value>())
                {
                    Ok(v) => {
                        let msg = if let Some(otp) = v["otp"].as_str() {
                            format!("New OTP: {otp}  (share with joining node, single-use)")
                        } else {
                            format!("OTP error: {}", v["error"].as_str().unwrap_or("unknown"))
                        };
                        state2.lock().unwrap().push_log(msg);
                    }
                    Err(e) => { state2.lock().unwrap().push_log(format!("OTP request failed: {e}")); }
                }
            });
        }
        ui.add_space(4.0);
        ui.separator();
    }

    // ── Peer table (always shown when in cluster) ─────────────────────────────
    if !is_standalone {
        if !is_coordinator { // title already shown above for coordinator
            ui.horizontal(|ui| {
                ui.strong("Cluster Worker");
                ui.label(RichText::new(format!("({} nodes, {} ring slots)", s.cluster_nodes.len(), s.ring_slots)).small().color(Color32::GRAY));
            });
            ui.separator();
        }
        egui::Grid::new("cl").num_columns(4).spacing([16.0, 4.0]).striped(true).show(ui, |ui| {
            ui.label(RichText::new("ID").strong());
            ui.label(RichText::new("Address").strong());
            ui.label(RichText::new("CPU").strong());
            ui.label(RichText::new("OS").strong());
            ui.end_row();
            for p in &s.cluster_nodes {
                let id_text = if p.is_coord {
                    RichText::new(&p.short_id).color(Color32::from_rgb(100, 220, 100))
                } else {
                    RichText::new(&p.short_id)
                };
                ui.label(id_text);
                ui.label(&p.addr);
                ui.label(format!("{} cores", p.cores));
                ui.label(&p.os);
                ui.end_row();
            }
        });
    }
}

fn tab_neural(ui: &mut Ui, s: &NodeState) {
    egui::Grid::new("ne").num_columns(2).spacing([20.0, 5.0]).show(ui, |ui| {
        kv(ui, "Pools (shards)",  &s.neuro_pools.to_string());
        kv(ui, "Unique labels",   &s.neuro_labels.to_string());
        kv(ui, "Active streams",  &s.neuro_streams.len().to_string());
    });
    if !s.neuro_streams.is_empty() {
        ui.add_space(6.0);
        ui.strong("Streams");
        for stream in &s.neuro_streams {
            ui.label(RichText::new(format!("  {stream}")).monospace().small());
        }
    }
    ui.add_space(8.0);
    ui.label(RichText::new("Training scripts push observations via POST /neuro/train.").small().color(Color32::GRAY));
}

fn tab_log(ui: &mut Ui, s: &NodeState) {
    egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
        for line in &s.log {
            ui.label(RichText::new(line).monospace().small());
        }
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn tab_btn(ui: &mut Ui, cur: &mut Tab, target: Tab, label: &str) {
    if ui.selectable_label(*cur == target, label).clicked() { *cur = target; }
}

fn kv(ui: &mut Ui, key: &str, val: &str) {
    ui.label(RichText::new(key).color(Color32::GRAY));
    ui.label(RichText::new(val).monospace());
    ui.end_row();
}

fn fmt_uptime(s: u64) -> String {
    if s < 60      { format!("{s}s") }
    else if s < 3600 { format!("{}m {}s", s/60, s%60) }
    else           { format!("{}h {}m", s/3600, (s%3600)/60) }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let cfg = Cfg::from_args();

    let state = Arc::new(Mutex::new(NodeState {
        node_url:     cfg.node_url.clone(),
        api_url:      cfg.api_url.clone(),
        cluster_role: "Standalone".into(),
        api_status:   "connecting…".into(),
        ..Default::default()
    }));

    spawn_poller(state.clone());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("W1z4rD Dashboard")
            .with_inner_size([780.0, 520.0])
            .with_min_inner_size([500.0, 360.0]),
        ..Default::default()
    };

    eframe::run_native(
        "W1z4rD Dashboard",
        options,
        Box::new(|_cc| Ok(Box::new(Dashboard::new(state)))),
    ).map_err(|e| anyhow::anyhow!("{e}"))
}
