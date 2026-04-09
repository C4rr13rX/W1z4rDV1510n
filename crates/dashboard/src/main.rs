//! W1z4rD V1510n Dashboard — desktop GUI front-end.

use eframe::egui::{self, Align, Color32, FontId, Frame, Layout, Margin, RichText, Rounding,
                   Stroke, TextStyle, Ui, Vec2};
use serde::{Deserialize, Serialize};
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

// ── Theme ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
enum Theme {
    Dark,    // default — logo colors (dark navy / steel blue)
    Classic, // terminal green-on-black
    Light,   // brand colors on light background
}

impl Theme {
    fn label(&self) -> &'static str {
        match self {
            Theme::Dark    => "Dark (W1z4rD)",
            Theme::Classic => "Classic (Terminal)",
            Theme::Light   => "Light",
        }
    }
}

struct Palette {
    bg:           Color32,
    panel:        Color32,
    card:         Color32,
    accent:       Color32,
    accent_dim:   Color32,
    text:         Color32,
    text_dim:     Color32,
    text_code:    Color32,
    success:      Color32,
    warning:      Color32,
    error:        Color32,
    border:       Color32,
    tab_active:   Color32,
    tab_inactive: Color32,
}

fn palette(theme: Theme) -> Palette {
    match theme {
        Theme::Dark => Palette {
            bg:           Color32::from_rgb(10,  12,  18),
            panel:        Color32::from_rgb(13,  20,  32),
            card:         Color32::from_rgb(18,  28,  44),
            accent:       Color32::from_rgb(74,  143, 212),
            accent_dim:   Color32::from_rgb(45,  95,  166),
            text:         Color32::from_rgb(216, 232, 248),
            text_dim:     Color32::from_rgb(100, 136, 168),
            text_code:    Color32::from_rgb(106, 175, 232),
            success:      Color32::from_rgb(61,  184, 136),
            warning:      Color32::from_rgb(212, 144, 60),
            error:        Color32::from_rgb(212, 64,  64),
            border:       Color32::from_rgb(30,  52,  90),
            tab_active:   Color32::from_rgb(74,  143, 212),
            tab_inactive: Color32::from_rgb(36,  56,  90),
        },
        Theme::Classic => Palette {
            bg:           Color32::from_rgb(10,  12,  10),
            panel:        Color32::from_rgb(18,  22,  18),
            card:         Color32::from_rgb(26,  30,  26),
            accent:       Color32::from_rgb(0,   200, 80),
            accent_dim:   Color32::from_rgb(0,   130, 50),
            text:         Color32::from_rgb(200, 220, 200),
            text_dim:     Color32::from_rgb(100, 140, 100),
            text_code:    Color32::from_rgb(0,   200, 80),
            success:      Color32::from_rgb(0,   200, 80),
            warning:      Color32::from_rgb(200, 180, 0),
            error:        Color32::from_rgb(200, 50,  50),
            border:       Color32::from_rgb(0,   80,  30),
            tab_active:   Color32::from_rgb(0,   200, 80),
            tab_inactive: Color32::from_rgb(30,  50,  30),
        },
        Theme::Light => Palette {
            bg:           Color32::from_rgb(232, 238, 248),
            panel:        Color32::from_rgb(210, 222, 240),
            card:         Color32::from_rgb(245, 248, 255),
            accent:       Color32::from_rgb(45,  95,  160),
            accent_dim:   Color32::from_rgb(26,  74,  140),
            text:         Color32::from_rgb(13,  22,  40),
            text_dim:     Color32::from_rgb(74,  104, 136),
            text_code:    Color32::from_rgb(26,  74,  140),
            success:      Color32::from_rgb(20,  140, 90),
            warning:      Color32::from_rgb(160, 100, 20),
            error:        Color32::from_rgb(160, 30,  30),
            border:       Color32::from_rgb(160, 188, 220),
            tab_active:   Color32::from_rgb(45,  95,  160),
            tab_inactive: Color32::from_rgb(180, 200, 224),
        },
    }
}

fn apply_theme(ctx: &egui::Context, theme: Theme) {
    let p = palette(theme);
    let mut style = (*ctx.style()).clone();

    // Fonts
    style.text_styles = [
        (TextStyle::Heading,   FontId::proportional(18.0)),
        (TextStyle::Body,      FontId::proportional(13.0)),
        (TextStyle::Monospace, FontId::monospace(12.0)),
        (TextStyle::Button,    FontId::proportional(13.0)),
        (TextStyle::Small,     FontId::proportional(11.0)),
    ].into();

    style.spacing.item_spacing    = Vec2::new(8.0, 5.0);
    style.spacing.button_padding  = Vec2::new(12.0, 5.0);
    style.spacing.window_margin   = Margin::same(14.0);
    style.spacing.indent          = 16.0;

    let mut vis = egui::Visuals::dark();
    vis.window_fill              = p.panel;
    vis.panel_fill               = p.bg;
    vis.faint_bg_color           = p.card;
    vis.extreme_bg_color         = p.panel;
    vis.code_bg_color            = p.card;
    vis.window_stroke            = Stroke::new(1.0, p.border);
    vis.widgets.noninteractive.bg_fill   = p.card;
    vis.widgets.noninteractive.fg_stroke = Stroke::new(1.0, p.text_dim);
    vis.widgets.inactive.bg_fill         = p.panel;
    vis.widgets.inactive.fg_stroke       = Stroke::new(1.0, p.text_dim);
    vis.widgets.hovered.bg_fill          = p.accent_dim;
    vis.widgets.hovered.fg_stroke        = Stroke::new(1.5, p.accent);
    vis.widgets.active.bg_fill           = p.accent;
    vis.widgets.active.fg_stroke         = Stroke::new(2.0, p.text);
    vis.selection.bg_fill                = p.accent_dim;
    vis.selection.stroke                 = Stroke::new(1.0, p.accent);
    vis.override_text_color              = Some(p.text);
    vis.hyperlink_color                  = p.accent;
    vis.window_rounding                  = Rounding::same(8.0);
    vis.widgets.noninteractive.rounding  = Rounding::same(4.0);
    vis.widgets.inactive.rounding        = Rounding::same(4.0);
    vis.widgets.active.rounding          = Rounding::same(4.0);
    style.visuals = vis;

    ctx.set_style(style);
}

// ── App state ─────────────────────────────────────────────────────────────────

enum AppPhase {
    Splash(Instant),
    Main,
}

// ── Node state (polled) ───────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct NodeState {
    node_url:       String,
    api_url:        String,
    node_reachable: bool,
    api_reachable:  bool,
    last_poll:      Option<Instant>,
    node_id:        String,
    uptime_secs:    u64,
    api_status:     String,
    total_requests: u64,
    completed_req:  u64,
    neuro_pools:    usize,
    neuro_labels:   usize,
    neuro_streams:  Vec<String>,
    cluster_nodes:  Vec<ClusterPeer>,
    cluster_role:   String,
    cluster_id:     String,
    ring_slots:     usize,
    log:            VecDeque<String>,
}

impl NodeState {
    fn push_log(&mut self, s: impl Into<String>) {
        let msg = s.into();
        self.log.push_back(msg);
        while self.log.len() > 500 { self.log.pop_front(); }
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

#[derive(Deserialize, Default)]
struct HealthResp {
    #[serde(default)] status:             String,
    #[serde(default)] uptime_secs:        u64,
    #[serde(default)] total_requests:     u64,
    #[serde(default)] completed_requests: u64,
}

#[derive(Deserialize, Default)]
struct NeuroSnap {
    #[serde(default)] pools:   Vec<serde_json::Value>,
    #[serde(default)] streams: Vec<String>,
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

            let api_ok;
            match client.get(format!("{api_url}/healthz")).send()
                .and_then(|r| r.json::<HealthResp>())
            {
                Ok(h) => {
                    api_ok = true;
                    let mut s = state.lock().unwrap();
                    s.api_reachable  = true;
                    s.api_status     = h.status;
                    s.uptime_secs    = h.uptime_secs;
                    s.total_requests = h.total_requests;
                    s.completed_req  = h.completed_requests;
                }
                Err(e) => {
                    api_ok = false;
                    let mut s = state.lock().unwrap();
                    s.api_reachable = false;
                    s.push_log(format!("[neuro API] {e}"));
                }
            }

            if api_ok {
                if let Ok(snap) = client.get(format!("{api_url}/neuro/snapshot")).send()
                    .and_then(|r| r.json::<NeuroSnap>())
                {
                    let mut s = state.lock().unwrap();
                    s.neuro_pools   = snap.pools.len();
                    s.neuro_streams = snap.streams;
                    s.neuro_labels  = snap.pools.iter()
                        .filter_map(|p| p.get("label_count").and_then(|v| v.as_u64()))
                        .sum::<u64>() as usize;
                }
            }

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
                    s.push_log(format!("[node API] {e}"));
                }
            }

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
                s.cluster_role = v["role"].as_str().unwrap_or("standalone").to_string();
                s.ring_slots   = v["ring_size"].as_u64().unwrap_or(0) as usize;
            }
        }
    });
}

// ── Cluster controls (mutable, not polled) ────────────────────────────────────

#[derive(Default)]
struct ClusterControls {
    coord_input: String,
    otp_input:   String,
}

impl ClusterControls {
    fn new() -> Self {
        Self {
            coord_input: "192.168.1.84:51611".to_string(),
            otp_input:   String::new(),
        }
    }
}

// ── Main app ──────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Tab { Overview, Cluster, Neural, Setup, Log }

struct Dashboard {
    phase:   AppPhase,
    logo:    Option<egui::TextureHandle>,
    state:   Arc<Mutex<NodeState>>,
    tab:     Tab,
    theme:   Theme,
    cluster: ClusterControls,
}

impl Dashboard {
    fn new(state: Arc<Mutex<NodeState>>) -> Self {
        Self {
            phase:   AppPhase::Splash(Instant::now()),
            logo:    None,
            state,
            tab:     Tab::Overview,
            theme:   Theme::Dark,
            cluster: ClusterControls::new(),
        }
    }

    fn load_logo(&mut self, ctx: &egui::Context) {
        if self.logo.is_some() { return; }
        let bytes = include_bytes!("../assets/logo.png");
        if let Ok(img) = image::load_from_memory(bytes) {
            let img = img.to_rgba8();
            let (w, h) = img.dimensions();
            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                img.as_raw(),
            );
            self.logo = Some(ctx.load_texture("logo", color_image, egui::TextureOptions::LINEAR));
        }
    }
}

impl eframe::App for Dashboard {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(500));
        self.load_logo(ctx);
        apply_theme(ctx, self.theme);
        let p = palette(self.theme);

        // ── Splash ────────────────────────────────────────────────────────────
        if let AppPhase::Splash(start) = self.phase {
            let elapsed = start.elapsed().as_secs_f32();
            if elapsed > 2.8 {
                self.phase = AppPhase::Main;
            } else {
                egui::CentralPanel::default()
                    .frame(Frame::none().fill(p.bg))
                    .show(ctx, |ui| {
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                            ui.add_space(ui.available_height() * 0.12);
                            if let Some(tex) = &self.logo {
                                let max = 320.0f32.min(ui.available_width() * 0.55);
                                ui.add(egui::Image::new(tex).max_size(Vec2::splat(max)));
                            }
                            ui.add_space(28.0);
                            ui.label(
                                RichText::new("W1z4rD V1510n")
                                    .size(32.0)
                                    .color(p.accent)
                                    .strong(),
                            );
                            ui.label(
                                RichText::new("Neural Fabric Node")
                                    .size(14.0)
                                    .color(p.text_dim),
                            );
                            ui.add_space(18.0);
                            ui.label(
                                RichText::new("by C4rr13rX")
                                    .size(11.0)
                                    .color(p.text_dim),
                            );
                            // Fade-out progress bar
                            let fade = ((2.8 - elapsed) / 0.5).clamp(0.0, 1.0);
                            ui.add_space(32.0);
                            let bar_w = 200.0f32.min(ui.available_width() * 0.4);
                            ui.add(egui::ProgressBar::new(1.0 - elapsed / 2.8)
                                .desired_width(bar_w)
                                .show_percentage());
                            let _ = fade;
                        });
                    });
                return;
            }
        }

        // ── Main UI ───────────────────────────────────────────────────────────
        let snap = self.state.lock().unwrap().clone();

        // Top bar
        egui::TopBottomPanel::top("top")
            .frame(Frame::none().fill(p.panel).inner_margin(Margin::symmetric(14.0, 8.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Logo thumbnail
                    if let Some(tex) = &self.logo {
                        ui.add(egui::Image::new(tex).max_size(Vec2::new(32.0, 32.0)));
                        ui.add_space(4.0);
                    }
                    ui.label(
                        RichText::new("W1z4rD V1510n")
                            .size(16.0)
                            .color(p.accent)
                            .strong(),
                    );
                    ui.separator();
                    for (t, label, icon) in [
                        (Tab::Overview, "Overview", "◈"),
                        (Tab::Cluster,  "Cluster",  "⬡"),
                        (Tab::Neural,   "Neural",   "⬤"),
                        (Tab::Setup,    "Setup",    "⚙"),
                        (Tab::Log,      "Log",      "≡"),
                    ] {
                        let active = self.tab == t;
                        let col = if active { p.tab_active } else { p.tab_inactive };
                        if ui.add(
                            egui::Button::new(
                                RichText::new(format!("{icon} {label}"))
                                    .color(if active { p.text } else { p.text_dim })
                                    .size(12.5),
                            )
                            .fill(col)
                            .rounding(Rounding::same(5.0))
                            .min_size(Vec2::new(80.0, 26.0)),
                        ).clicked() {
                            self.tab = t;
                        }
                    }

                    // Right: status dots + theme selector
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        egui::ComboBox::from_id_source("theme")
                            .selected_text(self.theme.label())
                            .width(140.0)
                            .show_ui(ui, |ui| {
                                for t in [Theme::Dark, Theme::Classic, Theme::Light] {
                                    ui.selectable_value(&mut self.theme, t, t.label());
                                }
                            });
                        ui.add_space(8.0);
                        let dot = |up: bool| {
                            if up { RichText::new("●").color(p.success) }
                            else  { RichText::new("○").color(p.text_dim) }
                        };
                        ui.label(dot(snap.api_reachable));
                        ui.label(RichText::new(":8080").small().color(p.text_dim));
                        ui.add_space(4.0);
                        ui.label(dot(snap.node_reachable));
                        ui.label(RichText::new(":8090").small().color(p.text_dim));
                    });
                });
            });

        // Bottom log strip
        egui::TopBottomPanel::bottom("log_strip")
            .min_height(52.0)
            .max_height(52.0)
            .frame(Frame::none().fill(p.panel).inner_margin(Margin::symmetric(14.0, 6.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                    for line in snap.log.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev() {
                        ui.label(RichText::new(line).monospace().small().color(p.text_dim));
                    }
                });
            });

        // Central panel
        let node_url = snap.node_url.clone();
        let state_arc = self.state.clone();
        egui::CentralPanel::default()
            .frame(Frame::none().fill(p.bg).inner_margin(Margin::same(16.0)))
            .show(ctx, |ui| {
                match self.tab {
                    Tab::Overview => tab_overview(ui, &snap, &p),
                    Tab::Cluster  => tab_cluster(ui, &snap, &p, &mut self.cluster, &node_url, state_arc),
                    Tab::Neural   => tab_neural(ui, &snap, &p),
                    Tab::Setup    => tab_setup(ui, &p, &node_url, state_arc),
                    Tab::Log      => tab_log(ui, &snap, &p),
                }
            });
    }
}

// ── Overview tab ─────────────────────────────────────────────────────────────

fn tab_overview(ui: &mut Ui, s: &NodeState, p: &Palette) {
    section_title(ui, p, "Node Status");
    card(ui, p, |ui| {
        kv_row(ui, p, "Node ID",    if s.node_id.is_empty() { "connecting…" } else { &s.node_id });
        kv_row(ui, p, "API Status", &s.api_status);
        kv_row(ui, p, "Uptime",     &fmt_uptime(s.uptime_secs));
        kv_row(ui, p, "Requests",   &format!("{}  completed: {}", s.total_requests, s.completed_req));
        kv_row(ui, p, "Node URL",   &s.node_url);
        kv_row(ui, p, "API URL",    &s.api_url);
        if !s.cluster_id.is_empty() {
            kv_row(ui, p, "Cluster", &s.cluster_id[..8.min(s.cluster_id.len())]);
            kv_row(ui, p, "Role",    &s.cluster_role);
        }
    });

    ui.add_space(12.0);
    status_pills(ui, p, s);
}

fn status_pills(ui: &mut Ui, p: &Palette, s: &NodeState) {
    ui.horizontal_wrapped(|ui| {
        pill(ui, p, ":8080 Neuro API", s.api_reachable);
        pill(ui, p, ":8090 Node API",  s.node_reachable);
        if !s.cluster_id.is_empty() {
            pill(ui, p, &format!("Cluster ({} nodes)", s.cluster_nodes.len()), true);
        }
    });
}

fn pill(ui: &mut Ui, p: &Palette, label: &str, up: bool) {
    let (bg, fg) = if up {
        (p.success.gamma_multiply(0.2), p.success)
    } else {
        (p.error.gamma_multiply(0.2), p.error)
    };
    let dot = if up { "● " } else { "○ " };
    Frame::none()
        .fill(bg)
        .rounding(Rounding::same(12.0))
        .inner_margin(Margin::symmetric(10.0, 4.0))
        .show(ui, |ui| {
            ui.label(RichText::new(format!("{dot}{label}")).small().color(fg));
        });
    ui.add_space(4.0);
}

// ── Cluster tab ───────────────────────────────────────────────────────────────

fn tab_cluster(
    ui: &mut Ui,
    s: &NodeState,
    p: &Palette,
    ctrl: &mut ClusterControls,
    node_url: &str,
    state_arc: Arc<Mutex<NodeState>>,
) {
    let is_standalone   = s.cluster_nodes.is_empty() && s.cluster_role != "coordinator" && s.cluster_role != "worker";
    let is_coordinator  = s.cluster_role == "coordinator";

    if is_standalone {
        section_title(ui, p, "Join or Create a Cluster");
        ui.label(RichText::new(
            "A cluster distributes neural fabric work across multiple machines. \
             Each machine runs one node. One node is the coordinator; the rest are workers."
        ).small().color(p.text_dim));
        ui.add_space(10.0);

        // Init card
        card(ui, p, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("⬡  Start a New Cluster").color(p.accent).strong());
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new("This machine becomes the coordinator").small().color(p.text_dim));
                });
            });
            ui.add_space(4.0);
            if accent_button(ui, p, "  Init Cluster on This Machine  ").clicked() {
                let url = format!("{node_url}/cluster/init");
                let st2 = state_arc.clone();
                std::thread::spawn(move || {
                    let client = reqwest::blocking::Client::builder()
                        .timeout(Duration::from_secs(10)).build().unwrap();
                    match client.post(&url).json(&serde_json::json!({})).send()
                        .and_then(|r| r.json::<serde_json::Value>())
                    {
                        Ok(v) => {
                            let msg = if let Some(otp) = v["otp"].as_str() {
                                format!("[cluster] Started. OTP: {}  id={}",
                                    otp, &v["cluster_id"].as_str().unwrap_or("?")[..8])
                            } else {
                                format!("[cluster] Error: {}", v["error"].as_str().unwrap_or("?"))
                            };
                            st2.lock().unwrap().push_log(msg);
                        }
                        Err(e) => { st2.lock().unwrap().push_log(format!("[cluster] Init failed: {e}")); }
                    }
                });
            }
        });

        ui.add_space(8.0);

        // Join card
        card(ui, p, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("⬡  Join an Existing Cluster").color(p.accent).strong());
            });
            ui.add_space(6.0);
            ui.label(RichText::new("Enter the coordinator's IP:port and the OTP printed when it started.").small().color(p.text_dim));
            ui.add_space(4.0);
            egui::Grid::new("join_form").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label(RichText::new("Coordinator").color(p.text_dim));
                ui.add(egui::TextEdit::singleline(&mut ctrl.coord_input)
                    .hint_text("192.168.x.x:51611")
                    .desired_width(220.0));
                ui.end_row();
                ui.label(RichText::new("OTP").color(p.text_dim));
                ui.add(egui::TextEdit::singleline(&mut ctrl.otp_input)
                    .hint_text("WORD-NNNN")
                    .desired_width(220.0));
                ui.end_row();
            });
            ui.add_space(6.0);
            ui.add_enabled_ui(!ctrl.coord_input.is_empty() && !ctrl.otp_input.is_empty(), |ui| {
                if accent_button(ui, p, "  Join Cluster  ").clicked() {
                    let url  = format!("{node_url}/cluster/join");
                    let body = serde_json::json!({
                        "coordinator": ctrl.coord_input,
                        "otp":         ctrl.otp_input,
                    });
                    let st2 = state_arc.clone();
                    ctrl.otp_input.clear();
                    std::thread::spawn(move || {
                        let client = reqwest::blocking::Client::builder()
                            .timeout(Duration::from_secs(15)).build().unwrap();
                        match client.post(&url).json(&body).send()
                            .and_then(|r| r.json::<serde_json::Value>())
                        {
                            Ok(v) => {
                                let msg = if v["status"].as_str() == Some("ok") {
                                    format!("[cluster] Joined {}  ({} nodes)",
                                        &v["cluster_id"].as_str().unwrap_or("?")[..8],
                                        v["node_count"].as_u64().unwrap_or(0))
                                } else {
                                    format!("[cluster] Join failed: {}", v["error"].as_str().unwrap_or("?"))
                                };
                                st2.lock().unwrap().push_log(msg);
                            }
                            Err(e) => { st2.lock().unwrap().push_log(format!("[cluster] Join failed: {e}")); }
                        }
                    });
                }
            });
        });
    } else {
        // In cluster — show header + optional OTP generation
        let role_label = if is_coordinator { "Coordinator" } else { "Worker" };
        section_title(ui, p, &format!("Cluster  —  {role_label}"));

        if is_coordinator {
            card(ui, p, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Generate OTP for next joiner").color(p.text_dim));
                    ui.add_space(8.0);
                    if accent_button(ui, p, "  New OTP  ").clicked() {
                        let url = format!("{node_url}/cluster/otp");
                        let st2 = state_arc.clone();
                        std::thread::spawn(move || {
                            let client = reqwest::blocking::Client::builder()
                                .timeout(Duration::from_secs(5)).build().unwrap();
                            match client.post(&url).send()
                                .and_then(|r| r.json::<serde_json::Value>())
                            {
                                Ok(v) => {
                                    let msg = if let Some(otp) = v["otp"].as_str() {
                                        format!("[otp] {otp}  (single-use, 10 min)")
                                    } else {
                                        format!("[otp] Error: {}", v["error"].as_str().unwrap_or("?"))
                                    };
                                    st2.lock().unwrap().push_log(msg);
                                }
                                Err(e) => { st2.lock().unwrap().push_log(format!("[otp] {e}")); }
                            }
                        });
                    }
                    ui.label(RichText::new("OTP appears in the Log bar below").small().color(p.text_dim));
                });
            });
            ui.add_space(8.0);
        }

        // Peer table
        card(ui, p, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Peers").color(p.accent).strong());
                ui.label(RichText::new(
                    format!("  {} nodes  ·  {} ring slots", s.cluster_nodes.len(), s.ring_slots)
                ).small().color(p.text_dim));
                if !s.cluster_id.is_empty() {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(&s.cluster_id[..8.min(s.cluster_id.len())])
                            .monospace().small().color(p.text_dim));
                        ui.label(RichText::new("id ").small().color(p.text_dim));
                    });
                }
            });
            ui.add_space(4.0);
            egui::Grid::new("peers").num_columns(5).spacing([16.0, 4.0]).striped(true).show(ui, |ui| {
                for lbl in ["ID", "Address", "Role", "CPU", "OS"] {
                    ui.label(RichText::new(lbl).small().color(p.text_dim).strong());
                }
                ui.end_row();
                for peer in &s.cluster_nodes {
                    let id_col = if peer.is_coord { p.success } else { p.text };
                    ui.label(RichText::new(&peer.short_id).monospace().small().color(id_col));
                    ui.label(RichText::new(&peer.addr).monospace().small());
                    ui.label(RichText::new(if peer.is_coord { "coordinator" } else { "worker" })
                        .small().color(if peer.is_coord { p.success } else { p.text_dim }));
                    ui.label(RichText::new(format!("{} cores", peer.cores)).small());
                    ui.label(RichText::new(&peer.os).small());
                    ui.end_row();
                }
            });
        });
    }
}

// ── Neural tab ────────────────────────────────────────────────────────────────

fn tab_neural(ui: &mut Ui, s: &NodeState, p: &Palette) {
    section_title(ui, p, "Neural Fabric");
    card(ui, p, |ui| {
        kv_row(ui, p, "Pools (shards)",  &s.neuro_pools.to_string());
        kv_row(ui, p, "Unique labels",   &s.neuro_labels.to_string());
        kv_row(ui, p, "Active streams",  &s.neuro_streams.len().to_string());
    });
    if !s.neuro_streams.is_empty() {
        ui.add_space(8.0);
        section_title(ui, p, "Streams");
        card(ui, p, |ui| {
            for stream in &s.neuro_streams {
                ui.label(RichText::new(stream).monospace().small().color(p.text_code));
            }
        });
    }
    ui.add_space(8.0);
    ui.label(RichText::new(
        "Training scripts push observations via POST /neuro/train on :8080."
    ).small().color(p.text_dim));
}

// ── Setup tab ─────────────────────────────────────────────────────────────────

fn tab_setup(ui: &mut Ui, p: &Palette, node_url: &str, state_arc: Arc<Mutex<NodeState>>) {
    section_title(ui, p, "Node Setup & Installation");

    ui.label(RichText::new(
        "Use this tab to set up a fresh machine as a W1z4rD V1510n node, \
         open the required firewall ports, and join the network."
    ).small().color(p.text_dim));
    ui.add_space(10.0);

    // Step 1 — Firewall
    card(ui, p, |ui| {
        ui.label(RichText::new("Step 1  —  Open Firewall Ports").color(p.accent).strong());
        ui.add_space(4.0);
        ui.label(RichText::new(
            "The node needs three inbound ports to be reachable on the local network:"
        ).small().color(p.text_dim));
        ui.add_space(4.0);
        egui::Grid::new("ports").num_columns(3).spacing([12.0, 4.0]).show(ui, |ui| {
            for (port, proto, desc) in [
                ("8080", "TCP", "Neuro API — training data, predictions"),
                ("8090", "TCP", "Node API  — cluster mgmt, dashboard"),
                ("51611", "TCP", "SIGIL     — cluster ring / heartbeat"),
            ] {
                ui.label(RichText::new(port).monospace().color(p.text_code));
                ui.label(RichText::new(proto).small().color(p.text_dim));
                ui.label(RichText::new(desc).small());
                ui.end_row();
            }
        });
        ui.add_space(6.0);
        ui.label(RichText::new("Run the installer script as Administrator to apply these rules automatically:").small().color(p.text_dim));
        ui.add_space(4.0);
        ui.label(RichText::new("  scripts\\install_node.ps1").monospace().small().color(p.text_code));
        ui.add_space(6.0);
        if accent_button(ui, p, "  Open PowerShell installer  ").clicked() {
            let _ = std::process::Command::new("powershell")
                .args([
                    "-Command",
                    "Start-Process powershell -Verb RunAs -ArgumentList '-File','scripts\\install_node.ps1'",
                ])
                .spawn();
        }
    });

    ui.add_space(8.0);

    // Step 2 — Cluster
    card(ui, p, |ui| {
        ui.label(RichText::new("Step 2  —  Join the Cluster").color(p.accent).strong());
        ui.add_space(4.0);
        ui.label(RichText::new(
            "After firewall rules are set, go to the Cluster tab to init a new cluster \
             on this machine or join an existing one on your network."
        ).small().color(p.text_dim));
    });

    ui.add_space(8.0);

    // Step 3 — Verify
    card(ui, p, |ui| {
        ui.label(RichText::new("Step 3  —  Verify").color(p.accent).strong());
        ui.add_space(4.0);
        ui.label(RichText::new(
            "The Overview tab shows live status. Both \":8080\" and \":8090\" dots \
             should be green. The Cluster tab should show all peer nodes."
        ).small().color(p.text_dim));
        ui.add_space(6.0);
        if accent_button(ui, p, "  Check node health  ").clicked() {
            let url = format!("{node_url}/health");
            let st2 = state_arc.clone();
            std::thread::spawn(move || {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(5)).build().unwrap();
                match client.get(&url).send().and_then(|r| r.json::<serde_json::Value>()) {
                    Ok(v)  => { st2.lock().unwrap().push_log(format!("[health] {v}")); }
                    Err(e) => { st2.lock().unwrap().push_log(format!("[health] {e}")); }
                }
            });
        }
    });
}

// ── Log tab ───────────────────────────────────────────────────────────────────

fn tab_log(ui: &mut Ui, s: &NodeState, p: &Palette) {
    section_title(ui, p, "Event Log");
    egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
        for line in &s.log {
            ui.label(RichText::new(line).monospace().small().color(p.text_dim));
        }
    });
}

// ── Widget helpers ────────────────────────────────────────────────────────────

fn section_title(ui: &mut Ui, p: &Palette, title: &str) {
    ui.label(RichText::new(title).size(15.0).color(p.accent).strong());
    ui.add(egui::Separator::default().spacing(6.0));
    ui.add_space(4.0);
}

fn card<R>(ui: &mut Ui, p: &Palette, content: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::none()
        .fill(p.card)
        .rounding(Rounding::same(6.0))
        .stroke(Stroke::new(1.0, p.border))
        .inner_margin(Margin::same(12.0))
        .show(ui, content)
        .inner
}

fn kv_row(ui: &mut Ui, p: &Palette, key: &str, val: &str) {
    ui.horizontal(|ui| {
        ui.add_sized(
            Vec2::new(130.0, 16.0),
            egui::Label::new(RichText::new(key).small().color(p.text_dim)),
        );
        ui.label(RichText::new(val).monospace().small().color(p.text));
    });
}

fn accent_button(ui: &mut Ui, p: &Palette, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).color(p.text).small())
            .fill(p.accent_dim)
            .stroke(Stroke::new(1.0, p.accent))
            .rounding(Rounding::same(5.0)),
    )
}

fn fmt_uptime(s: u64) -> String {
    if s < 60       { format!("{s}s") }
    else if s < 3600 { format!("{}m {}s", s / 60, s % 60) }
    else             { format!("{}h {}m", s / 3600, (s % 3600) / 60) }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let cfg = Cfg::from_args();

    let state = Arc::new(Mutex::new(NodeState {
        node_url:     cfg.node_url.clone(),
        api_url:      cfg.api_url.clone(),
        cluster_role: "standalone".into(),
        api_status:   "connecting…".into(),
        ..Default::default()
    }));

    spawn_poller(state.clone());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("W1z4rD V1510n")
            .with_inner_size([860.0, 560.0])
            .with_min_inner_size([540.0, 380.0]),
        ..Default::default()
    };

    eframe::run_native(
        "W1z4rD V1510n",
        options,
        Box::new(|_cc| Ok(Box::new(Dashboard::new(state)))),
    ).map_err(|e| anyhow::anyhow!("{e}"))
}
