use crate::energy::{EnergyBreakdown, EnergyModel};
use crate::schema::{DynamicState, EnvironmentSnapshot, Population};
use crate::search::{PathResult, SearchModule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    pub best_state: DynamicState,
    pub best_energy: f64,
    pub clusters: Vec<ClusterSummary>,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSummary {
    pub representative_state: DynamicState,
    pub cluster_weight: f64,
    pub mean_energy: f64,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    pub energy_trace: Vec<f64>,
    pub diversity_metric: f64,
    #[serde(default)]
    pub best_state_breakdown: Option<EnergyBreakdown>,
    #[serde(default)]
    pub path_report: Option<PathDiagnosticsReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathDiagnosticsReport {
    pub per_symbol: HashMap<String, SymbolPathDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolPathDiagnostics {
    pub feasible: bool,
    #[serde(default)]
    pub failure_reason: Option<String>,
    pub path_length: f64,
    pub cost: f64,
    pub visited_nodes: usize,
    pub constraint_violations: usize,
}

pub fn analyze_results(
    population: &Population,
    energy_trace: Vec<f64>,
    energy_model: &EnergyModel,
    search_module: Option<&SearchModule>,
    snapshot_0: &EnvironmentSnapshot,
) -> Results {
    let mut sorted = population.particles.clone();
    sorted.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
    let best_particle = sorted.first().expect("Population cannot be empty");
    let clusters = sorted
        .iter()
        .take(3)
        .map(|particle| ClusterSummary {
            representative_state: particle.current_state.clone(),
            cluster_weight: particle.weight,
            mean_energy: particle.energy,
            size: 1,
        })
        .collect();
    let diversity_metric = compute_diversity(population);
    let breakdown = energy_model.energy_breakdown(&best_particle.current_state);
    let path_report = search_module.map(|module| {
        compute_path_report(module, snapshot_0, &best_particle.current_state)
    });
    Results {
        best_state: best_particle.current_state.clone(),
        best_energy: best_particle.energy,
        clusters,
        diagnostics: Diagnostics {
            energy_trace,
            diversity_metric,
            best_state_breakdown: Some(breakdown),
            path_report,
        },
    }
}

fn compute_diversity(population: &Population) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for (i, a) in population.particles.iter().enumerate() {
        for b in population.particles.iter().skip(i + 1) {
            let dist = spatial_distance(
                &a.current_state
                    .symbol_states
                    .values()
                    .next()
                    .map(|s| s.position)
                    .unwrap_or_default(),
                &b.current_state
                    .symbol_states
                    .values()
                    .next()
                    .map(|s| s.position)
                    .unwrap_or_default(),
            );
            total += dist;
            count += 1.0;
        }
    }
    if count == 0.0 { 0.0 } else { total / count }
}

fn spatial_distance(a: &crate::schema::Position, b: &crate::schema::Position) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn compute_path_report(
    search: &SearchModule,
    snapshot_0: &EnvironmentSnapshot,
    state: &DynamicState,
) -> PathDiagnosticsReport {
    let mut per_symbol = HashMap::new();
    let paths = search.compute_paths(snapshot_0, state);
    for (symbol_id, path) in paths {
        let entry = match path {
            PathResult::Feasible { diagnostics, .. } => SymbolPathDiagnostics {
                feasible: true,
                failure_reason: None,
                path_length: diagnostics.length,
                cost: diagnostics.cost,
                visited_nodes: diagnostics.visited_nodes,
                constraint_violations: diagnostics.constraint_violations,
            },
            PathResult::Infeasible { reason, diagnostics } => SymbolPathDiagnostics {
                feasible: false,
                failure_reason: Some(format!("{reason:?}")),
                path_length: diagnostics.length,
                cost: diagnostics.cost,
                visited_nodes: diagnostics.visited_nodes,
                constraint_violations: diagnostics.constraint_violations,
            },
        };
        per_symbol.insert(symbol_id, entry);
    }
    PathDiagnosticsReport { per_symbol }
}
