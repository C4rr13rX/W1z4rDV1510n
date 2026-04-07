/// Spatial threat field tensor.
///
/// The threat field is a 2D grid over the observed environment.  Each cell
/// stores a threat intensity derived from:
///   - Proxemics violations (entity in another's intimate/personal zone)
///   - Gaze / orientation vectors (where entities are looking, intersection)
///   - Behavioural intent signals (from `intent.rs`)
///   - Concealment and control signals (blocking exits, scanning for witnesses)
///
/// The field updates per-frame and accumulates through a temporal EMA so that
/// a spike in one frame doesn't immediately dominate — the field must persist
/// to become significant.
///
/// Proxemics zones (Hall, 1966 — measured in metres):
///   Intimate   0.00–0.45 m  high threat if violated by stranger
///   Personal   0.45–1.20 m  moderate threat if violated
///   Social     1.20–3.70 m  low threat if violated by unknown
///   Public     >3.70 m      baseline

use crate::schema::Timestamp;
use crate::threat::health::HealthDimension;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Proxemics ────────────────────────────────────────────────────────────────

/// Proxemics zone classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxemicsZone {
    Intimate,
    Personal,
    Social,
    Public,
}

impl ProxemicsZone {
    pub fn from_distance_m(dist: f32) -> Self {
        match dist {
            d if d < 0.45 => ProxemicsZone::Intimate,
            d if d < 1.20 => ProxemicsZone::Personal,
            d if d < 3.70 => ProxemicsZone::Social,
            _             => ProxemicsZone::Public,
        }
    }

    /// Base threat multiplier for an unexpected entry into this zone.
    pub fn threat_weight(self) -> f32 {
        match self {
            ProxemicsZone::Intimate => 1.0,
            ProxemicsZone::Personal => 0.6,
            ProxemicsZone::Social   => 0.2,
            ProxemicsZone::Public   => 0.0,
        }
    }
}

// ─── Entity spatial record ────────────────────────────────────────────────────

/// Last-known spatial state of one entity, used for threat field computation.
#[derive(Debug, Clone)]
pub struct EntitySpatialRecord {
    pub entity_id: String,
    /// Position (x, y, z) in metres relative to scene origin.
    pub position: [f32; 3],
    /// Facing direction as unit vector (x, y) in the horizontal plane.
    pub orientation: [f32; 2],
    /// Estimated entity type ("human", "vehicle", "unknown").
    pub entity_type: String,
    /// Whether the entity carries a detected concealed object.
    pub concealed_object: bool,
    /// Gaze velocity (proxy for attentional arousal — high = scanning fast).
    pub gaze_velocity: f32,
    /// Time this record was last updated.
    pub last_seen: Timestamp,
    /// Is this entity stationary?
    pub stationary: bool,
}

// ─── Threat cell ──────────────────────────────────────────────────────────────

/// One cell of the spatial threat grid.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreatCell {
    /// Grid indices.
    pub grid_x: i32,
    pub grid_y: i32,
    /// World position of cell centre.
    pub world_x: f32,
    pub world_y: f32,
    /// Threat intensity [0,1].  0 = safe, 1 = imminent harm predicted.
    pub intensity: f32,
    /// Which entity is the dominant threat source for this cell.
    pub dominant_source: Option<String>,
    /// Which health dimensions are primarily threatened (weights per dim).
    pub dimension_weights: [f32; 6],
    /// Prediction confidence [0,1].
    pub confidence: f32,
    /// How many seconds ahead this prediction extends.
    pub time_horizon_secs: f32,
}

impl ThreatCell {
    /// The single health dimension most threatened in this cell.
    pub fn dominant_dimension(&self) -> HealthDimension {
        let max_idx = self.dimension_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        HealthDimension::ALL[max_idx]
    }

    /// Threat colour as #RRGGBB for overlay rendering.
    /// Uses a red-green gradient: green (safe) → yellow → orange → red (threat).
    pub fn color_hex(&self) -> String {
        let i = self.intensity.clamp(0.0, 1.0);
        let r = (i * 255.0) as u8;
        let g = ((1.0 - i) * 255.0) as u8;
        format!("#{:02X}{:02X}00", r, g)
    }
}

// ─── Threat field ─────────────────────────────────────────────────────────────

/// The full spatial threat field for one observation frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatField {
    pub timestamp: Timestamp,
    /// Non-zero threat cells only (sparse representation).
    pub cells: Vec<ThreatCell>,
    /// Cell resolution in metres.
    pub resolution_m: f32,
    /// Scene bounds [min_x, max_x, min_y, max_y].
    pub bounds: [f32; 4],
    /// Overall peak intensity anywhere in the field.
    pub peak_intensity: f32,
    /// Entity pair that produces the highest proxemics threat.
    pub hotspot_pair: Option<(String, String)>,
}

impl ThreatField {
    pub fn empty(timestamp: Timestamp) -> Self {
        Self {
            timestamp,
            cells: Vec::new(),
            resolution_m: 0.5,
            bounds: [-10.0, 10.0, -10.0, 10.0],
            peak_intensity: 0.0,
            hotspot_pair: None,
        }
    }
}

// ─── Threat field engine ──────────────────────────────────────────────────────

/// Builds and maintains the spatial threat field from entity observations.
#[derive(Debug)]
pub struct ThreatFieldEngine {
    /// Cell size in metres.
    resolution_m: f32,
    /// Scene bounds [min_x, max_x, min_y, max_y].
    bounds: [f32; 4],
    /// Temporal smoothing — how much of the previous field is retained.
    temporal_alpha: f32,
    /// Previous frame's cell intensities for EMA smoothing.
    prev_intensities: HashMap<(i32, i32), f32>,
    /// Spatial records per entity, updated each frame.
    entity_records: HashMap<String, EntitySpatialRecord>,
}

impl Default for ThreatFieldEngine {
    fn default() -> Self {
        Self {
            resolution_m: 0.5,
            bounds: [-10.0, 10.0, -10.0, 10.0],
            temporal_alpha: 0.4, // 40% new, 60% retained — threat field is sticky
            prev_intensities: HashMap::new(),
            entity_records: HashMap::new(),
        }
    }
}

impl ThreatFieldEngine {
    pub fn new(resolution_m: f32, bounds: [f32; 4]) -> Self {
        Self { resolution_m, bounds, ..Self::default() }
    }

    /// Update one entity's spatial record.
    pub fn update_entity(&mut self, record: EntitySpatialRecord) {
        self.entity_records.insert(record.entity_id.clone(), record);
    }

    /// Update from token batch attributes (called from streaming processor).
    pub fn ingest_from_attributes(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        timestamp: Timestamp,
        attrs: &HashMap<String, serde_json::Value>,
    ) {
        let px = attr_f32(attrs, "pos_x").unwrap_or(0.0);
        let py = attr_f32(attrs, "pos_y").unwrap_or(0.0);
        let pz = attr_f32(attrs, "pos_z").unwrap_or(0.0);
        let ox = attr_f32(attrs, "orient_x").unwrap_or(0.0);
        let oy = attr_f32(attrs, "orient_y").unwrap_or(1.0);
        let concealed = attrs.get("concealed_object")
            .and_then(|v| v.as_bool()).unwrap_or(false);
        let gaze_vel = attr_f32(attrs, "gaze_velocity").unwrap_or(0.0);
        let stationary = attr_f32(attrs, "motion_energy").unwrap_or(1.0) < 0.05;
        let rec = EntitySpatialRecord {
            entity_id: entity_id.to_string(),
            position: [px, py, pz],
            orientation: [ox, oy],
            entity_type: entity_type.to_string(),
            concealed_object: concealed,
            gaze_velocity: gaze_vel,
            last_seen: timestamp,
            stationary,
        };
        self.update_entity(rec);
    }

    /// Compute the threat field for the current set of entity records.
    pub fn compute(&mut self, timestamp: Timestamp, intent_map: &HashMap<String, f32>) -> ThreatField {
        if self.entity_records.is_empty() {
            return ThreatField::empty(timestamp);
        }

        // Compute pairwise proxemics threats.
        let entity_ids: Vec<String> = self.entity_records.keys().cloned().collect();
        let mut cell_threats: HashMap<(i32, i32), (f32, String, [f32; 6])> = HashMap::new();
        let mut hotspot_pair: Option<(String, String)> = None;
        let mut hotspot_intensity = 0.0f32;

        for i in 0..entity_ids.len() {
            for j in (i + 1)..entity_ids.len() {
                let id_a = &entity_ids[i];
                let id_b = &entity_ids[j];
                let rec_a = &self.entity_records[id_a];
                let rec_b = &self.entity_records[id_b];

                let dist = distance2d(rec_a.position, rec_b.position);
                let zone = ProxemicsZone::from_distance_m(dist);
                let prox_threat = zone.threat_weight();

                // Orientation convergence: are they facing each other?
                let convergence = orientation_convergence(
                    rec_a.position, rec_a.orientation,
                    rec_b.position, rec_b.orientation,
                );

                // Intent modifier from intent engine output.
                let intent_a = intent_map.get(id_a).copied().unwrap_or(0.0);
                let intent_b = intent_map.get(id_b).copied().unwrap_or(0.0);
                let intent_factor = (intent_a + intent_b) * 0.5;

                // Concealment bonus.
                let conceal_bonus = if rec_a.concealed_object || rec_b.concealed_object { 0.3 } else { 0.0 };

                let raw_threat = (prox_threat * 0.4
                    + convergence * 0.3
                    + intent_factor * 0.2
                    + conceal_bonus * 0.1)
                    .clamp(0.0, 1.0);

                if raw_threat < 0.02 { continue; }

                // Which entity is the threat SOURCE (higher intent = source).
                let (source, target) = if intent_a >= intent_b { (id_a, id_b) } else { (id_b, id_a) };

                // Which dimensions are threatened at the target.
                let dim_weights = threatened_dimensions(zone, rec_a.concealed_object || rec_b.concealed_object);

                // Spread threat over cells near the TARGET position.
                let target_rec = &self.entity_records[target];
                self.spread_threat(target_rec.position, raw_threat, source.clone(), dim_weights, &mut cell_threats);

                if raw_threat > hotspot_intensity {
                    hotspot_intensity = raw_threat;
                    hotspot_pair = Some((source.clone(), target.clone()));
                }
            }
        }

        // Apply temporal EMA.
        let mut cells: Vec<ThreatCell> = Vec::new();
        let mut peak_intensity = 0.0f32;
        for (&(gx, gy), &(raw_intensity, ref source, ref dim_weights)) in cell_threats.iter() {
            let prev = self.prev_intensities.get(&(gx, gy)).copied().unwrap_or(0.0);
            let intensity = self.temporal_alpha * raw_intensity + (1.0 - self.temporal_alpha) * prev;
            if intensity < 0.01 { continue; }
            peak_intensity = peak_intensity.max(intensity);
            let wx = self.bounds[0] + gx as f32 * self.resolution_m + self.resolution_m * 0.5;
            let wy = self.bounds[2] + gy as f32 * self.resolution_m + self.resolution_m * 0.5;
            cells.push(ThreatCell {
                grid_x: gx,
                grid_y: gy,
                world_x: wx,
                world_y: wy,
                intensity,
                dominant_source: Some(source.clone()),
                dimension_weights: *dim_weights,
                confidence: intensity,
                time_horizon_secs: predict_horizon(intensity),
            });
        }

        // Persist for next frame.
        self.prev_intensities = cells.iter()
            .map(|c| ((c.grid_x, c.grid_y), c.intensity))
            .collect();

        ThreatField {
            timestamp,
            cells,
            resolution_m: self.resolution_m,
            bounds: self.bounds,
            peak_intensity,
            hotspot_pair,
        }
    }

    fn spread_threat(
        &self,
        pos: [f32; 3],
        intensity: f32,
        source: String,
        dim_weights: [f32; 6],
        cell_threats: &mut HashMap<(i32, i32), (f32, String, [f32; 6])>,
    ) {
        let radius = 2.0f32; // spread over ~2m radius
        let gx0 = world_to_grid(pos[0], self.bounds[0], self.resolution_m);
        let gy0 = world_to_grid(pos[1], self.bounds[2], self.resolution_m);
        let r_cells = (radius / self.resolution_m).ceil() as i32;
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                let gx = gx0 + dx;
                let gy = gy0 + dy;
                let wx = self.bounds[0] + gx as f32 * self.resolution_m;
                let wy = self.bounds[2] + gy as f32 * self.resolution_m;
                let dist = ((wx - pos[0]).powi(2) + (wy - pos[1]).powi(2)).sqrt();
                let falloff = (1.0 - dist / radius).max(0.0);
                let cell_intensity = intensity * falloff;
                if cell_intensity < 0.01 { continue; }
                let entry = cell_threats.entry((gx, gy)).or_insert((0.0, source.clone(), [0.0; 6]));
                if cell_intensity > entry.0 {
                    *entry = (cell_intensity, source.clone(), dim_weights);
                }
            }
        }
    }

    pub fn entity_count(&self) -> usize { self.entity_records.len() }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

fn attr_f32(attrs: &HashMap<String, serde_json::Value>, key: &str) -> Option<f32> {
    attrs.get(key)?.as_f64().filter(|v| v.is_finite()).map(|v| v as f32)
}

fn distance2d(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// How much are two entities converging on each other?
/// Returns [0,1] where 1 = both facing each other directly.
fn orientation_convergence(
    pos_a: [f32; 3], orient_a: [f32; 2],
    pos_b: [f32; 3], orient_b: [f32; 2],
) -> f32 {
    let dx = pos_b[0] - pos_a[0];
    let dy = pos_b[1] - pos_a[1];
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let dir_ab = [dx / len, dy / len];
    let dir_ba = [-dir_ab[0], -dir_ab[1]];
    // Dot products: how much each entity faces the other.
    let dot_a = (orient_a[0] * dir_ab[0] + orient_a[1] * dir_ab[1]).clamp(-1.0, 1.0);
    let dot_b = (orient_b[0] * dir_ba[0] + orient_b[1] * dir_ba[1]).clamp(-1.0, 1.0);
    // Both facing each other = high convergence.
    ((dot_a + dot_b) * 0.5).max(0.0)
}

/// Which health dimensions are primarily threatened given the situation.
fn threatened_dimensions(zone: ProxemicsZone, concealed: bool) -> [f32; 6] {
    // Index: 0=SI, 1=EF, 2=RC, 3=FO, 4=AR, 5=TC
    let mut w = [0.0f32; 6];
    match zone {
        ProxemicsZone::Intimate => {
            w[0] = 1.0; // SI — direct physical access
            w[1] = 0.7; // EF — potential interruption of energy/blood
            w[3] = 0.5; // FO — functional capacity threatened
            w[4] = 0.8; // AR — reserve critically stressed at intimate range
        }
        ProxemicsZone::Personal => {
            w[0] = 0.5;
            w[4] = 0.6;
            w[2] = 0.4; // RC — regulatory response activated
            w[5] = 0.5; // TC — rhythm disrupted
        }
        ProxemicsZone::Social => {
            w[2] = 0.3;
            w[5] = 0.3;
            w[4] = 0.2;
        }
        ProxemicsZone::Public => {} // no dimensional threat
    }
    if concealed {
        // Concealed object amplifies SI and FO threats.
        w[0] = (w[0] * 1.5).min(1.0);
        w[3] = (w[3] * 1.3).min(1.0);
    }
    // Normalize so max is 1.0.
    let max = w.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 { for v in &mut w { *v /= max; } }
    w
}

/// Predict time horizon (seconds ahead) from threat intensity.
fn predict_horizon(intensity: f32) -> f32 {
    // High intensity = imminent (short horizon); low = distant.
    (60.0 * (1.0 - intensity)).max(5.0)
}

fn world_to_grid(world: f32, min: f32, resolution: f32) -> i32 {
    ((world - min) / resolution).floor() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxemics_zones_correct() {
        assert_eq!(ProxemicsZone::from_distance_m(0.3), ProxemicsZone::Intimate);
        assert_eq!(ProxemicsZone::from_distance_m(0.8), ProxemicsZone::Personal);
        assert_eq!(ProxemicsZone::from_distance_m(2.0), ProxemicsZone::Social);
        assert_eq!(ProxemicsZone::from_distance_m(5.0), ProxemicsZone::Public);
    }

    #[test]
    fn threat_field_two_entities_close() {
        let mut engine = ThreatFieldEngine::default();
        let ts = Timestamp { unix: 0 };
        engine.update_entity(EntitySpatialRecord {
            entity_id: "clerk".to_string(),
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 1.0],
            entity_type: "human".to_string(),
            concealed_object: false,
            gaze_velocity: 0.1,
            last_seen: ts,
            stationary: true,
        });
        engine.update_entity(EntitySpatialRecord {
            entity_id: "robber".to_string(),
            position: [0.3, 0.0, 0.0], // intimate zone
            orientation: [0.0, -1.0],
            entity_type: "human".to_string(),
            concealed_object: true,
            gaze_velocity: 0.8,
            last_seen: ts,
            stationary: false,
        });
        let mut intent_map = HashMap::new();
        intent_map.insert("robber".to_string(), 0.9);
        let field = engine.compute(ts, &intent_map);
        assert!(field.peak_intensity > 0.3, "close concealed entity should produce threat");
        assert!(field.hotspot_pair.is_some());
    }
}
