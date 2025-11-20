use crate::config::EnergyConfig;
use crate::schema::{DynamicState, Position, Trajectory};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EnergyCalibration {
    pub recommended_motion: f64,
    pub recommended_collision: f64,
    pub recommended_goal: f64,
    pub recommended_group: f64,
}

impl EnergyCalibration {
    pub fn to_energy_config(&self, base: &EnergyConfig) -> EnergyConfig {
        EnergyConfig {
            w_motion: self.recommended_motion,
            w_collision: self.recommended_collision,
            w_goal: self.recommended_goal,
            w_group_cohesion: self.recommended_group,
            ..base.clone()
        }
    }
}

pub fn calibrate_from_trajectories(trajectories: &[Trajectory]) -> EnergyCalibration {
    let mut motion_acc = 0.0;
    let mut motion_samples = 0.0;
    let mut collision_events = 0.0;
    let mut collision_samples = 0.0;
    let mut goal_error = 0.0;
    let mut goal_samples = 0.0;
    let mut group_spread = 0.0;
    let mut group_samples = 0.0;

    for trajectory in trajectories {
        if trajectory.sequence.len() < 2 {
            continue;
        }
        let first = &trajectory.sequence[0];
        let last = trajectory.sequence.last().unwrap();
        for (symbol_id, final_state) in last.symbol_states.iter() {
            if let Some(initial_state) = first.symbol_states.get(symbol_id) {
                motion_acc += distance_squared(&initial_state.position, &final_state.position);
                motion_samples += 1.0;
            }
            if let Some(goal) = final_state
                .internal_state
                .get("goal_position")
                .and_then(value_to_position)
            {
                goal_error += distance(&goal, &final_state.position);
                goal_samples += 1.0;
            }
        }

        for state in &trajectory.sequence {
            collision_samples += 1.0;
            collision_events += count_close_pairs(state);
            if let Some(spread) = average_group_spread(state) {
                group_spread += spread;
                group_samples += 1.0;
            }
        }
    }

    let motion_mean = if motion_samples > 0.0 {
        (motion_acc / motion_samples).max(1e-3)
    } else {
        1.0
    };
    let collision_rate = if collision_samples > 0.0 {
        (collision_events / collision_samples).max(1e-3)
    } else {
        1.0
    };
    let goal_mean = if goal_samples > 0.0 {
        (goal_error / goal_samples).max(1e-3)
    } else {
        1.0
    };
    let group_mean = if group_samples > 0.0 {
        (group_spread / group_samples).max(1e-3)
    } else {
        1.0
    };

    EnergyCalibration {
        recommended_motion: (1.0 / motion_mean).clamp(0.1, 10.0),
        recommended_collision: (1.0 / collision_rate).clamp(0.1, 10.0),
        recommended_goal: (1.0 / goal_mean).clamp(0.1, 10.0),
        recommended_group: (1.0 / group_mean).clamp(0.1, 10.0),
    }
}

fn count_close_pairs(state: &DynamicState) -> f64 {
    let ids: Vec<_> = state.symbol_states.values().collect();
    let mut count = 0.0;
    for (i, first) in ids.iter().enumerate() {
        for second in ids.iter().skip(i + 1) {
            let d = distance(&first.position, &second.position);
            if d < 0.6 {
                count += 1.0;
            }
        }
    }
    count
}

fn average_group_spread(state: &DynamicState) -> Option<f64> {
    let mut spread = 0.0;
    let mut members = 0.0;
    for group_id in ["group_id", "team"] {
        let groups = groups_for_state(state, group_id);
        if groups.is_empty() {
            continue;
        }
        for members_list in groups.values() {
            if members_list.len() < 2 {
                continue;
            }
            let centroid = centroid(members_list);
            for member in members_list {
                spread += distance(&member.position, &centroid);
                members += 1.0;
            }
        }
    }
    if members > 0.0 {
        Some(spread / members)
    } else {
        None
    }
}

fn groups_for_state<'a>(
    state: &'a DynamicState,
    key: &str,
) -> HashMap<String, Vec<&'a crate::schema::SymbolState>> {
    let mut groups: HashMap<String, Vec<&crate::schema::SymbolState>> = HashMap::new();
    for (_symbol_id, symbol_state) in state.symbol_states.iter() {
        if let Some(group_id) = symbol_state
            .internal_state
            .get(key)
            .and_then(|value| value.as_str())
        {
            groups
                .entry(group_id.to_string())
                .or_default()
                .push(symbol_state);
        }
    }
    groups
}

fn centroid(states: &[&crate::schema::SymbolState]) -> Position {
    let n = states.len() as f64;
    let mut sum = Position {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    for state in states {
        sum.x += state.position.x;
        sum.y += state.position.y;
        sum.z += state.position.z;
    }
    Position {
        x: sum.x / n,
        y: sum.y / n,
        z: sum.z / n,
    }
}

fn distance(a: &Position, b: &Position) -> f64 {
    distance_squared(a, b).sqrt()
}

fn distance_squared(a: &Position, b: &Position) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn value_to_position(value: &Value) -> Option<Position> {
    match value {
        Value::Object(map) => {
            let x = map.get("x").and_then(|v| v.as_f64())?;
            let y = map.get("y").and_then(|v| v.as_f64())?;
            let z = map.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some(Position { x, y, z })
        }
        Value::Array(values) if values.len() >= 2 => {
            let x = values.get(0)?.as_f64()?;
            let y = values.get(1)?.as_f64()?;
            let z = values.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some(Position { x, y, z })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{DynamicState, SymbolState, Timestamp};
    use serde_json::json;

    #[test]
    fn calibration_outputs_positive_weights() {
        let mut initial = DynamicState::default();
        let mut final_state = DynamicState::default();
        initial.timestamp = Timestamp { unix: 0 };
        final_state.timestamp = Timestamp { unix: 5 };
        initial.symbol_states.insert(
            "a".into(),
            SymbolState {
                position: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Default::default()
            },
        );
        final_state.symbol_states.insert(
            "a".into(),
            SymbolState {
                position: Position {
                    x: 5.0,
                    y: 0.0,
                    z: 0.0,
                },
                internal_state: {
                    let mut map = HashMap::new();
                    map.insert("goal_position".into(), json!({"x": 6.0, "y": 0.0 }));
                    map
                },
                ..Default::default()
            },
        );
        let traj = Trajectory {
            sequence: vec![initial, final_state],
            metadata: HashMap::new(),
        };
        let calib = calibrate_from_trajectories(&[traj]);
        assert!(calib.recommended_motion > 0.0);
        assert!(calib.recommended_goal > 0.0);
        assert!(calib.recommended_collision > 0.0);
        assert!(calib.recommended_group > 0.0);
    }
}
