use crate::schema::Timestamp;
use crate::spike::{SpikeConfig, SpikeFrame, SpikeInput, SpikeMessage, SpikeMessageBus, SpikePool};
use crate::streaming::schema::{LayerKind, LayerState};
use std::collections::HashMap;

pub struct UltradianSpikeRouter {
    pool: SpikePool,
    neuron_map: HashMap<LayerKind, u32>,
}

impl UltradianSpikeRouter {
    pub fn new(id: impl Into<String>, config: SpikeConfig) -> Self {
        Self {
            pool: SpikePool::new(id, config),
            neuron_map: HashMap::new(),
        }
    }

    pub fn route_layers(&mut self, layers: &[LayerState]) -> SpikeFrame {
        let mut inputs = Vec::new();
        let timestamp = latest_timestamp(layers);
        for layer in layers {
            if !is_ultradian(layer.kind) {
                continue;
            }
            let target = self.neuron_for(layer.kind);
            let excitatory =
                (layer.amplitude * (1.0 + layer.coherence)).clamp(0.0, 4.0) as f32;
            let inhibitory = (1.0 - layer.coherence).clamp(0.0, 1.0) as f32;
            inputs.push(SpikeInput {
                target,
                excitatory,
                inhibitory,
            });
        }
        self.pool.enqueue_inputs(inputs);
        self.pool.step(timestamp)
    }

    pub fn route_layers_with_bus(
        &mut self,
        layers: &[LayerState],
        bus: &mut SpikeMessageBus,
    ) -> SpikeFrame {
        let frame = self.route_layers(layers);
        let message = SpikeMessage {
            pool_id: self.pool.id.clone(),
            frame: frame.clone(),
        };
        bus.publish(message);
        frame
    }

    fn neuron_for(&mut self, kind: LayerKind) -> u32 {
        if let Some(id) = self.neuron_map.get(&kind).copied() {
            return id;
        }
        let neuron = self.pool.add_neuron(crate::spike::NeuronKind::Excitatory);
        self.neuron_map.insert(kind, neuron);
        neuron
    }
}

fn is_ultradian(kind: LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::UltradianMicroArousal
            | LayerKind::UltradianBrac
            | LayerKind::UltradianMeso
    )
}

fn latest_timestamp(layers: &[LayerState]) -> Timestamp {
    let mut latest = Timestamp { unix: 0 };
    for layer in layers {
        if layer.timestamp.unix > latest.unix {
            latest = layer.timestamp;
        }
    }
    latest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_spikes_on_ultradian_layer() {
        let mut router = UltradianSpikeRouter::new("ultra", SpikeConfig::default());
        let layers = vec![LayerState {
            kind: LayerKind::UltradianMicroArousal,
            timestamp: Timestamp { unix: 10 },
            phase: 0.1,
            amplitude: 1.2,
            coherence: 0.9,
            attributes: HashMap::new(),
        }];
        let frame = router.route_layers(&layers);
        assert!(!frame.spikes.is_empty());
    }

    #[test]
    fn router_ignores_non_ultradian_layers() {
        let mut router = UltradianSpikeRouter::new("ultra", SpikeConfig::default());
        let layers = vec![LayerState {
            kind: LayerKind::FlowDensity,
            timestamp: Timestamp { unix: 10 },
            phase: 0.0,
            amplitude: 1.0,
            coherence: 0.0,
            attributes: HashMap::new(),
        }];
        let frame = router.route_layers(&layers);
        assert!(frame.spikes.is_empty());
    }
}
