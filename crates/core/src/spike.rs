use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NeuronKind {
    Excitatory,
    Inhibitory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpikeConfig {
    pub threshold: f32,
    pub membrane_decay: f32,
    pub refractory_steps: u32,
}

impl Default for SpikeConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            membrane_decay: 0.95,
            refractory_steps: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpikeSynapse {
    pub target: u32,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    pub id: u32,
    pub kind: NeuronKind,
    pub membrane: f32,
    pub refractory_left: u32,
    pub threshold: f32,
    pub outgoing: Vec<SpikeSynapse>,
}

impl SpikingNeuron {
    fn new(id: u32, kind: NeuronKind, threshold: f32) -> Self {
        Self {
            id,
            kind,
            membrane: 0.0,
            refractory_left: 0,
            threshold,
            outgoing: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpikeInput {
    pub target: u32,
    pub excitatory: f32,
    pub inhibitory: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeFiring {
    pub neuron_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeFrame {
    pub step: u64,
    pub timestamp: Timestamp,
    pub spikes: Vec<SpikeFiring>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeMessage {
    pub pool_id: String,
    pub frame: SpikeFrame,
}

#[derive(Debug, Default)]
pub struct SpikeMessageBus {
    queue: Vec<SpikeMessage>,
}

impl SpikeMessageBus {
    pub fn publish(&mut self, message: SpikeMessage) {
        self.queue.push(message);
    }

    pub fn drain(&mut self) -> Vec<SpikeMessage> {
        std::mem::take(&mut self.queue)
    }
}

pub struct SpikePool {
    pub id: String,
    config: SpikeConfig,
    neurons: Vec<SpikingNeuron>,
    pending_inputs: Vec<SpikeInput>,
    step: u64,
}

impl SpikePool {
    pub fn new(id: impl Into<String>, config: SpikeConfig) -> Self {
        Self {
            id: id.into(),
            config,
            neurons: Vec::new(),
            pending_inputs: Vec::new(),
            step: 0,
        }
    }

    pub fn add_neuron(&mut self, kind: NeuronKind) -> u32 {
        let id = self.neurons.len() as u32;
        let neuron = SpikingNeuron::new(id, kind, self.config.threshold);
        self.neurons.push(neuron);
        id
    }

    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn connect(&mut self, from: u32, to: u32, weight: f32) {
        if let Some(neuron) = self.neurons.get_mut(from as usize) {
            neuron.outgoing.push(SpikeSynapse { target: to, weight });
        }
    }

    pub fn enqueue_inputs<I>(&mut self, inputs: I)
    where
        I: IntoIterator<Item = SpikeInput>,
    {
        self.pending_inputs.extend(inputs);
    }

    pub fn step(&mut self, timestamp: Timestamp) -> SpikeFrame {
        let mut excitatory = HashMap::new();
        let mut inhibitory = HashMap::new();
        for input in self.pending_inputs.drain(..) {
            *excitatory.entry(input.target).or_insert(0.0) += input.excitatory;
            *inhibitory.entry(input.target).or_insert(0.0) += input.inhibitory;
        }
        let mut spikes = Vec::new();
        let mut next_inputs = Vec::new();
        for neuron in &mut self.neurons {
            if neuron.refractory_left > 0 {
                neuron.refractory_left -= 1;
                neuron.membrane = 0.0;
                continue;
            }
            let e = excitatory.remove(&neuron.id).unwrap_or(0.0);
            let i = inhibitory.remove(&neuron.id).unwrap_or(0.0);
            let drive = e - i;
            neuron.membrane = neuron.membrane * self.config.membrane_decay + drive;
            if neuron.membrane >= neuron.threshold {
                neuron.membrane = 0.0;
                neuron.refractory_left = self.config.refractory_steps;
                spikes.push(SpikeFiring { neuron_id: neuron.id });
                let is_inhibitory = matches!(neuron.kind, NeuronKind::Inhibitory);
                for syn in &neuron.outgoing {
                    let (e_out, i_out) = if is_inhibitory {
                        (0.0, syn.weight)
                    } else {
                        (syn.weight, 0.0)
                    };
                    next_inputs.push(SpikeInput {
                        target: syn.target,
                        excitatory: e_out,
                        inhibitory: i_out,
                    });
                }
            }
        }
        self.pending_inputs = next_inputs;
        let frame = SpikeFrame {
            step: self.step,
            timestamp,
            spikes,
        };
        self.step = self.step.saturating_add(1);
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_spikes_and_enters_refractory() {
        let mut pool = SpikePool::new("test", SpikeConfig::default());
        let n0 = pool.add_neuron(NeuronKind::Excitatory);
        pool.enqueue_inputs([SpikeInput {
            target: n0,
            excitatory: 1.5,
            inhibitory: 0.0,
        }]);
        let frame = pool.step(Timestamp { unix: 0 });
        assert_eq!(frame.spikes.len(), 1);
        assert_eq!(frame.spikes[0].neuron_id, n0);

        pool.enqueue_inputs([SpikeInput {
            target: n0,
            excitatory: 1.5,
            inhibitory: 0.0,
        }]);
        let frame2 = pool.step(Timestamp { unix: 1 });
        assert!(
            frame2.spikes.is_empty(),
            "neuron should be refractory"
        );
    }
}
