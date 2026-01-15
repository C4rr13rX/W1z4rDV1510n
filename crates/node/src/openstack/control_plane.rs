use crate::config::{OpenStackConfig, OpenStackMode};
use crate::openstack::services::{OpenStackNodePlan, OpenStackProvider, ProvisionedNode};
use anyhow::{anyhow, Result};
use std::sync::Mutex;

#[derive(Debug)]
struct LocalState {
    next_id: u64,
}

#[derive(Debug)]
pub struct OpenStackControlPlane {
    config: OpenStackConfig,
    state: Mutex<LocalState>,
}

impl OpenStackControlPlane {
    pub fn new(config: OpenStackConfig) -> Self {
        Self {
            config,
            state: Mutex::new(LocalState { next_id: 1 }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn mode(&self) -> OpenStackMode {
        self.config.mode.clone()
    }
}

impl OpenStackProvider for OpenStackControlPlane {
    fn provision_node(&self, plan: OpenStackNodePlan) -> Result<ProvisionedNode> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("local control plane mutex poisoned"))?;
        let instance_id = format!("local-{}", state.next_id);
        state.next_id = state.next_id.saturating_add(1);
        let public_ip = if plan.network.public_ip {
            Some(format!("127.0.0.{}", (state.next_id % 254).max(1)))
        } else {
            None
        };
        Ok(ProvisionedNode {
            instance_id,
            public_ip,
        })
    }
}
