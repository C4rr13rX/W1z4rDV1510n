//! Robot body morphology and physical skeletal frame control module.
//!
//! Enables `w1z4rd-brain` to control 3D embodied physical agents in virtual simulation
//! (e.g. Multiscale Robot World) and transfer learned motor and sensory representations
//! directly to real-world robots across humanoid, quadruped, differential drive,
//! and articulated manipulator bodies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::multisensory::{KinematicProprioceptionEncoder, MultisensoryExperience, SenseModality};
use crate::neuron::PoolId;

/// Morphological taxonomy of embodied robotic agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphologyType {
    /// Humanoid bipedal skeletal frame with biological DoFs.
    Humanoid,
    /// Quadrupedal 4-legged walking chassis.
    Quadruped,
    /// Wheeled mobile robot with differential steering.
    DifferentialDrive,
    /// Articulated multi-axis robotic arm manipulator.
    ArticulatedArm,
}

impl MorphologyType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Humanoid => "humanoid",
            Self::Quadruped => "quadruped",
            Self::DifferentialDrive => "differential_drive",
            Self::ArticulatedArm => "articulated_arm",
        }
    }
}

/// Physical parameters and kinematic limits of a single joint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointDescriptor {
    pub joint_id: String,
    pub parent_id: Option<String>,
    pub min_angle_rad: f32,
    pub max_angle_rad: f32,
    pub max_torque_nm: f32,
    pub max_velocity_rad_s: f32,
    pub current_angle_rad: f32,
    pub current_velocity_rad_s: f32,
    pub current_torque_nm: f32,
    pub target_angle_rad: f32,
}

impl JointDescriptor {
    pub fn new(
        joint_id: impl Into<String>,
        parent_id: Option<&str>,
        min_angle_rad: f32,
        max_angle_rad: f32,
        max_torque_nm: f32,
        max_velocity_rad_s: f32,
    ) -> Self {
        Self {
            joint_id: joint_id.into(),
            parent_id: parent_id.map(|s| s.to_string()),
            min_angle_rad,
            max_angle_rad,
            max_torque_nm,
            max_velocity_rad_s,
            current_angle_rad: 0.0,
            current_velocity_rad_s: 0.0,
            current_torque_nm: 0.0,
            target_angle_rad: 0.0,
        }
    }
}

/// Embodied robot body model with skeletal joints and physical properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotBodyModel {
    pub morphology: MorphologyType,
    pub joints: HashMap<String, JointDescriptor>,
    pub joint_order: Vec<String>,
    pub total_mass_kg: f32,
    pub total_height_m: f32,
}

impl RobotBodyModel {
    pub fn new(morphology: MorphologyType, total_mass_kg: f32, total_height_m: f32) -> Self {
        Self {
            morphology,
            joints: HashMap::new(),
            joint_order: Vec::new(),
            total_mass_kg,
            total_height_m,
        }
    }

    /// Add a joint descriptor to the kinematic chain.
    pub fn add_joint(&mut self, joint: JointDescriptor) {
        let id = joint.joint_id.clone();
        if !self.joints.contains_key(&id) {
            self.joint_order.push(id.clone());
        }
        self.joints.insert(id, joint);
    }

    /// Factory for a calibrated standard Humanoid Robot (18 biological DoFs).
    pub fn humanoid_standard() -> Self {
        let mut model = Self::new(MorphologyType::Humanoid, 72.0, 1.75);

        // Core and Spine
        model.add_joint(JointDescriptor::new("pelvis", None, -0.5, 0.5, 250.0, 3.0));
        model.add_joint(JointDescriptor::new("spine_lower", Some("pelvis"), -0.6, 0.6, 200.0, 3.0));
        model.add_joint(JointDescriptor::new("spine_upper", Some("spine_lower"), -0.5, 0.5, 180.0, 3.0));
        model.add_joint(JointDescriptor::new("neck", Some("spine_upper"), -0.8, 0.8, 40.0, 5.0));
        model.add_joint(JointDescriptor::new("head", Some("neck"), -0.7, 0.7, 30.0, 5.0));

        // Upper Limbs (Left)
        model.add_joint(JointDescriptor::new("shoulder_l", Some("spine_upper"), -1.57, 3.14, 80.0, 4.0));
        model.add_joint(JointDescriptor::new("elbow_l", Some("shoulder_l"), 0.0, 2.50, 50.0, 4.5));
        model.add_joint(JointDescriptor::new("wrist_l", Some("elbow_l"), -1.2, 1.2, 25.0, 6.0));
        model.add_joint(JointDescriptor::new("hand_l", Some("wrist_l"), -0.5, 0.5, 15.0, 6.0));

        // Upper Limbs (Right)
        model.add_joint(JointDescriptor::new("shoulder_r", Some("spine_upper"), -1.57, 3.14, 80.0, 4.0));
        model.add_joint(JointDescriptor::new("elbow_r", Some("shoulder_r"), 0.0, 2.50, 50.0, 4.5));
        model.add_joint(JointDescriptor::new("wrist_r", Some("elbow_r"), -1.2, 1.2, 25.0, 6.0));
        model.add_joint(JointDescriptor::new("hand_r", Some("wrist_r"), -0.5, 0.5, 15.0, 6.0));

        // Lower Limbs (Left)
        model.add_joint(JointDescriptor::new("hip_l", Some("pelvis"), -1.57, 0.8, 200.0, 3.5));
        model.add_joint(JointDescriptor::new("knee_l", Some("hip_l"), 0.0, 2.45, 160.0, 4.0));
        model.add_joint(JointDescriptor::new("ankle_l", Some("knee_l"), -0.8, 0.6, 120.0, 4.0));

        // Lower Limbs (Right)
        model.add_joint(JointDescriptor::new("hip_r", Some("pelvis"), -1.57, 0.8, 200.0, 3.5));
        model.add_joint(JointDescriptor::new("knee_r", Some("hip_r"), 0.0, 2.45, 160.0, 4.0));
        model.add_joint(JointDescriptor::new("ankle_r", Some("knee_r"), -0.8, 0.6, 120.0, 4.0));

        model
    }

    /// Factory for a standard Quadruped Robot (12 DoF).
    pub fn quadruped_standard() -> Self {
        let mut model = Self::new(MorphologyType::Quadruped, 32.0, 0.65);
        model.add_joint(JointDescriptor::new("torso", None, -0.3, 0.3, 180.0, 3.0));

        for leg in &["fl", "fr", "rl", "rr"] {
            model.add_joint(JointDescriptor::new(format!("hip_{leg}"), Some("torso"), -0.8, 0.8, 90.0, 5.0));
            model.add_joint(JointDescriptor::new(format!("thigh_{leg}"), Some(&format!("hip_{leg}")), -1.2, 1.2, 90.0, 5.0));
            model.add_joint(JointDescriptor::new(format!("calf_{leg}"), Some(&format!("thigh_{leg}")), 0.0, 2.2, 70.0, 6.0));
        }

        model
    }

    /// Factory for a Differential Drive mobile base.
    pub fn differential_drive_standard() -> Self {
        let mut model = Self::new(MorphologyType::DifferentialDrive, 15.0, 0.25);
        model.add_joint(JointDescriptor::new("wheel_left", None, -1000.0, 1000.0, 30.0, 15.0));
        model.add_joint(JointDescriptor::new("wheel_right", None, -1000.0, 1000.0, 30.0, 15.0));
        model
    }

    /// Factory for a 6-DoF Articulated Manipulator Arm.
    pub fn articulated_arm_standard() -> Self {
        let mut model = Self::new(MorphologyType::ArticulatedArm, 22.0, 0.95);
        model.add_joint(JointDescriptor::new("base_yaw", None, -3.14, 3.14, 150.0, 3.0));
        model.add_joint(JointDescriptor::new("shoulder_pitch", Some("base_yaw"), -1.8, 1.8, 150.0, 2.5));
        model.add_joint(JointDescriptor::new("elbow_pitch", Some("shoulder_pitch"), -2.2, 2.2, 80.0, 3.0));
        model.add_joint(JointDescriptor::new("forearm_roll", Some("elbow_pitch"), -3.14, 3.14, 40.0, 5.0));
        model.add_joint(JointDescriptor::new("wrist_pitch", Some("forearm_roll"), -1.8, 1.8, 25.0, 5.0));
        model.add_joint(JointDescriptor::new("wrist_yaw", Some("wrist_pitch"), -3.14, 3.14, 15.0, 6.0));
        model.add_joint(JointDescriptor::new("gripper", Some("wrist_yaw"), 0.0, 0.08, 40.0, 2.0));
        model
    }

    /// Sets commanded joint target angle and torque with physical bounds clamping.
    pub fn set_joint_target(&mut self, joint_id: &str, target_angle_rad: f32, torque_nm: Option<f32>) -> bool {
        if let Some(dof) = self.joints.get_mut(joint_id) {
            dof.target_angle_rad = target_angle_rad.clamp(dof.min_angle_rad, dof.max_angle_rad);
            if let Some(t) = torque_nm {
                dof.current_torque_nm = t.clamp(-dof.max_torque_nm, dof.max_torque_nm);
            }
            true
        } else {
            false
        }
    }

    /// Advances physics simulation by dt seconds, moving joints toward targets under velocity limits.
    pub fn step_physics(&mut self, dt_seconds: f32) {
        for dof in self.joints.values_mut() {
            let diff = dof.target_angle_rad - dof.current_angle_rad;
            let max_step = dof.max_velocity_rad_s * dt_seconds;
            let step = diff.signum() * diff.abs().min(max_step);

            dof.current_angle_rad += step;
            dof.current_velocity_rad_s = if dt_seconds > 0.0 { step / dt_seconds } else { 0.0 };
        }
    }

    /// Serializes joint state into normalized 16-bit integer kinematics vector [angle_norm, torque_norm, ...].
    pub fn get_kinematic_state_vector(&self) -> Vec<i16> {
        let mut vector = Vec::with_capacity(self.joint_order.len() * 2);
        for id in &self.joint_order {
            if let Some(dof) = self.joints.get(id) {
                let (q_angle, q_torque) = KinematicProprioceptionEncoder::quantize(
                    dof.current_angle_rad,
                    dof.current_torque_nm,
                    dof.max_torque_nm,
                );
                vector.push(q_angle);
                vector.push(q_torque);
            }
        }
        vector
    }

    /// Actuates joints from a normalized 16-bit brain command vector.
    pub fn actuate_from_brain_vector(&mut self, vector: &[i16]) {
        let order = self.joint_order.clone();
        let mut idx = 0;
        for id in &order {
            if idx >= vector.len() {
                break;
            }
            let norm_angle = (vector[idx] as f32) / 32767.0;
            let target_rad = norm_angle * std::f32::consts::PI;
            self.set_joint_target(id, target_rad, None);
            idx += 2; // skip torque in position command
        }
    }

    /// Converts current body state into a tick-aligned MultisensoryExperience for brain ingestion.
    pub fn to_multisensory_experience(&self, timestamp_ms: u64, pool_id: PoolId) -> MultisensoryExperience {
        let state_vector = self.get_kinematic_state_vector();
        let mut raw_bytes = Vec::with_capacity(state_vector.len() * 2);
        for &val in &state_vector {
            raw_bytes.extend_from_slice(&val.to_le_bytes());
        }

        MultisensoryExperience::new(timestamp_ms)
            .with_channel(pool_id, SenseModality::Proprioceptive, raw_bytes)
            .with_context(format!("{}:body_kinematics", self.morphology.name()))
    }
}
