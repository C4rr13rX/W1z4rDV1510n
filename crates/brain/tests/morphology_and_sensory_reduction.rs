use w1z4rd_brain::{
    Audio16BitQuantizer, CorticalVisualEncoder, KinematicProprioceptionEncoder,
    MorphologyType, RobotBodyModel, SenseModality,
};

#[test]
fn test_humanoid_body_morphology_and_physical_constraints() {
    let mut humanoid = RobotBodyModel::humanoid_standard();

    assert_eq!(humanoid.morphology, MorphologyType::Humanoid);
    assert_eq!(humanoid.total_mass_kg, 72.0);
    assert_eq!(humanoid.total_height_m, 1.75);
    assert_eq!(humanoid.joint_order.len(), 19);

    // Verify key anatomical joints exist
    assert!(humanoid.joints.contains_key("pelvis"));
    assert!(humanoid.joints.contains_key("head"));
    assert!(humanoid.joints.contains_key("shoulder_l"));
    assert!(humanoid.joints.contains_key("elbow_l"));
    assert!(humanoid.joints.contains_key("hip_r"));
    assert!(humanoid.joints.contains_key("knee_r"));

    // Verify torque and articulation limits on knee
    let knee = humanoid.joints.get("knee_r").unwrap();
    assert_eq!(knee.max_torque_nm, 160.0);
    assert_eq!(knee.min_angle_rad, 0.0);
    assert!((knee.max_angle_rad - 2.45).abs() < 1e-4);

    // Command within limits
    let ok = humanoid.set_joint_target("elbow_l", 1.5, Some(35.0));
    assert!(ok);
    humanoid.step_physics(0.5); // 0.5s * 4.5 rad/s = 2.25 rad capacity -> reaches 1.5 rad
    let elbow = humanoid.joints.get("elbow_l").unwrap();
    assert!((elbow.current_angle_rad - 1.5).abs() < 1e-2);

    // Command beyond limit: 3.5 rad -> clamped to 2.5 rad
    humanoid.set_joint_target("elbow_l", 3.5, None);
    let elbow_clamped = humanoid.joints.get("elbow_l").unwrap();
    assert!((elbow_clamped.target_angle_rad - 2.50).abs() < 1e-4);

    // Test 16-bit kinematic state vector round-trip
    let state_vec = humanoid.get_kinematic_state_vector();
    assert_eq!(state_vec.len(), 19 * 2);

    let mut twin = RobotBodyModel::humanoid_standard();
    twin.actuate_from_brain_vector(&state_vec);
    twin.step_physics(0.5);
    let twin_elbow = twin.joints.get("elbow_l").unwrap();
    assert!((twin_elbow.current_angle_rad - 1.5).abs() < 0.1);
}

#[test]
fn test_modular_body_morphologies() {
    let quad = RobotBodyModel::quadruped_standard();
    assert_eq!(quad.morphology, MorphologyType::Quadruped);
    assert!(quad.joints.contains_key("torso"));
    assert!(quad.joints.contains_key("calf_fl"));
    assert!(quad.joints.contains_key("calf_rr"));

    let diff = RobotBodyModel::differential_drive_standard();
    assert_eq!(diff.morphology, MorphologyType::DifferentialDrive);
    assert!(diff.joints.contains_key("wheel_left"));
    assert!(diff.joints.contains_key("wheel_right"));

    let arm = RobotBodyModel::articulated_arm_standard();
    assert_eq!(arm.morphology, MorphologyType::ArticulatedArm);
    assert!(arm.joints.contains_key("base_yaw"));
    assert!(arm.joints.contains_key("gripper"));
}

#[test]
fn test_audio_16bit_quantization_and_feature_extraction() {
    // Generate a 1600-sample test wave (100ms at 16kHz)
    let mut float_samples = Vec::with_capacity(1600);
    for i in 0..1600 {
        let t = i as f32 / 16000.0;
        let s = 0.75 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        float_samples.push(s);
    }

    let pcm16 = Audio16BitQuantizer::quantize_f32_to_pcm16(&float_samples);
    assert_eq!(pcm16.len(), 1600);
    assert_ne!(pcm16[100], 0);

    let bands = Audio16BitQuantizer::extract_band_energies_16(&pcm16);
    assert_eq!(bands.len(), 16);

    let atoms = Audio16BitQuantizer::atomize_bands(&bands);
    assert_eq!(atoms.len(), 17);
    assert_eq!(atoms[0], "aud:modality");
    assert!(atoms[1].starts_with("aud:b00:v"));
}

#[test]
fn test_biological_cortical_visual_reduction_angles_of_light() {
    let width = 32;
    let height = 32;

    // Synthetic image with vertical light edge in the center
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let val = if x >= 16 { 240 } else { 20 };
            rgba[idx] = val;
            rgba[idx + 1] = val;
            rgba[idx + 2] = val;
            rgba[idx + 3] = 255;
        }
    }

    let result = CorticalVisualEncoder::process_frame(width, height, &rgba, 4, 20.0);

    assert!(!result.sparse_edges.is_empty());
    // In Hubel & Wiesel receptive field orientation, vertical edge = 90 degrees
    let vertical_edges: Vec<_> = result.sparse_edges.iter().filter(|e| e.angle_deg == 90).collect();
    assert!(vertical_edges.len() > 10);
    assert!(result.compression_ratio > 2.0);

    // Verify suppression of flat homogeneous regions
    let flat_gray = vec![128u8; width * height * 4];
    let flat_result = CorticalVisualEncoder::process_frame(width, height, &flat_gray, 4, 25.0);
    assert_eq!(flat_result.sparse_edges.len(), 0);
}

#[test]
fn test_proprioceptive_multisensory_integration() {
    let humanoid = RobotBodyModel::humanoid_standard();
    let experience = humanoid.to_multisensory_experience(1000, 42);

    assert_eq!(experience.timestamp_ms, 1000);
    assert_eq!(experience.channels.len(), 1);
    assert_eq!(experience.channels[0].1, SenseModality::Proprioceptive);
    assert_eq!(experience.channels[0].2.len(), 19 * 2 * 2); // 19 joints * 2 fields * 2 bytes (i16)

    // Test KinematicProprioceptionEncoder atoms
    let (q_angle, q_torque) = KinematicProprioceptionEncoder::quantize(1.57, 40.0, 80.0);
    let atoms = KinematicProprioceptionEncoder::atomize_joint("shoulder_l", q_angle, q_torque);
    assert_eq!(atoms[0], "kin:modality");
    assert!(atoms[1].starts_with("kin:jshoulder_l:"));
}
