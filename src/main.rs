use bevy::prelude::*;
use bevy_flycam::prelude::*;
use problem::PointCloudRegistration;
use spawn::*;
use evolutionary::*;

mod evolutionary;
mod orb;
mod render;
mod spawn;
mod problem;

const IMG1_COLOR_PATH: &str = "assets/00000-color.png";
const IMG1_DEPTH_PATH: &str = "assets/00000-depth.png";
const IMG2_COLOR_PATH: &str = "assets/00050-color.png";
const IMG2_DEPTH_PATH: &str = "assets/00050-depth.png";
/*const IMG3_COLOR_PATH: &str = "assets/00100-color.png";
const IMG3_DPETH_PATH: &str = "assets/00100-depth.png";
const IMG4_COLOR_PATH: &str = "assets/00150-color.png";
const IMG4_DEPTH_PATH: &str = "assets/00150-depth.png";
const IMG5_COLOR_PATH: &str = "assets/00200-color.png";
const IMG5_DEPTH_PATH: &str = "assets/00200-depth.png";*/

// Reference pose (from IMG1)
const POSE1_QUAT: Quat = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0); // Identity quaternion
const POSE1_TRANSLATION: Vec3 = Vec3::new(0.0, 0.0, 0.0);

fn main() {
    let mut keypoints1: Vec<[f32; 3]> = Vec::new();
    let mut keypoints2: Vec<[f32; 3]> = Vec::new();
    let mut matches: Vec<[usize; 2]> = Vec::new();
    if let Ok((kps1, kps2, mts)) = orb::extract_features(
        IMG1_COLOR_PATH, 
        IMG2_COLOR_PATH
    ) {
        keypoints1 = render::give_depth(kps1, IMG1_DEPTH_PATH);
        keypoints2 = render::give_depth(kps2, IMG2_DEPTH_PATH);
        matches = mts
            .iter()
            .map(|m| [m.query_idx as usize, m.train_idx as usize])
            .collect();
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015, // default: 0.00012
            speed: 10.0,          // default: 12.0
        })
        .insert_resource(CameraTransform(Transform { 
            translation: Vec3::new(0.649504*10.0, 0.394082*10.0, 0.590801*10.0), 
            rotation: Quat::from_xyzw(0.0610943, -0.324556, 0.149797, 0.931926), 
            ..Default::default() 
        }))
        //.insert_resource(ClearColor(Color::srgb(0.9, 0.9, 0.9)))
        .insert_resource(ExtractedKeypoints((keypoints1, keypoints2, matches)))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            input_handler, 
            button_click, 
            update_object_position, 
            update_text
        ))
        .run();
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    keypoints: Res<ExtractedKeypoints>
) {
    // Reference pose
    let pose1: Transform = Transform {
        rotation: POSE1_QUAT,
        translation: POSE1_TRANSLATION,
        ..Default::default()
    };

    // True IMG2 pose
    /*// (x: -qX, y: -qY, z: qZ, w: qW)
    let pose2_quat = Quat::from_xyzw(0.0059421, -0.0373319, 0.0209614, 0.999065);
    // (x: -worldX, y: -worldY, z: worldZ)
    let pose2_translation = Vec3::new(0.067494*10.0, 0.058187*10.0, 0.0369303*10.0);
    let pose2 = Transform {
        rotation: pose2_quat,
        translation: pose2_translation,
        ..Default::default()
    };*/
    
    // Initial IMG2 pose (It's actually the IMG5 pose)
    // (x: -qX, y: -qY, z: qZ, w: qW)
    let pose2_quat = Quat::from_xyzw(0.0610943, -0.324556, 0.149797, 0.931926);
    // (x: -worldX, y: -worldY, z: worldZ)
    let pose2_translation = Vec3::new(0.649504*10.0, 0.394082*10.0, 0.590801*10.0);
    let pose2 = Transform {
        rotation: pose2_quat,
        translation: pose2_translation,
        ..Default::default()
    };

    // Spawn reference mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG1_COLOR_PATH, IMG1_DEPTH_PATH, pose1, false);

    // Spawn predicted mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG2_COLOR_PATH, IMG2_DEPTH_PATH, pose2, true);
    //spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG4_COLOR_PATH, IMG4_DEPTH_PATH, pose4);

    // Spawn keypoints in both images (reference and target)
    let (keypoints1, keypoints2, matches) = &keypoints.0;
    spawn_keypoints(&mut commands, &mut meshes, &mut materials, matches,keypoints1,keypoints2,pose1,pose2);

    // Spawn reference camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose1, false);

    // Spawn predicted camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose2, true);
    //spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose4);

    // Display instructions
    spawn_instructions(&mut commands);

    // Display controls
    spawn_controls(&mut commands);

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::from_array([0.0, 0.0, 1.0]), Vec3::Y),
        FlyCam
    ));
}

// System to receive input from the user
fn input_handler(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    keypoints: Res<ExtractedKeypoints>,
    object_position: ResMut<CameraTransform>,
    mut param_set: ParamSet<(
        Query<&mut Visibility, With<ToggleImage>>,
        Query<&mut Visibility, With<ToggleKeypoints>>,
        Query<&mut Visibility, With<ToggleMatches>>
    )>,
) {
    // Toggle second image visibility
    if keyboard_input.just_pressed(KeyCode::KeyI) {
        for mut visibility in param_set.p0().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }

    // Toggle keypoints visibility
    if keyboard_input.just_pressed(KeyCode::KeyK) {
        for mut visibility in param_set.p1().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }

    // Toggle matches visibility
    if keyboard_input.just_pressed(KeyCode::KeyM) {
        for mut visibility in param_set.p2().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }

    // Execute algorithm
    if keyboard_input.just_pressed(KeyCode::KeyE) {
        println!("Running algorithm!");
        run_evolution_algorithm(keypoints, object_position);
    }
}

fn button_click(
    mut interaction_query: Query<(&Interaction, &TransformButton), Changed<Interaction>>,
    mut object_position: ResMut<CameraTransform>
) {
    let translation_movement = 0.5;
    let rotation_movement = 0.1;
    for (interaction, axis_button) in interaction_query.iter_mut() {
        if *interaction == Interaction::Pressed {
            match axis_button {
                TransformButton::IncrementTranslationX => object_position.0.translation.x += translation_movement,
                TransformButton::DecrementTranslationX => object_position.0.translation.x -= translation_movement,
                TransformButton::IncrementTranslationY => object_position.0.translation.y += translation_movement,
                TransformButton::DecrementTranslationY => object_position.0.translation.y -= translation_movement,
                TransformButton::IncrementTranslationZ => object_position.0.translation.z += translation_movement,
                TransformButton::DecrementTranslationZ => object_position.0.translation.z -= translation_movement,
                TransformButton::IncrementRotationW => object_position.0.rotation.w += rotation_movement,
                TransformButton::DecrementRotationW => object_position.0.rotation.w -= rotation_movement,
                TransformButton::IncrementRotationX => object_position.0.rotation.x += rotation_movement,
                TransformButton::DecrementRotationX => object_position.0.rotation.x -= rotation_movement,
                TransformButton::IncrementRotationY => object_position.0.rotation.y += rotation_movement,
                TransformButton::DecrementRotationY => object_position.0.rotation.y -= rotation_movement,
                TransformButton::IncrementRotationZ => object_position.0.rotation.z += rotation_movement,
                TransformButton::DecrementRotationZ => object_position.0.rotation.z -= rotation_movement
            }
            object_position.0.rotation = object_position.0.rotation.normalize();
        }
    }
}

fn update_object_position(
    object_position: Res<CameraTransform>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut param_set: ParamSet<(
        Query<&mut Transform, With<MovableObject>>,
        Query<(&mut Transform, &mut MovableKeypoint)>,
        Query<(&mut Transform, &mut MovableMatch, &Mesh3d)>
    )>
) {
    if object_position.is_changed() {
        // Update camera and mesh
        for mut transform in param_set.p0().iter_mut() {
            transform.translation = object_position.0.translation;
            transform.rotation = object_position.0.rotation;
        }

        // Update keypoints
        for (mut transform, keypoint) in param_set.p1().iter_mut() {
            let transformed_position = object_position.0.transform_point(keypoint.original_position);
            transform.translation = transformed_position;
        }

        // Update matches
        for (mut transform, match_link, mesh_handle) in param_set.p2().iter_mut() {
            let start = match_link.original_start;
            let end = object_position.0.transform_point(match_link.original_end);
            let direction = end - start;
            let length = direction.length();
            let midpoint = start + direction * 0.5;
            let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());

            transform.translation = midpoint;
            transform.rotation = rotation;

            let thickness = 0.05;

            // Update the cylinder's mesh if the length changes
            let new_mesh = Mesh::from(Cylinder::new(thickness / 2.0, length));
            let mesh = meshes.get_mut(mesh_handle).unwrap();
            *mesh = new_mesh;
        }
    }
}

fn update_text(
    object_position: Res<CameraTransform>,
    mut query: Query<(&TextLabel, &mut Text)>,
) {
    if object_position.is_changed() {
        for (label, mut text) in query.iter_mut() {
            let value = match label {
                TextLabel::TranslationX => object_position.0.translation.x,
                TextLabel::TranslationY => object_position.0.translation.y,
                TextLabel::TranslationZ => object_position.0.translation.z,
                TextLabel::RotationW => object_position.0.rotation.w,
                TextLabel::RotationX => object_position.0.rotation.x,
                TextLabel::RotationY => object_position.0.rotation.y,
                TextLabel::RotationZ => object_position.0.rotation.z,
            };
            **text = format!("{:?}: {:.3}", label, value);
        }
    }
}

fn run_evolution_algorithm(
    keypoints: Res<ExtractedKeypoints>,
    mut object_position: ResMut<CameraTransform>
) {
    // Problem input data
    let transform1 = Transform{ 
        rotation: POSE1_QUAT, 
        translation: POSE1_TRANSLATION,
        ..Default::default()
    };
    let (keypoints1, keypoints2, matches) = &keypoints.0;

    // Define problem
    let problem = PointCloudRegistration::new(
        transform1, 
        keypoints1.to_vec(), 
        keypoints2.to_vec(), 
        matches.to_vec()
    );

    // Run the evolution strategy
    let best_element = evolution_strategy(
        &problem,
        Some("convex_recombination"),
        Some("simple_mutation"),
        5000, // Population size
        100, // Max iterations
        new_population_truncation
    );

    // Update the Transform
    object_position.0.translation = best_element.encoding.translation;
    object_position.0.rotation = best_element.encoding.rotation;
}
