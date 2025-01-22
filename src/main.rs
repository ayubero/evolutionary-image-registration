use std::mem;
use bevy::prelude::*;
use bevy_flycam::prelude::*;
use config::{CORRECT_POSE2, POSE1, POSE2};
use ga::genetic_algorithm;
//use problem::Problem;
use spawn::*;
//use evolutionary::*;
use icp::*;

mod evolutionary;
mod ga;
mod orb;
mod render;
mod spawn;
mod problem;
mod icp;
mod utils;
mod config;

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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015, // default: 0.00012
            speed: 10.0,          // default: 12.0
        })
        .insert_resource(CameraTransform(POSE2))
        .insert_resource(PointClouds{ source: Vec::new(), target: Vec::new() })
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
    mut point_clouds: ResMut<PointClouds>,
    camera_transform: Res<CameraTransform>
) {
    // Reference pose
    let pose1: Transform = POSE1;

    // Initial IMG2 pose
    let pose2 = POSE2;

    // Spawn reference mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, &mut point_clouds, IMG1_COLOR_PATH, IMG1_DEPTH_PATH, pose1, false);

    // Spawn predicted mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, &mut point_clouds, IMG2_COLOR_PATH, IMG2_DEPTH_PATH, pose2, true);
    
    // Spawn correspondences
    spawn_correspondences(&mut commands, &mut meshes, &mut materials, point_clouds.into(), camera_transform, Visibility::Hidden);

    // Spawn reference camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose1, false);

    // Spawn predicted camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose2, true);

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
    point_clouds: Res<PointClouds>,
    mut object_position: ResMut<CameraTransform>,
    mut param_set: ParamSet<(
        Query<&mut Visibility, With<ToggleImage>>,
        Query<&mut Visibility, With<ToggleCorrespondence>>
    )>,
) {
    // Toggle second image visibility
    if keyboard_input.just_pressed(KeyCode::KeyI) {
        for mut visibility in param_set.p0().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }

    // Toggle correspondences visibility
    if keyboard_input.just_pressed(KeyCode::KeyC) {
        for mut visibility in param_set.p1().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }

    // Execute algorithm
    if keyboard_input.just_pressed(KeyCode::KeyE) {
        run_algorithm(point_clouds, &mut object_position);
    }

    // Reset position
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        object_position.0 = POSE2;
    }

    // Show true position (V for Verity)
    if keyboard_input.just_pressed(KeyCode::KeyV) {
        object_position.0 = CORRECT_POSE2;
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
    point_clouds: Res<PointClouds>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut param_set: ParamSet<(
        Query<&mut Transform, With<MovableObject>>,
        Query<&mut Visibility, With<ToggleCorrespondence>>,
        Query<Entity, With<RemovableCorrespondence>>
    )>
) {
    if object_position.is_changed() {
        // Update camera and mesh
        for mut transform in param_set.p0().iter_mut() {
            transform.translation = object_position.0.translation;
            transform.rotation = object_position.0.rotation;
        }

        // Get current correspondence visibility
        let visibility: Visibility = mem::take(&mut *param_set.p1().iter_mut().next().unwrap());

        // Update correspondences
        // Remove current correspondences
        for entity in param_set.p2().iter_mut() {
            commands.entity(entity).despawn();
        }
        // Respawn them again
        spawn_correspondences(&mut commands, &mut meshes, &mut materials, point_clouds, object_position, visibility);
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

fn run_algorithm(
    point_clouds: Res<PointClouds>,
    object_position: &mut ResMut<CameraTransform>
) {
    println!("Running algorithm!");

    let source_points = &point_clouds.source;
    let target_points = &point_clouds.target;

    // ICP
    /*let result = iterative_closest_point(
        &source_points, 
        &target_points, 
        15, 
        0.5
    );*/

    // GA
    let result = genetic_algorithm(
        &source_points, 
        &target_points, 
        100, 
        30, 
        0.5, 
        0.5
    );
    match result {
        Ok(transform) => {
            println!("Algorithm succeeded!");
            println!("Translation: {:?}", transform.translation);
            println!("Rotation: {:?}", transform.rotation);

            // Update the Transform
            object_position.0.translation = transform.translation;
            object_position.0.rotation = transform.rotation;
        }
        Err(err) => {
            eprintln!("Algorithm failed: {}", err);
        }
    }

    /*// Problem input data
    let transform1 = Transform{ 
        rotation: POSE1_QUAT, 
        translation: POSE1_TRANSLATION,
        ..Default::default()
    };
    let (keypoints1, keypoints2, matches) = &keypoints.0;

    // Define problem
    let problem = Problem::new(
        transform1, 
        keypoints1.to_vec(), 
        keypoints2.to_vec(), 
        matches.to_vec()
    );

    // Run the evolution strategy
    let best_element = evolution_strategy(
        problem,
        Some("convex_recombination"),
        Some("simple_mutation"),
        5000, // Population size
        100, // Max iterations
        new_population_truncation
    );

    // Update the Transform
    object_position.0.translation = best_element.encoding.translation;
    object_position.0.rotation = best_element.encoding.rotation;*/
}
