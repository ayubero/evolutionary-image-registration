use bevy::prelude::*;
use bevy_flycam::prelude::*;
use config::{POSE1, POSE2};
use nalgebra::Point3;
use rand::Rng;
//use problem::Problem;
use spawn::*;
//use evolutionary::*;
use icp::*;

mod evolutionary;
mod orb;
mod render;
mod spawn;
mod problem;
mod icp;
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
        .insert_resource(CameraTransform(/*Transform { 
            translation: Vec3::new(0.649504*10.0, 0.394082*10.0, 0.590801*10.0), 
            rotation: Quat::from_xyzw(0.0610943, -0.324556, 0.149797, 0.931926), 
            ..Default::default() 
        }*/POSE2))
        //.insert_resource(ClearColor(Color::srgb(0.9, 0.9, 0.9)))
        .insert_resource(ExtractedKeypoints((keypoints1, keypoints2, matches)))
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
    keypoints: Res<ExtractedKeypoints>
) {
    // Reference pose
    let pose1: Transform = POSE1;

    // Initial IMG2 pose
    let pose2 = POSE2;

    // Spawn reference mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, &mut point_clouds, IMG1_COLOR_PATH, IMG1_DEPTH_PATH, pose1, false);

    // Spawn predicted mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, &mut point_clouds, IMG2_COLOR_PATH, IMG2_DEPTH_PATH, pose2, true);
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

    draw_correspondences(&mut commands, &mut meshes, &mut materials, point_clouds.into());

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
    point_clouds: Res<PointClouds>,
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
        run_evolution_algorithm(keypoints, point_clouds, object_position);
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

fn draw_correspondences(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    point_clouds: Res<PointClouds>
) {
    let source = &point_clouds.source;
    let target = convert_vec(&point_clouds.target);
    let transformed_source = source.clone().into_iter().map(
        |src| {
            let src_point = Vec3 { x: src[0], y: src[1], z: src[2] };
            let transformed_src = POSE2.transform_point(src_point);
            let transformed_point = Point3::from([
                transformed_src.x, transformed_src.y, transformed_src.z
            ]);
            transformed_point
        }
    ).collect();
    let correspondences = find_correspondences(
        &transformed_source, &target
    );
    let mut i = 0;
    for (source, target) in correspondences {
        //if i < 1 {
            //println!("Correspondences drawn");
            let position1 = Vec3::new(source.x, source.y, source.z);
            let position2 = Vec3::new(target.x, target.y, target.z);
            //println!("Position1: {:?}", position1);
            //println!("Position2: {:?}", position2);
            let mut rng = rand::thread_rng();
            let color = Color::hsv(
                rng.gen_range(0.0..360.0),
                1.0,
                1.0,
            );
            let start = position1;
            let end = position2;
            let direction = end - start;
            let length = direction.length();
            let midpoint = start + direction * 0.5;
            let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());
            commands.spawn((
                Mesh3d(meshes.add(Cylinder::new(0.05 / 2.0, length))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    alpha_mode: AlphaMode::Opaque,
                    emissive: color.into(),
                    ..default()
                })),
                Transform {
                    translation: midpoint,
                    rotation,
                    ..Default::default()
                }
            ));
            let icosphere_mesh = meshes.add(Sphere::new(0.1).mesh().ico(7).unwrap());
            commands.spawn((
                Mesh3d(icosphere_mesh.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    alpha_mode: AlphaMode::Opaque,
                    emissive: color.into(),
                    ..default()
                })),
                Transform::from_xyz(position1.x, position1.y, position1.z)
            ));
            commands.spawn((
                Mesh3d(icosphere_mesh.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    alpha_mode: AlphaMode::Opaque,
                    emissive: color.into(),
                    ..default()
                })),
                Transform::from_xyz(position2.x, position2.y, position2.z)
            ));
        /*} else {
            break;
        }*/
        i += 1;
    }
}

fn run_evolution_algorithm(
    keypoints: Res<ExtractedKeypoints>,
    point_clouds: Res<PointClouds>,
    mut object_position: ResMut<CameraTransform>
) {
    let source_points = &point_clouds.source;
    let target_points = &point_clouds.target;

    let icp_result = iterative_closest_point(&source_points, &target_points, 10, 50.0);
    match icp_result {
        Ok(transform) => {
            println!("ICP succeeded!");
            println!("Translation: {:?}", transform.translation);
            println!("Rotation: {:?}", transform.rotation);

            // Update the Transform
            object_position.0.translation = transform.translation;
            object_position.0.rotation = transform.rotation;
        }
        Err(err) => {
            eprintln!("ICP failed: {}", err);
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
