use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy_flycam::prelude::*;
use rand::Rng;

mod orb;
mod render;

const IMG1_COLOR_PATH: &str = "assets/00000-color.png";
const IMG1_DEPTH_PATH: &str = "assets/00000-depth.png";
const IMG2_COLOR_PATH: &str = "assets/00050-color.png";
//const IMG2_DEPTH_PATH: &str = "assets/00050-depth.png";

#[derive(Resource)]
struct ExtractedKeypoints(Arc<Mutex<Vec<[f32; 3]>>>);

fn main() {
    let mut keypoints1: Vec<[f32; 3]> = Vec::new();
    if let Ok(kp1) = orb::extract_features(IMG1_COLOR_PATH, IMG2_COLOR_PATH) {
        keypoints1 = render::give_depth(kp1, IMG1_DEPTH_PATH);
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015, // default: 0.00012
            speed: 10.0,          // default: 12.0
        })
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 500.0,
        })
        .insert_resource(ExtractedKeypoints(Arc::new(Mutex::new(keypoints1))))
        .add_systems(Startup, setup)
        .run();
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    keypoints: Res<ExtractedKeypoints>
) {
    // Get mesh from RGBD image
    let mesh = render::rgbd_to_mesh(
        IMG1_COLOR_PATH, 
        IMG1_DEPTH_PATH
    );

    // Spawn the points mesh
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: None,
            unlit: true, // Makes the points ignore lighting
            ..default()
        })),
        Transform::default(),
    ));

    // Spawn keypoints
    spawn_keypoints(&mut commands, &mut meshes, &mut materials, keypoints);

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 0.0), //.looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam
    ));
}

fn spawn_keypoints(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    keypoints: Res<ExtractedKeypoints>,
) {
    let kps = keypoints.0.lock().unwrap(); // Safely access the keypoints
    let icosphere_mesh = meshes.add(Sphere::new(0.1).mesh().ico(7).unwrap());

    // Spawn all keypoints
    for keypoint in kps.iter() {
        let mut rng = rand::thread_rng();
    
        // Generate a random color
        let color = Color::hsv(
            rng.gen_range(0.0..360.0),
            1.0,
            1.0,
        );
        
        // Spawn a ball (sphere) at position (x, y, z)
        commands.spawn((
            Mesh3d(icosphere_mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                alpha_mode: AlphaMode::Opaque,
                ..default()
            })),
            Transform::from_xyz(keypoint[0], keypoint[1], keypoint[2])
        ));
    }
}