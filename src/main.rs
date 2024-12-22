use std::{path::Path, sync::{Arc, Mutex}};

use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use bevy_flycam::prelude::*;
use rand::Rng;
use render::CENTER;

mod orb;
mod render;

const IMG1_COLOR_PATH: &str = "assets/00000-color.png";
const IMG1_DEPTH_PATH: &str = "assets/00000-depth.png";
const IMG2_COLOR_PATH: &str = "assets/00050-color.png";
//const IMG2_DEPTH_PATH: &str = "assets/00050-depth.png";
const IMG4_COLOR_PATH: &str = "assets/00200-color.png";
const IMG4_DEPTH_PATH: &str = "assets/00200-depth.png";


#[derive(Resource)]
struct ExtractedKeypoints(Arc<Mutex<Vec<[f32; 3]>>>);

fn main() {
    let mut keypoints1: Vec<[f32; 3]> = Vec::new();
    let mut keypoints2: Vec<[f32; 3]> = Vec::new();
    if let Ok((kp1, kp2, _matches)) = orb::extract_features(
        IMG1_COLOR_PATH, 
        IMG2_COLOR_PATH
    ) {
        keypoints1 = render::give_depth(kp1, IMG1_DEPTH_PATH);
        keypoints2 = render::give_depth(kp2, IMG4_DEPTH_PATH);
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
        .add_systems(Update, input_handler)
        .run();
}

#[derive(Component)]
struct ToggleKeypoints;

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    keypoints: Res<ExtractedKeypoints>
) {
    // Pose IMG1
    let pose1_quat = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0); // Identity quaternion
    let pose1_translation = Vec3::new(0.0, 0.0, 0.0);
    let pose1 = Transform {
        rotation: pose1_quat,
        translation: pose1_translation,
        ..Default::default()
    };

    // Pose IMG2
    //let pose2_quat = Quat::from_xyzw(-0.0059421, 0.0373319, 0.0209614, 0.999065);
    //let pose2_translation = Vec3::new(-0.067494, -0.058187, 0.0369303);
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
    spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG1_COLOR_PATH, IMG1_DEPTH_PATH, pose1);

    // Spawn predicted mesh
    spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG4_COLOR_PATH, IMG4_DEPTH_PATH, pose2);

    // Spawn reference keypoints
    spawn_keypoints(&mut commands, &mut meshes, &mut materials, keypoints);

    // Spawn reference camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose1);

    // Spawn predicted camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose2);

    // Display instructions
    spawn_instructions(&mut commands);

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
    mut query: Query<&mut Visibility, With<ToggleKeypoints>>
) {
    // Toggle keypoints visibility
    if keyboard_input.just_pressed(KeyCode::KeyK) {
        for mut visibility in query.iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }
}

fn spawn_mesh<T: AsRef<Path>>(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    color_path: T, 
    depth_path: T,
    transform: Transform
) {
    // Get mesh from RGBD image
    let mesh = render::rgbd_to_mesh(color_path, depth_path);

    // Spawn the points mesh
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: None,
            unlit: true, // Makes the points ignore lighting
            ..default()
        })),
        transform
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
            Transform::from_xyz(keypoint[0], keypoint[1], keypoint[2]),
            Visibility::Hidden,
            ToggleKeypoints
        ));
    }
}

fn spawn_pyramid_camera(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    transform: Transform
) {
    // Pyramid vertices
    const SCALE: f32 = 700.0;
    const FOCAL: f32 = 200.0; // Made-up focal length
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),                                           // Top
        Vec3::new(-CENTER[0]/2.0/SCALE, -CENTER[1]/2.0/SCALE, FOCAL/SCALE), // Base bottom-left
        Vec3::new(CENTER[0]/2.0/SCALE, -CENTER[1]/2.0/SCALE, FOCAL/SCALE),  // Base bottom-right
        Vec3::new(CENTER[0]/2.0/SCALE, CENTER[1]/2.0/SCALE, FOCAL/SCALE),   // Base top-right
        Vec3::new(-CENTER[0]/2.0/SCALE, CENTER[1]/2.0/SCALE, FOCAL/SCALE),  // Base top-left
    ];

    // Edges of the pyramid (pairs of vertex indices)
    let edges = vec![
        (0, 1), (0, 2), (0, 3), (0, 4), // Sides
        (1, 2), (2, 3), (3, 4), (4, 1), // Base
    ];

    // Create line mesh
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    let positions: Vec<[f32; 3]> = edges
        .iter()
        .flat_map(|&(start, end)| {
            vec![
                vertices[start].to_array(),
                vertices[end].to_array(),
            ]
        })
        .collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    // Quaternion for orientation
    //let rotation: Quat = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);

    // Spawn pyramid edges
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(Color::WHITE)),
        transform
    ));
}

fn spawn_instructions(commands: &mut Commands) {
    commands.spawn((
        Text::new("Show keypoints (K)\nShow matches (M)\nRun algorithm (R)"),
        TextFont {
            font_size: 12.0,
            ..Default::default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}