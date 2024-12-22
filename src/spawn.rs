use std::{path::Path, sync::{Arc, Mutex}};

use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use bevy_flycam::prelude::*;
use rand::Rng;
use crate::render::{self, CENTER};

#[derive(Resource)]
pub struct ExtractedKeypoints(pub Arc<Mutex<Vec<[f32; 3]>>>);

#[derive(Component)]
pub struct ToggleKeypoints;

pub fn spawn_mesh<T: AsRef<Path>>(
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

pub fn spawn_keypoints(
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

pub fn spawn_pyramid_camera(
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

    // Spawn pyramid edges
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(Color::WHITE)),
        transform
    ));
}

pub fn spawn_instructions(commands: &mut Commands) {
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