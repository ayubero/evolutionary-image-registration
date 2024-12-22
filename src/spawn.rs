use std::path::Path;
use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use rand::Rng;
use crate::render::{self, CENTER};

#[derive(Resource)]
pub struct ExtractedKeypoints(pub (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[usize; 2]>));

#[derive(Component)]
pub struct ToggleKeypoints;

#[derive(Component)]
pub struct ToggleMatches;

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

/// Draws keypoints and their corresponding matches
/// 
/// * matches: Matches from the first image to the second one, which means that keypoints1[i]
/// has a corresponding point in keypoints2[matches[i]]
pub fn spawn_keypoints(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<StandardMaterial>>,
    matches: &Vec<[usize; 2]>,
    keypoints1: &Vec<[f32; 3]>,
    keypoints2: &Vec<[f32; 3]>,
    transform1: Transform,
    transform2: Transform
) {
    // Spawn all keypoints that match
    for dmatch in matches.iter() {
        let mut rng = rand::thread_rng();
    
        // Generate a random color
        let color = Color::hsv(
            rng.gen_range(0.0..360.0),
            1.0,
            1.0,
        );

        // Get keypoints coordinates and spawn them
        let query_idx = dmatch[0];
        let train_idx = dmatch[1];
        
        // Check array limits
        if query_idx < keypoints1.len() && train_idx < keypoints2.len() {
            // Draw points
            let keypoint1 = keypoints1[query_idx];
            let position1 = transform1.transform_point(Vec3::from_array(keypoint1));
            draw_keypoint(&mut commands, &mut meshes, &mut materials, position1, color);

            let keypoint2 = keypoints2[train_idx];
            let position2 = transform2.transform_point(Vec3::from_array(keypoint2));
            draw_keypoint(&mut commands, &mut meshes, &mut materials, position2, color);

            draw_line(
                &mut commands, 
                &mut meshes, 
                &mut materials, 
                position1, 
                position2, 
                color, 
                0.05
            );
        }
    }
}

fn draw_keypoint(
    commands: &mut Commands, 
    meshes: &mut ResMut<Assets<Mesh>>, 
    materials: &mut ResMut<Assets<StandardMaterial>>, 
    position: Vec3, 
    color: Color
) {
    let icosphere_mesh = meshes.add(Sphere::new(0.1).mesh().ico(7).unwrap());

    // Spawn a ball (sphere) at position (x, y, z)
    commands.spawn((
        Mesh3d(icosphere_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            alpha_mode: AlphaMode::Opaque,
            emissive: color.into(),
            ..default()
        })),
        Transform::from_xyz(position.x, position.y, position.z),
        Visibility::Hidden,
        ToggleKeypoints
    ));
}

fn draw_line(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    start: Vec3,
    end: Vec3,
    color: Color,
    thickness: f32,
) {
    let direction = end - start;
    let length = direction.length();
    let midpoint = start + direction * 0.5;
    let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());

    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(thickness / 2.0, length))),
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
        },
        Visibility::Hidden,
        ToggleMatches
    ));
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
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Opaque,
            emissive: Color::WHITE.into(),
            ..default()
        })),
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