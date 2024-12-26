use std::path::Path;
use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use rand::Rng;
use crate::render::{self, CENTER, NO_VALUE};

#[derive(Resource)]
pub struct ExtractedKeypoints(pub (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[usize; 2]>));

#[derive(Resource)]
pub struct CameraTransform(pub Transform);

#[derive(Component)]
pub struct ToggleKeypoints;

#[derive(Component)]
pub struct ToggleMatches;

#[derive(Component)]
pub struct MovableObject;

#[derive(Component)]
pub enum AxisButton {
    IncrementX,
    DecrementX,
    IncrementY,
    DecrementY,
    IncrementZ,
    DecrementZ,
}

#[derive(Component)]
#[derive(Debug)]
pub enum TextLabel {
    X,
    Y,
    Z,
}

pub fn spawn_mesh<T: AsRef<Path>>(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    color_path: T, 
    depth_path: T,
    transform: Transform,
    is_movable: bool
) {
    // Get mesh from RGBD image
    let mesh = render::rgbd_to_mesh(color_path, depth_path);

    // Spawn the points mesh
    let mut entity = commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: None,
            unlit: true, // Makes the points ignore lighting
            ..default()
        })),
        transform
    ));

    if is_movable {
        entity.insert(MovableObject);
    }
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
        let keypoint1 = keypoints1[query_idx];

        let train_idx = dmatch[1];
        let keypoint2 = keypoints2[train_idx];
        
        // Check if keypoint has depth
        if keypoint1[2] != NO_VALUE && keypoint2[2] != NO_VALUE {
            // Draw points
            let position1 = transform1.transform_point(Vec3::from_array(keypoint1));
            draw_keypoint(&mut commands, &mut meshes, &mut materials, position1, color);
            
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
    transform: Transform,
    is_movable: bool
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
    let mut entity = commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Opaque,
            emissive: Color::WHITE.into(),
            ..default()
        })),
        transform
    ));

    if is_movable {
        entity.insert(MovableObject);
    }
}

pub fn spawn_instructions(commands: &mut Commands) {
    commands.spawn((
        Text::new("I - Show target image\nK - Show keypoints\nM - Show matches\nR - Run algorithm"),
        TextFont {
            font_size: 16.0,
            ..Default::default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            right: Val::Px(10.0),
            ..default()
        },
    ));
}

const NORMAL_BUTTON: Color = Color::srgb(0.15, 0.15, 0.15);

pub fn spawn_controls(commands: &mut Commands) {
    spawn_button(commands, "+", 10.0, 50.0, AxisButton::IncrementX);
    spawn_button(commands, "-", 10.0, 10.0, AxisButton::DecrementX);
    spawn_text(commands, TextLabel::X, "X: 6.432", 10.0, 90.0);
    spawn_button(commands, "+", 50.0, 50.0, AxisButton::IncrementY);
    spawn_button(commands, "-", 50.0, 10.0, AxisButton::DecrementY);
    spawn_text(commands, TextLabel::Y, "Y: 3.214", 50.0, 90.0);
    spawn_button(commands, "+", 90.0, 50.0, AxisButton::IncrementZ);
    spawn_button(commands, "-", 90.0, 10.0, AxisButton::DecrementZ);
    spawn_text(commands, TextLabel::Z, "Z: 1.532", 90.0, 90.0);
}

fn spawn_button(commands: &mut Commands, text: &str, top: f32, left: f32, axis_button: AxisButton) {
    commands.spawn((
        Button,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(top),
            left: Val::Px(left),
            width: Val::Px(30.0),
            height: Val::Px(30.0),
            border: UiRect::all(Val::Px(2.0)),
            // Horizontally center child text
            justify_content: JustifyContent::Center,
            // Vertically center child text
            align_items: AlignItems::Center,
            ..default()
        },
        BorderColor(Color::BLACK),
        BorderRadius::MAX,
        BackgroundColor(NORMAL_BUTTON),
    ))
    .insert(axis_button)
    .with_child((
        Text::new(text),
        TextFont {
            font_size: 16.0,
            ..Default::default()
        }
    ));
}

fn spawn_text(commands: &mut Commands, label: TextLabel, text: &str, top: f32, left: f32) {
    commands.spawn((
        Text::new(text),
        TextFont {
            font_size: 16.0,
            ..Default::default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(top),
            left: Val::Px(left),
            ..default()
        },
        label
    ));
}