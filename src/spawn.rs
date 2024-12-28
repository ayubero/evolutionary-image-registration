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
pub struct ToggleImage;

#[derive(Component)]
pub struct MovableObject;

#[derive(Component)]
pub struct MovableKeypoint{ pub original_position: Vec3 }

#[derive(Component)]
pub struct MovableMatch{ 
    pub original_start: Vec3,
    pub original_end: Vec3
}

#[derive(Component)]
pub enum TransformButton {
    IncrementTranslationX,
    DecrementTranslationX,
    IncrementTranslationY,
    DecrementTranslationY,
    IncrementTranslationZ,
    DecrementTranslationZ,
    IncrementRotationW,
    DecrementRotationW,
    IncrementRotationX,
    DecrementRotationX,
    IncrementRotationY,
    DecrementRotationY,
    IncrementRotationZ,
    DecrementRotationZ
}

#[derive(Component)]
#[derive(Debug)]
pub enum TextLabel {
    TranslationX,
    TranslationY,
    TranslationZ,
    RotationW,
    RotationX,
    RotationY,
    RotationZ
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
        entity.insert((Visibility::Hidden, MovableObject, ToggleImage));
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
            //let position1 = transform1.transform_point(Vec3::from_array(keypoint1));
            draw_keypoint(&mut commands, &mut meshes, &mut materials, keypoint1.into(), transform1, color, false);
            
            //let position2 = transform2.transform_point(Vec3::from_array(keypoint2));
            draw_keypoint(&mut commands, &mut meshes, &mut materials, keypoint2.into(), transform2, color, true);

            draw_match(
                &mut commands, 
                &mut meshes, 
                &mut materials, 
                keypoint1.into(), 
                keypoint2.into(), 
                transform1,
                transform2,
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
    transform: Transform,
    color: Color,
    is_movable: bool
) {
    let icosphere_mesh = meshes.add(Sphere::new(0.1).mesh().ico(7).unwrap());

    let transformed_position = transform.transform_point(position);

    // Spawn a ball (sphere) at position (x, y, z)
    let mut entity = commands.spawn((
        Mesh3d(icosphere_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            alpha_mode: AlphaMode::Opaque,
            emissive: color.into(),
            ..default()
        })),
        Transform::from_xyz(transformed_position.x, transformed_position.y, transformed_position.z),
        Visibility::Hidden,
        ToggleKeypoints
    ));

    if is_movable {
        entity.insert(MovableKeypoint{ original_position: position });
    }
}

fn draw_match(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    position1: Vec3,
    position2: Vec3,
    transform1: Transform,
    transform2: Transform,
    color: Color,
    thickness: f32,
) {
    let start = transform1.transform_point(position1);
    let end = transform2.transform_point(position2);
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
        ToggleMatches,
        MovableMatch {
            original_start: start,
            original_end: position2
        }
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
        entity.insert((Visibility::Hidden, MovableObject, ToggleImage));
    }
}

pub fn spawn_instructions(commands: &mut Commands) {
    commands.spawn((
        Text::new("I - Show target image\nK - Show keypoints\nM - Show matches\nE - Execute algorithm\nR - Reset"),
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
    spawn_button(commands, "+", 10.0, 50.0, TransformButton::IncrementTranslationX);
    spawn_button(commands, "-", 10.0, 10.0, TransformButton::DecrementTranslationX);
    spawn_text(commands, TextLabel::TranslationX, "TranslationX: 6.432", 12.0, 90.0);
    spawn_button(commands, "+", 50.0, 50.0, TransformButton::IncrementTranslationY);
    spawn_button(commands, "-", 50.0, 10.0, TransformButton::DecrementTranslationY);
    spawn_text(commands, TextLabel::TranslationY, "TranslationY: 3.214", 52.0, 90.0);
    spawn_button(commands, "+", 90.0, 50.0, TransformButton::IncrementTranslationZ);
    spawn_button(commands, "-", 90.0, 10.0, TransformButton::DecrementTranslationZ);
    spawn_text(commands, TextLabel::TranslationZ, "TranslationZ: 1.532", 92.0, 90.0);
    spawn_button(commands, "+", 130.0, 50.0, TransformButton::IncrementRotationW);
    spawn_button(commands, "-", 130.0, 10.0, TransformButton::DecrementRotationW);
    spawn_text(commands, TextLabel::RotationW, "RotationW: 1.532", 132.0, 90.0);
    spawn_button(commands, "+", 170.0, 50.0, TransformButton::IncrementRotationX);
    spawn_button(commands, "-", 170.0, 10.0, TransformButton::DecrementRotationX);
    spawn_text(commands, TextLabel::RotationX, "RotationX: 1.532", 172.0, 90.0);
    spawn_button(commands, "+", 210.0, 50.0, TransformButton::IncrementRotationY);
    spawn_button(commands, "-", 210.0, 10.0, TransformButton::DecrementRotationY);
    spawn_text(commands, TextLabel::RotationY, "RotationY: 1.532", 212.0, 90.0);
    spawn_button(commands, "+", 250.0, 50.0, TransformButton::IncrementRotationZ);
    spawn_button(commands, "-", 250.0, 10.0, TransformButton::DecrementRotationZ);
    spawn_text(commands, TextLabel::RotationZ, "RotationZ: 1.532", 252.0, 90.0);
}

fn spawn_button(commands: &mut Commands, text: &str, top: f32, left: f32, axis_button: TransformButton) {
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