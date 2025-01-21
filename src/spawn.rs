use std::path::Path;
use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use nalgebra::Point3;
use rand::Rng;

use crate::{render::{self, CENTER}, utils::{self, compute_residual_error}};
use utils::{convert_vec, find_correspondences};

#[derive(Resource)]
pub struct PointClouds {
    pub source: Vec<[f32; 3]>,
    pub target: Vec<[f32; 3]>,
}

#[derive(Resource)]
pub struct CameraTransform(pub Transform);

#[derive(Component)]
pub struct ToggleCorrespondence;

#[derive(Component)]
pub struct ToggleImage;

#[derive(Component)]
pub struct MovableObject;

#[derive(Component)]
pub struct RemovableCorrespondence;

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
    point_clouds: &mut ResMut<PointClouds>,
    color_path: T, 
    depth_path: T,
    transform: Transform,
    is_movable: bool
) {
    // Get mesh from RGBD image
    let (positions, mesh) = render::rgbd_to_mesh(color_path, depth_path);

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

        // Source points to transform
        point_clouds.source = positions;
    } else {
        // Reference point cloud
        point_clouds.target = positions;
    }
}

/// Draws correspondences
pub fn spawn_correspondences(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    point_clouds: Res<PointClouds>,
    camera_transform: Res<CameraTransform>,
    visibility: Visibility
) {
    let source = &point_clouds.source;
    let target = convert_vec(&point_clouds.target);
    let transformed_source = source.clone().into_iter().map(
        |src| {
            let src_point = Vec3 { x: src[0], y: src[1], z: src[2] };
            let transformed_src = camera_transform.0.transform_point(src_point);
            let transformed_point = Point3::from([
                transformed_src.x, transformed_src.y, transformed_src.z
            ]);
            transformed_point
        }
    ).collect();
    let correspondences = find_correspondences(
        &transformed_source, &target
    );
    let residual_error = compute_residual_error(&correspondences);
    println!("Residual error: {}", residual_error);
    //let mut i = 0;
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
                },
                visibility,
                ToggleCorrespondence,
                RemovableCorrespondence
            ));
            let icosphere_mesh = meshes.add(Sphere::new(0.1).mesh().ico(7).unwrap());
            // Movable source point
            commands.spawn((
                Mesh3d(icosphere_mesh.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    alpha_mode: AlphaMode::Opaque,
                    emissive: color.into(),
                    ..default()
                })),
                Transform::from_xyz(position1.x, position1.y, position1.z),
                visibility,
                ToggleCorrespondence,
                RemovableCorrespondence
            ));
            // Reference points, they don't move
            commands.spawn((
                Mesh3d(icosphere_mesh.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    alpha_mode: AlphaMode::Opaque,
                    emissive: color.into(),
                    ..default()
                })),
                Transform::from_xyz(position2.x, position2.y, position2.z),
                visibility,
                ToggleCorrespondence
            ));
        /*} else {
            break;
        }*/
        //i += 1;
    }
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
        Text::new("I - Show target image\nC - Show correspondences\nE - Execute algorithm\nR - Reset"),
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