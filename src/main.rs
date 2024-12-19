use bevy::prelude::*;
use bevy_flycam::prelude::*;
mod render;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015, // default: 0.00012
            speed: 100.0,          // default: 12.0
        })
        .add_systems(Startup, setup)
        .run();
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Get mesh from RGBD image
    let mesh = render::rgbd_image_to_mesh(
        "assets/desk_1_1.png", 
        "assets/desk_1_1_depth.png"
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

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 0.0), //.looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam
    ));
}

