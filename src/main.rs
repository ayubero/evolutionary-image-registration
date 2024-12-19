use bevy::prelude::*;
use bevy_flycam::prelude::*;
use opencv::{
    core::{KeyPoint, Mat, Vector}, 
    features2d::{draw_keypoints, DrawMatchesFlags, ORB_ScoreType, ORB}, 
    imgcodecs::{imread, imwrite, IMREAD_GRAYSCALE},
    prelude::*, 
    Result
};

mod render;

fn main() -> Result<()> {
    let image_path = "assets/desk_1_1.png";
    let img = imread(image_path, IMREAD_GRAYSCALE)?;
    if img.empty() {
        panic!("Failed to load image at {}", image_path);
    }

    // Create ORB detector
    let mut orb = ORB::create(
        500, // Number of keypoints
        1.2, // Scale factor
        8,   // Number of levels
        31,  // Edge threshold
        0,   // First level
        2,   // WTA_K
        ORB_ScoreType::HARRIS_SCORE,   // Score type (HARRIS_SCORE or FAST_SCORE)
        31,  // Patch size
        20,  // Fast threshold
    )?;

    // Detect keypoints and compute descriptors
    let mut keypoints = Vector::<KeyPoint>::new();
    let mut descriptors = Mat::default();
    orb.detect_and_compute(&img, &Mat::default(), &mut keypoints, &mut descriptors, false)?;

    // Print some information about the results
    println!("Number of keypoints detected: {}", keypoints.len());
    println!("Descriptor size: {:?}", descriptors.size()?);

    // Draw keypoints on the image
    let mut img_with_keypoints = Mat::default();
    draw_keypoints(
        &img, 
        &keypoints, 
        &mut img_with_keypoints, 
        opencv::core::Scalar::all(-1.0), 
        DrawMatchesFlags::DEFAULT
    )?;

    // Save or display the image with keypoints
    let output_path = "output_with_keypoints.png";
    let params = Vector::<i32>::new();
    imwrite(output_path, &img_with_keypoints, &params)?;

    // Optionally, display the image in a window (if using OpenCV GUI)
    // opencv::highgui::imshow("Keypoints", &img_with_keypoints)?;
    // opencv::highgui::wait_key(0)?;

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            sensitivity: 0.00015, // default: 0.00012
            speed: 100.0,          // default: 12.0
        })
        .add_systems(Startup, setup)
        .run();

    Ok(())
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

