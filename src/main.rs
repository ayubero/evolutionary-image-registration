use bevy::prelude::*;
use bevy_flycam::prelude::*;
use spawn::*;

mod orb;
mod render;
mod spawn;

const IMG1_COLOR_PATH: &str = "assets/00000-color.png";
const IMG1_DEPTH_PATH: &str = "assets/00000-depth.png";
const IMG2_COLOR_PATH: &str = "assets/00050-color.png";
const IMG2_DEPTH_PATH: &str = "assets/00050-depth.png";
//const IMG4_COLOR_PATH: &str = "assets/00200-color.png";
//const IMG4_DEPTH_PATH: &str = "assets/00200-depth.png";

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
        /*.insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 500.0,
        })*/
        .insert_resource(ExtractedKeypoints((keypoints1, keypoints2, matches)))
        .add_systems(Startup, setup)
        .add_systems(Update, input_handler)
        .run();
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    keypoints: Res<ExtractedKeypoints>
) {
    // Reference pose (from IMG1)
    let pose1_quat = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0); // Identity quaternion
    let pose1_translation = Vec3::new(0.0, 0.0, 0.0);
    let pose1 = Transform {
        rotation: pose1_quat,
        translation: pose1_translation,
        ..Default::default()
    };

    // True IMG2 pose
    /*// (x: -qX, y: -qY, z: qZ, w: qW)
    let pose2_quat = Quat::from_xyzw(0.0059421, -0.0373319, 0.0209614, 0.999065);
    // (x: -worldX, y: -worldY, z: worldZ)
    let pose2_translation = Vec3::new(0.067494*10.0, 0.058187*10.0, 0.0369303*10.0);
    let pose2 = Transform {
        rotation: pose2_quat,
        translation: pose2_translation,
        ..Default::default()
    };*/
    
    // Initial IMG2 pose (It's actually the IMG4 pose)
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
    spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG2_COLOR_PATH, IMG2_DEPTH_PATH, pose2);
    //spawn_mesh(&mut commands, &mut meshes, &mut materials, IMG4_COLOR_PATH, IMG4_DEPTH_PATH, pose4);

    // Spawn keypoints in both images (reference and target)
    let (keypoints1, keypoints2, matches) = &keypoints.0;
    spawn_keypoints(&mut commands, &mut meshes, &mut materials, matches,keypoints1,keypoints2,pose1,pose2);

    // Spawn reference camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose1);

    // Spawn predicted camera
    spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose2);
    //spawn_pyramid_camera(&mut commands, &mut meshes, &mut materials, pose4);

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
    mut param_set: ParamSet<(
        Query<&mut Visibility, With<ToggleKeypoints>>,
        Query<&mut Visibility, With<ToggleMatches>>,
    )>,
) {
    // Toggle keypoints visibility
    if keyboard_input.just_pressed(KeyCode::KeyK) {
        for mut visibility in param_set.p0().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }

    // Toggle matches visibility
    if keyboard_input.just_pressed(KeyCode::KeyM) {
        for mut visibility in param_set.p1().iter_mut() {
            visibility.toggle_visible_hidden();
        }
    }
}