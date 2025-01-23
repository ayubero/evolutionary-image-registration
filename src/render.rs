use std::path::Path;
use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use image::{DynamicImage, Pixel};
/*use rand::seq::SliceRandom;
use rand::thread_rng;*/

// Camera intrinsics (these values should come from your camera calibration; but were obtained from the dataset)
pub const CENTER: [f32; 2] = [320.0, 240.0]; // Image center
pub const CONSTANT: f32 = 570.3;             // Focal length (in pixels, typically in x and y directions)
pub const MM_PER_M: f32 = 1000.0;            // Conversion factor (millimeters to meters)
pub const NO_VALUE: f32 = 0.0;

pub fn rgbd_to_mesh<T: AsRef<Path>>(color_path: T, depth_path: T) -> (Vec<[f32; 3]>, Mesh) {
    // Load the depth and color images
    let color_image = image::open(color_path).expect("Failed to load color image");
    let depth_image = image::open(depth_path).expect("Failed to load depth image");

    // Create a mesh for the points
    let mut mesh = Mesh::new(
        PrimitiveTopology::PointList,
        RenderAssetUsages::default() // Use default asset usage
    );

    let mut positions = Vec::new();
    let mut filtered_positions = Vec::new();
    let mut colors = Vec::new();

    if let DynamicImage::ImageRgb8(color_buffer) = color_image {
        if let DynamicImage::ImageLuma16(depth_buffer) = depth_image {
            let (width, height) = depth_buffer.dimensions();
            for y in 0..height {
                for x in 0..width {
                    // Extract the depth value from the red channel (index 0)
                    let depth_value = depth_buffer.get_pixel(x, y).channels()[0] as f32;

                    if depth_value != NO_VALUE {
                        let coordinates = compute_world_coordinates(x as f32, y as f32, depth_value);
                        positions.push(coordinates);

                        if x % 30 == 0 && y % 30 == 0  { // Filter some points
                            filtered_positions.push(coordinates)
                        }
            
                        // Get the corresponding color from the color image
                        let color = color_buffer.get_pixel(x, y);
                        colors.push([
                            color.channels()[0] as f32 / 255.0,
                            color.channels()[1] as f32 / 255.0,
                            color.channels()[2] as f32 / 255.0,
                            1.0, // Alpha channel
                        ]);
                    }
                }
            }
        } else {
            panic!("Expected an 16-bit grayscale image!");
        }
    } else {
        panic!("Expected an 8-bit RGB image!");
    }

    //positions = uniform_random_sample_positions(&positions, 1000);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);

    println!("Mesh with {} points", positions.len());

    (filtered_positions, mesh)
}

fn compute_world_coordinates(x: f32, y: f32, depth: f32) -> [f32; 3] {
    // Convert pixel (x, y) to normalized camera coordinates
    let x_rel = x - CENTER[0]; // x - cx
    let y_rel = y - CENTER[1]; // y - cy

    // Compute the 3D world coordinates
    let world_x = (x_rel * depth) / CONSTANT / MM_PER_M;
    let world_y = (y_rel * depth) / CONSTANT / MM_PER_M;
    let world_z = depth / MM_PER_M;

    [-world_x, -world_y, world_z]
}