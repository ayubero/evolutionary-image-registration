use std::path::Path;

use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::PrimitiveTopology};
use image::{DynamicImage, Pixel};

pub fn rgbd_image_to_mesh<T: AsRef<Path>>(color_image: T, depth_image: T) -> Mesh {
    // Load the depth and color images
    let color_image = image::open(color_image).expect("Failed to load color image");
    let depth_image = image::open(depth_image).expect("Failed to load depth image");

    // Create a mesh for the points
    let mut mesh = Mesh::new(
        PrimitiveTopology::PointList,
        RenderAssetUsages::default(), // Use default asset usage
    );

    let mut positions = Vec::new();
    let mut colors = Vec::new();

    if let DynamicImage::ImageRgb8(color_buffer) = color_image {
        if let DynamicImage::ImageLuma16(depth_buffer) = depth_image {
            let (width, height) = depth_buffer.dimensions();
            for y in 0..height {
                for x in 0..width {
                    // Extract the depth value from the red channel (index 0)
                    let depth_value = depth_buffer.get_pixel(x, y).channels()[0] as f32;

                    if depth_value != 0.0 {
                        // Extract coordinates
                        let world_x = x as f32 - width as f32 / 2.0;
                        let world_y = y as f32 - height as f32 / 2.0;
                        let world_z = depth_value;
            
                        positions.push([world_x, -world_y, -world_z]);
            
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

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);

    mesh
}