use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::Point3;

// Compute squared distance between two points
fn distance_squared(p1: Point3<f32>, p2: Point3<f32>) -> f32 {
    let dx = p1.x - p2.x;
    let dy = p1.y - p2.y;
    let dz = p1.z - p2.z;
    dx * dx + dy * dy + dz * dz
}

fn build_kd_tree(points: &Vec<Point3<f32>>) -> KdTree<f32, 3> {
    let mut kdtree = KdTree::new();
    for (index, point) in points.iter().enumerate() {
        kdtree.add(&[point.x, point.y, point.z], index.try_into().unwrap());
    }
    kdtree
}

pub fn find_correspondences(
  source_points: &Vec<Point3<f32>>,
  target_points: &Vec<Point3<f32>>,
) -> Vec<(Point3<f32>, Point3<f32>)> {
    let kdtree = build_kd_tree(&target_points);

    let mut correspondences = Vec::new();

    for src_point in source_points {
        // Find the nearest neighbor in the k-d tree
        let nearest = kdtree.nearest_one::<SquaredEuclidean>(&[src_point.x, src_point.y, src_point.z]);

        // Extract the nearest point's coordinates
        let nearest_id = nearest.item as usize;
        let tgt_point = target_points[nearest_id];

        correspondences.push((*src_point, tgt_point));
    }

    correspondences
}

pub fn convert_vec(input: &Vec<[f32; 3]>) -> Vec<Point3<f32>> {
    input.iter().map(|&arr| Point3::from(arr)).collect()
}

pub fn compute_residual_error(correspondences: &Vec<(Point3<f32>, Point3<f32>)>) -> f32 {
    let residual_error: f32 = correspondences.iter()
        .map(|(src, tgt)| {
            distance_squared(*src, *tgt)
        })
        .sum::<f32>() / correspondences.len() as f32;
    residual_error
}

/*
use kiddo::KdTree;
use bevy::math::Vec3;

// Fitness function: Calculate the sum of the closest point distances using a k-d tree
fn fitness_with_kdtree(
    transform: &Transform,
    source_points: &Vec<Vec3>,
    target_kdtree: &KdTree<f32, 3>,
) -> f32 {
    source_points
        .iter()
        .map(|p| {
            let transformed_point = transform.rotation * *p + transform.translation;
            let nearest_distance = target_kdtree
                .nearest(&[transformed_point.x, transformed_point.y, transformed_point.z], 1, &squared_euclidean)
                .map(|nearest| nearest.0) // Extract the distance
                .unwrap_or(f32::INFINITY);
            nearest_distance.sqrt() // Convert squared distance to actual distance
        })
        .sum()
}
*/