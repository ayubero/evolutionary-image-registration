use bevy::prelude::*;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::config::POSE2;

pub fn iterative_closest_point(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    max_iterations: usize,
    convergence_threshold: f32,
    verbose: bool
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

    let current_transform = POSE2;
    let mut source_points: Vec<Vector3<f32>> = source.iter().map(|&p| {
        let point = current_transform.transform_point(Vec3::new(p[0], p[1], p[2]));
        Vector3::from([point.x, point.y, point.z])
    }).collect();
    let target_points: Vec<Vector3<f32>> = target.iter().map(|&p| Vector3::from(p)).collect();

    let mut transform = Transform::IDENTITY;

    for i in 0..max_iterations {
        // Find closest points in target for each point in source
        let mut closest_points = Vec::with_capacity(source_points.len());
        for &src in &source_points {
            if let Some(&closest) = target_points
                .iter()
                .min_by(|&&a, &&b| a.metric_distance(&src).partial_cmp(&b.metric_distance(&src)).unwrap())
            {
                closest_points.push(closest);
            }
        }

        // Compute centroids
        let source_centroid = source_points.iter().sum::<Vector3<f32>>() / source_points.len() as f32;
        let target_centroid = closest_points.iter().sum::<Vector3<f32>>() / closest_points.len() as f32;

        // Center the points
        let source_centered: Vec<Vector3<f32>> = source_points.iter().map(|p| p - source_centroid).collect();
        let target_centered: Vec<Vector3<f32>> = closest_points.iter().map(|p| p - target_centroid).collect();

        // Compute cross-covariance matrix
        let mut covariance = Matrix3::zeros();
        for (src, tgt) in source_centered.iter().zip(target_centered.iter()) {
            covariance += src * tgt.transpose();
        }

        // Perform Singular Value Decomposition (SVD)
        let svd = covariance.svd(true, true);
        let u = svd.u.unwrap();
        let v_t = svd.v_t.unwrap();

        let rotation_matrix = v_t.transpose() * u.transpose();

        // Ensure a proper rotation matrix (determinant must be 1)
        let det = rotation_matrix.determinant();
        let rotation_matrix = if det < 0.0 {
            let mut u_fixed = u.clone();
            u_fixed.column_mut(2).scale_mut(-1.0);
            v_t.transpose() * u_fixed.transpose()
        } else {
            rotation_matrix
        };

        let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
        let translation = target_centroid - rotation * source_centroid;

        // Update transform
        transform.translation = Vec3::new(translation.x, translation.y, translation.z);
        transform.rotation = Quat::from_xyzw(rotation.i, rotation.j, rotation.k, rotation.w);

        // Apply transform to source points
        source_points = source_points
            .iter()
            .map(|p| rotation * p + translation)
            .collect();

        // Check for convergence
        let mean_error: f32 = source_points
            .iter()
            .zip(closest_points.iter())
            .map(|(src, tgt)| src.metric_distance(tgt))
            .sum::<f32>()
            / source_points.len() as f32;

        if verbose { println!("Iteration {} | Mean error {}", i, mean_error); }

        if mean_error < convergence_threshold {
            break;
        }
    }

    Ok(transform)
}