use bevy::prelude::*;
use kiddo::{KdTree, SquaredEuclidean};
//use kiddo::distance::squared_euclidean;
use nalgebra::{Matrix3, Point3, UnitQuaternion};

use crate::config::POSE2;

pub fn iterative_closest_point(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    max_iterations: usize,
    convergence_threshold: f32,
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty".to_string());
    }

    let source = convert_vec(source);
    let target = convert_vec(target);

    println!("Length of source points: {}", source.len());
    println!("Length of target points: {}", target.len());

    //let mut current_transform = Transform::IDENTITY;
    /*let mut current_transform = Transform {
        rotation: Quat::from_xyzw(0.0610943, -0.324556, 0.149797, 0.931926),
        translation: Vec3::new(0.649504*10.0, 0.394082*10.0, 0.590801*10.0),
        ..Default::default()
    };*/
    // Correct transformation
    let mut current_transform = POSE2;

    for i in 0..max_iterations {
        println!("Iteration {:?}", i);
        // Find the closest points in the target for each point in the source
        /*let correspondences: Vec<([f32; 3], [f32; 3])> = source
            .iter()
            .map(|&src_point| {
                let closest_point = target
                    .iter()
                    .min_by(|&&a, &&b| {
                        let dist_a: f32 = distance_squared(
                            current_transform.transform_point(src_point.into()), &a
                        );
                        let dist_b = distance_squared(
                            current_transform.transform_point(src_point.into()), &b
                        );
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .unwrap();
                (src_point, *closest_point)
            })
            .collect();*/
        let transformed_source = source.clone().into_iter().map(
            |src| {
                let src_point = Vec3 { x: src.x, y: src.y, z: src.z };
                let transformed_src = current_transform.transform_point(src_point);
                let transformed_point = Point3::from([
                    transformed_src.x, transformed_src.y, transformed_src.z
                ]);
                transformed_point
            }
        ).collect();
        let correspondences = find_correspondences(
            &transformed_source, &target
        );

        // Compute error
        let residual_error = compute_residual_error(&correspondences);

        // Check for convergence
        if residual_error < convergence_threshold {
            break;
        }

        // Compute the centroids of source and target correspondences
        let (source_centroid, target_centroid) = compute_centroids(&correspondences);

        println!("Source centroid {:?}", source_centroid);
        println!("Target centroid {:?}", target_centroid);

        // Compute the optimal rotation and translation
        let (rotation, translation) = compute_alignment(&correspondences, source_centroid, target_centroid);

        println!("Aligment rotation {:?}", rotation);
        println!("Aligment translation {:?}", translation);

        // Update the current transform
        current_transform.translation = translation;

        let rotation_matrix = quat_to_matrix3(current_transform.rotation);
        let rotation_result = rotation * rotation_matrix;
        current_transform.rotation = matrix3_to_quat(rotation_result);

        /*
        if translation.length_squared() < convergence_threshold {
            break;
        } 
        */
    }

    Ok(current_transform)
}

fn compute_centroids(correspondences: &Vec<(Point3<f32>, Point3<f32>)>) -> (Point3<f32>, Point3<f32>) {
    let source_sum: Point3<f32> = correspondences
        .iter()
        .map(|(src, _)| src)
        .fold(Point3::new(0.0, 0.0, 0.0), |acc, point| acc + point.coords);

    let target_sum: Point3<f32> = correspondences
        .iter()
        .map(|(_, tgt)| tgt)
        .fold(Point3::new(0.0, 0.0, 0.0), |acc, point| acc + point.coords);

    let count = correspondences.len() as f32;
    (
        Point3::from(source_sum.coords / count),
        Point3::from(target_sum.coords / count),
    )
}

fn compute_alignment(
    correspondences: &Vec<(Point3<f32>, Point3<f32>)>,
    source_centroid: Point3<f32>,
    target_centroid: Point3<f32>,
) -> (Matrix3<f32>, Vec3) {
    // Compute covariance matrix
    let mut covariance: nalgebra::Matrix<f32, nalgebra::Const<3>, nalgebra::Const<3>, nalgebra::ArrayStorage<f32, 3, 3>> = Matrix3::zeros();
    for &(src, tgt) in correspondences {
        let src_centered = src.coords - source_centroid.coords;
        let tgt_centered = tgt.coords - target_centroid.coords;

        covariance += src_centered * tgt_centered.transpose();
    }

    // Perform SVD on the covariance matrix
    let svd = covariance.svd(true, true);

    // Compute the optimal rotation
    let rotation_matrix = svd.v_t.unwrap().transpose() * svd.u.unwrap();
    let det = rotation_matrix.determinant();
    let rotation_matrix = if det < 0.0 {
        // Correct for reflection
        let mut correction = Matrix3::identity();
        correction[(2, 2)] = -1.0;
        svd.v_t.unwrap().transpose() * correction * svd.u.unwrap()
    } else {
        rotation_matrix
    };

    /*let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    let bevy_rotation = Quat::from_xyzw(rotation.i, rotation.j, rotation.k, rotation.w).normalize();*/

    // Compute the optimal translation
    let translation_vector = target_centroid.coords - rotation_matrix * source_centroid.coords;
    let translation = Vec3::new(translation_vector.x, translation_vector.y, translation_vector.z);
    println!("Aligment translation {:?}", translation);

    (rotation_matrix, translation)
}

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

fn compute_residual_error(correspondences: &Vec<(Point3<f32>, Point3<f32>)>) -> f32 {
    let residual_error: f32 = correspondences.iter()
        .map(|(src, tgt)| {
            distance_squared(*src, *tgt)
        })
        .sum::<f32>() / correspondences.len() as f32;

    println!("Residual error: {}", residual_error);
    residual_error
}

fn quat_to_matrix3(quat: Quat) -> Matrix3<f32> {
    // Create a nalgebra quaternion from bevy::Quat
    let quaternion = UnitQuaternion::from_quaternion(
        nalgebra::Quaternion::new(quat.w, quat.x, quat.y, quat.z)
    );
    
    // Convert the quaternion into a 3x3 rotation matrix
    let rotation_matrix: Matrix3<f32> = quaternion.to_rotation_matrix().into_inner();

    // Return the 3x3 rotation matrix
    rotation_matrix
}

fn matrix3_to_quat(matrix: Matrix3<f32>) -> Quat {
    // Convert nalgebra::Matrix3 to bevy::Mat3
    let bevy_mat3 = bevy::math::Mat3::from_cols_array_2d(&matrix.data.0);

    // Convert bevy::Mat3 to bevy::Quat
    Quat::from_mat3(&bevy_mat3).normalize()
}


/*fn compute_alignment(
    correspondences: &[(Vec3, Vec3)],
    source_centroid: Vec3,
    target_centroid: Vec3,
) -> (Quat, Vec3) {
    // Compute covariance matrix
    let mut covariance = Matrix3::zeros();
    for &(src, tgt) in correspondences {
        let src_centered = Vector3::new(src.x, src.y, src.z) - Vector3::new(source_centroid.x, source_centroid.y, source_centroid.z);
        let tgt_centered = Vector3::new(tgt.x, tgt.y, tgt.z) - Vector3::new(target_centroid.x, target_centroid.y, target_centroid.z);

        covariance += src_centered * tgt_centered.transpose();
    }

    // Perform SVD on the covariance matrix
    let svd = covariance.svd(true, true);

    // Compute the optimal rotation
    let rotation_matrix = svd.v_t.unwrap().transpose() * svd.u.unwrap();
    let det = rotation_matrix.determinant();
    let rotation_matrix = if det < 0.0 {
        // Correct for reflection
        let mut correction = Matrix3::identity();
        correction[(2, 2)] = -1.0;
        svd.v_t.unwrap().transpose() * correction * svd.u.unwrap()
    } else {
        rotation_matrix
    };

    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    let bevy_rotation = Quat::from_xyzw(rotation.i, rotation.j, rotation.k, rotation.w);

    // Step 4: Compute the optimal translation
    let translation_vector = Vector3::new(target_centroid.x, target_centroid.y, target_centroid.z)
        - rotation_matrix * Vector3::new(source_centroid.x, source_centroid.y, source_centroid.z);

    let translation = Vec3::new(translation_vector.x, translation_vector.y, translation_vector.z);

    (bevy_rotation, translation)*/
    // Compute covariance matrix
    /*let mut covariance = Mat3::ZERO;
    for &(src, tgt) in correspondences {
        let src_centered = src - source_centroid;
        let tgt_centered = tgt - target_centroid;

        covariance += Mat3::from_cols(
            src_centered * tgt_centered.x,
            src_centered * tgt_centered.y,
            src_centered * tgt_centered.z,
        );
    }

    // Compute the optimal rotation using Singular Value Decomposition (SVD)
    let svd = covariance.svd();
    let rotation = Quat::from_mat3(&(svd.v_t * svd.u.transpose()));

    // Compute the optimal translation
    let translation = target_centroid - rotation.mul_vec3(source_centroid);

    (rotation, translation)*/
//}
