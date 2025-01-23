use bevy::prelude::Transform;

pub fn particle_swarm_optimization(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    iterations: usize,
    inertia_weight: f32,
    cognitive_weight: f32,
    social_weight: f32,
    convergence_threshold: f32
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

    use rand::prelude::*;
    use bevy::math::{Quat, Vec3};
    use bevy::prelude::Transform;

    // Fitness function: Calculate the sum of the closest point distances
    fn fitness(transform: &Transform, source_points: &Vec<Vec3>, target_points: &Vec<Vec3>) -> f32 {
        source_points
            .iter()
            .map(|p| {
                let transformed_point = transform.rotation * *p + transform.translation;
                target_points
                    .iter()
                    .map(|t| transformed_point.distance(*t))
                    .fold(f32::INFINITY, f32::min)
            })
            .sum()
    }

    let mut rng = thread_rng();

    // Preprocess points into `Vec3` for easier calculations
    let source_points: Vec<Vec3> = source.iter().map(|&p| Vec3::from(p)).collect();
    let target_points: Vec<Vec3> = target.iter().map(|&p| Vec3::from(p)).collect();

    // Particle representation
    struct Particle {
        position: Transform,
        velocity: Transform,
        best_position: Transform,
        best_fitness: f32,
    }

    // Initialize the particle swarm
    let mut particles: Vec<Particle> = (0..population_size)
        .map(|_| {
            let position = Transform {
                translation: Vec3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                ),
                rotation: Quat::from_euler(
                    bevy::math::EulerRot::XYZ,
                    rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
                    rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
                    rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
                ),
                ..Default::default()
            };

            Particle {
                position,
                velocity: Transform {
                    translation: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    ..Default::default()
                },
                best_position: position,
                best_fitness: f32::INFINITY,
            }
        })
        .collect();

    // Global best particle
    let mut global_best_position = Transform::default();
    let mut global_best_fitness = f32::INFINITY;

    for _ in 0..iterations {
        for particle in &mut particles {
            // Evaluate fitness
            let current_fitness = fitness(&particle.position, &source_points, &target_points);

            // Update personal best
            if current_fitness < particle.best_fitness {
                particle.best_fitness = current_fitness;
                particle.best_position = particle.position;
            }

            // Update global best
            if current_fitness < global_best_fitness {
                global_best_fitness = current_fitness;
                global_best_position = particle.position;
            }
        }

        // Check for convergence
        if global_best_fitness < convergence_threshold {
            break;
        }

        // Update velocity and position
        for particle in &mut particles {
            let inertia = particle.velocity.translation * inertia_weight;

            let cognitive = (particle.best_position.translation - particle.position.translation)
                * cognitive_weight
                * rng.gen::<f32>();

            let social = (global_best_position.translation - particle.position.translation)
                * social_weight
                * rng.gen::<f32>();

            particle.velocity.translation = inertia + cognitive + social;

            particle.position.translation += particle.velocity.translation;

            // Update rotation using SLERP for smoothness
            let rotation_inertia = particle.velocity.rotation.slerp(Quat::IDENTITY, inertia_weight);

            let rotation_cognitive = Quat::from_euler(
                bevy::math::EulerRot::XYZ,
                rng.gen::<f32>() * cognitive_weight,
                rng.gen::<f32>() * cognitive_weight,
                rng.gen::<f32>() * cognitive_weight,
            );

            let rotation_social = Quat::from_euler(
                bevy::math::EulerRot::XYZ,
                rng.gen::<f32>() * social_weight,
                rng.gen::<f32>() * social_weight,
                rng.gen::<f32>() * social_weight,
            );

            particle.velocity.rotation = rotation_inertia * rotation_cognitive * rotation_social;

            particle.position.rotation *= particle.velocity.rotation;
        }
    }

    // Return the best global transformation
    if global_best_fitness < convergence_threshold {
        Ok(global_best_position)
    } else {
        Err("PSO failed to converge to a solution.".to_string())
    }
}
