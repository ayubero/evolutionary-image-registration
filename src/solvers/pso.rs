use rayon::prelude::*;
use rand::{thread_rng, Rng};
use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform;

use crate::utils::fitness;

pub fn particle_swarm_optimization(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    iterations: usize,
    constriction_factor: f32,
    inertia_weight: f32,
    cognitive_weight: f32,
    social_weight: f32,
    initial_weight: f32,
    convergence_threshold: f32,
    verbose: bool,
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

    let mut rng = thread_rng();

    // Preprocess points into Vec3 for easier calculations
    let source_points: Vec<Vec3> = source.iter().map(|&p| Vec3::from(p)).collect();
    let target_points: Vec<Vec3> = target.iter().map(|&p| Vec3::from(p)).collect();

    // Particle representation
    struct Particle {
        position: Transform,
        velocity: Transform,
        best_position: Transform,
        best_fitness: f32,
        initial_position: Transform
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
                initial_position: position.clone()
            }
        })
        .collect();

    // Global best particle
    let mut global_best_position = Transform::default();
    let mut global_best_fitness = f32::INFINITY;

    for i in 0..iterations {
        // Evaluate particles in parallel
        particles
            .par_iter_mut()
            .for_each(|particle| {
                // Evaluate fitness
                let current_fitness = fitness(&particle.position, &source_points, &target_points);

                // Update personal best
                if current_fitness < particle.best_fitness {
                    particle.best_fitness = current_fitness;
                    particle.best_position = particle.position;
                }
            });

        // Find the global best particle (reduce operation)
        let (local_best_position, local_best_fitness) = particles
            .par_iter()
            .map(|particle| (particle.best_position, particle.best_fitness))
            .reduce(
                || (Transform::default(), f32::INFINITY),
                |acc, next| if next.1 < acc.1 { next } else { acc },
            );

        // Update global best if a better fitness is found
        if local_best_fitness < global_best_fitness {
            global_best_fitness = local_best_fitness;
            global_best_position = local_best_position;

            if verbose && i != 0 {
                println!("Iteration {} | Best fitness: {}", i, global_best_fitness);
            }
        }

        // Check for convergence
        if global_best_fitness < convergence_threshold {
            break;
        }

        // Update velocity and position in parallel
        particles
            .par_iter_mut()
            .for_each(|particle| {
                let mut thread_rng = thread_rng();

                let inertia = particle.velocity.translation * inertia_weight;

                let cognitive = (particle.best_position.translation - particle.position.translation)
                    * cognitive_weight
                    * thread_rng.gen::<f32>();

                let social = (global_best_position.translation - particle.position.translation)
                    * social_weight
                    * thread_rng.gen::<f32>();

                let initial_influence = (particle.initial_position.translation
                    - particle.position.translation)
                    * initial_weight
                    * thread_rng.gen::<f32>();

                particle.velocity.translation = constriction_factor * (
                    inertia + cognitive + social + initial_influence
                );

                particle.position.translation += particle.velocity.translation;

                // Update rotation using SLERP for smoothness
                let rotation_inertia = particle.velocity.rotation.slerp(Quat::IDENTITY, inertia_weight);

                // Calculate rotation cognitive component
                let rotation_cognitive = particle.position.rotation.slerp(
                    particle.best_position.rotation,
                    thread_rng.gen::<f32>() * cognitive_weight,
                );

                // Calculate rotation social component
                let rotation_social = particle.position.rotation.slerp(
                    global_best_position.rotation,
                    thread_rng.gen::<f32>() * social_weight,
                );

                // Calculate rotation initial influence
                let rotation_initial = particle.position.rotation.slerp(
                    particle.initial_position.rotation,
                    thread_rng.gen::<f32>() * initial_weight,
                );

                particle.velocity.rotation = rotation_inertia * rotation_cognitive * rotation_social * rotation_initial;

                particle.position.rotation *= particle.velocity.rotation;
            });
    }

    // Return the best global transformation
    if global_best_fitness < f32::INFINITY {
        Ok(global_best_position)
    } else {
        Err("PSO failed to converge to a solution.".to_string())
    }
}
