use rand::prelude::*;
use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform;
use rayon::prelude::*;

use crate::utils::fitness;

pub fn evolution_strategy(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    generations: usize,
    _learning_rate: f32,
    convergence_threshold: f32,
    verbose: bool
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

    // Preprocess points into Vec3
    let source_points: Vec<Vec3> = source.iter().map(|&p| Vec3::from(p)).collect();
    let target_points: Vec<Vec3> = target.iter().map(|&p| Vec3::from(p)).collect();

    // Individual representation: Transform (translation + rotation)
    struct Individual {
        transform: Transform,
        fitness: f32,
    }

    // Initialize the population with random transforms
    let mut population: Vec<Individual> = (0..population_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng(); // Thread-local RNG for safe random number generation
            let transform = Transform {
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

            let fitness_value = fitness(&transform, &source_points, &target_points);

            Individual {
                transform,
                fitness: fitness_value,
            }
        })
        .collect();

    // Evolution loop
    for g in 0..generations {
        // Compute the mean and standard deviation of the population
        let mean_translation: Vec3 = population
            .iter()
            .map(|ind| ind.transform.translation)
            .fold(Vec3::ZERO, |acc, t| acc + t)
            / population_size as f32;

        let mean_rotation: Quat = population
            .iter()
            .map(|ind| ind.transform.rotation)
            .fold(Quat::IDENTITY, |acc, r| acc * r)
            / population_size as f32;

        let std_dev_translation: Vec3 = population
            .iter()
            .map(|ind| {
                let diff = ind.transform.translation - mean_translation;
                Vec3::new(diff.x * diff.x, diff.y * diff.y, diff.z * diff.z)
            })
            .fold(Vec3::ZERO, |acc, d| acc + d)
            / population_size as f32;

        let std_dev_translation = Vec3::new(
            std_dev_translation.x.sqrt(),
            std_dev_translation.y.sqrt(),
            std_dev_translation.z.sqrt(),
        );

        // Mutation: Add Gaussian noise to each individual
        let mutated_population: Vec<Individual> = (0..population_size)
            .into_par_iter() // Convert the range into a parallel iterator
            .map(|_| {
                let mut rng = rand::thread_rng(); // Thread-local RNG for safe random number generation
                let translation = Vec3::new(
                    mean_translation.x + rng.gen::<f32>() * std_dev_translation.x,
                    mean_translation.y + rng.gen::<f32>() * std_dev_translation.y,
                    mean_translation.z + rng.gen::<f32>() * std_dev_translation.z,
                );

                let rotation = Quat::from_euler(
                    bevy::math::EulerRot::XYZ,
                    mean_rotation.to_euler(bevy::math::EulerRot::XYZ).0
                        + rng.gen_range(-0.1..0.1),
                    mean_rotation.to_euler(bevy::math::EulerRot::XYZ).1
                        + rng.gen_range(-0.1..0.1),
                    mean_rotation.to_euler(bevy::math::EulerRot::XYZ).2
                        + rng.gen_range(-0.1..0.1),
                );

                let transform = Transform {
                    translation,
                    rotation,
                    ..Default::default()
                };

                let fitness_value = fitness(&transform, &source_points, &target_points);

                Individual {
                    transform,
                    fitness: fitness_value,
                }
            })
            .collect();

        // Selection: Keep the best individuals
        population.extend(mutated_population);
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        population.truncate(population_size);

        if verbose { println!("Generation {} | Best fitness: {}", g, population[0].fitness); }

        // Check convergence
        if population[0].fitness < convergence_threshold {
            break;
        }
    }

    // Return the best solution
    if let Some(best_individual) = population.first() {
        Ok(best_individual.transform.clone())
    } else {
        Err("Failed to find a solution.".to_string())
    }
}
