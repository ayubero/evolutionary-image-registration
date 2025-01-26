use rand::prelude::*;
use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform;
use rayon::prelude::*;

use crate::utils::fitness;

pub fn differential_evolution(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    generations: usize,
    crossover_probability: f32,
    scale_factor: f32,
    convergence_threshold: f32,
    verbose: bool
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

    let mut rng = thread_rng();

    // Preprocess points into Vec3 for easier calculations
    let source_points: Vec<Vec3> = source.iter().map(|&p| Vec3::from(p)).collect();
    let target_points: Vec<Vec3> = target.iter().map(|&p| Vec3::from(p)).collect();

    // Individual representation: Transform (translation + rotation)
    struct Individual {
        transform: Transform,
        fitness: f32,
    }

    // Initialize the population
    let mut population: Vec<Individual> = (0..population_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
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

    // Perform Differential Evolution
    for g in 0..generations {
        for i in 0..population_size {
            // Select three random, distinct individuals (not including i)
            let mut indices: Vec<usize> = (0..population_size).filter(|&idx| idx != i).collect();
            indices.shuffle(&mut rng);
            let (a, b, c) = (indices[0], indices[1], indices[2]);

            // Perform mutation: Create a trial vector
            let mutant_translation = population[a].transform.translation
                + scale_factor
                    * (population[b].transform.translation - population[c].transform.translation);

            let mutant_rotation = Quat::from_euler(
                bevy::math::EulerRot::XYZ,
                population[a].transform.rotation.to_euler(bevy::math::EulerRot::XYZ).0
                    + scale_factor
                        * (population[b]
                            .transform
                            .rotation
                            .to_euler(bevy::math::EulerRot::XYZ)
                            .0
                            - population[c]
                                .transform
                                .rotation
                                .to_euler(bevy::math::EulerRot::XYZ)
                                .0),
                population[a].transform.rotation.to_euler(bevy::math::EulerRot::XYZ).1
                    + scale_factor
                        * (population[b]
                            .transform
                            .rotation
                            .to_euler(bevy::math::EulerRot::XYZ)
                            .1
                            - population[c]
                                .transform
                                .rotation
                                .to_euler(bevy::math::EulerRot::XYZ)
                                .1),
                population[a].transform.rotation.to_euler(bevy::math::EulerRot::XYZ).2
                    + scale_factor
                        * (population[b]
                            .transform
                            .rotation
                            .to_euler(bevy::math::EulerRot::XYZ)
                            .2
                            - population[c]
                                .transform
                                .rotation
                                .to_euler(bevy::math::EulerRot::XYZ)
                                .2),
            );

            let mutant = Transform {
                translation: mutant_translation,
                rotation: mutant_rotation,
                ..Default::default()
            };

            // Perform crossover
            let mut trial = population[i].transform.clone();
            if rng.gen::<f32>() < crossover_probability {
                trial.translation = mutant.translation;
            }
            if rng.gen::<f32>() < crossover_probability {
                trial.rotation = mutant.rotation;
            }

            // Evaluate trial individual
            let trial_fitness = fitness(&trial, &source_points, &target_points);

            // Selection: Replace if the trial is better
            if trial_fitness < population[i].fitness {
                population[i].transform = trial;
                population[i].fitness = trial_fitness;
            }
        }

        // Check for convergence
        let best_fitness = population.iter().map(|ind| ind.fitness).fold(f32::INFINITY, f32::min);
        if verbose { println!("Generation {} | Best fitness: {}", g, best_fitness); }

        if best_fitness < convergence_threshold {
            break;
        }
    }

    // Return the best solution
    if let Some(best_individual) = population.iter().min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()) {
        Ok(best_individual.transform.clone())
    } else {
        Err("Failed to find a solution.".to_string())
    }
}
