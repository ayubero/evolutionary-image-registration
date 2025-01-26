use rand::prelude::*;
use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform;
use rayon::prelude::*;

use crate::utils::fitness;

pub fn genetic_algorithm(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    generations: usize,
    mutation_rate: f32,
    tournament_size: usize,
    convergence_threshold: f32,
    stopping_threshold: usize, // If there is no improvement in stopping_threshold iterations, stop
    verbose: bool
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

    // Initialize population
    let mut rng = thread_rng();
    let mut population: Vec<Transform> = (0..population_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            Transform {
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
            }
        })
        .collect();

    // Pre-process points into Bevy Vec3 for easier calculations
    let source_points: Vec<Vec3> = source.iter().map(|&p| Vec3::from(p)).collect();
    let target_points: Vec<Vec3> = target.iter().map(|&p| Vec3::from(p)).collect();

    let mut best_transform = None;
    let mut best_fitness = f32::INFINITY;
    let mut no_improvement_counter = 0;

    for g in 0..generations {
        // Evaluate fitness
        let mut fitness_scores: Vec<(f32, &Transform)> = population
            .iter()
            .map(|t| (fitness(t, &source_points, &target_points), t))
            .collect();

        fitness_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Update the best transform
        if fitness_scores[0].0 < best_fitness {
            best_fitness = fitness_scores[0].0;
            best_transform = Some(*fitness_scores[0].1);
            no_improvement_counter = 0; // Reset counter if improvement
            if verbose { println!("Generation {} | Best fitness: {}", g, best_fitness); }
        } else {
            no_improvement_counter += 1;
        }

        // Stop if no improvement in the last stopping_threshold generations
        if no_improvement_counter >= stopping_threshold {
            if verbose { 
                println!("No improvement in the last {} generations. Stopping early.", stopping_threshold) 
            };
            break;
        }

        // Check for convergence
        if best_fitness < convergence_threshold {
            break;
        }

        // Truncated selection
        /*let selected: Vec<Transform> = fitness_scores
            .iter()
            .take(population_size / 2)
            .map(|(_, t)| **t)
            .collect();*/

        // Tournament selection
        let selected: Vec<Transform> = (0..population_size / 2)
            .map(|_| {
                // Build a tournament
                let tournament: Vec<&(f32, &Transform)> = (0..tournament_size)
                    .map(|_| {
                        let index = rng.gen_range(0..fitness_scores.len());
                        &fitness_scores[index]
                    })
                    .collect();

                // Select the best individual in the tournament
                tournament
                    .iter()
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_, t)| (*t).clone()) // Dereference &Transform to Transform
                    .unwrap()
            })
            .collect();

        // Crossover (combine parents to create new offspring)
        let mut new_population = Vec::with_capacity(population_size);
        while new_population.len() < population_size {
            let parent1 = &selected[rng.gen_range(0..selected.len())];
            let parent2 = &selected[rng.gen_range(0..selected.len())];

            let child = Transform {
                translation: Vec3::new(
                    (parent1.translation.x + parent2.translation.x) / 2.0,
                    (parent1.translation.y + parent2.translation.y) / 2.0,
                    (parent1.translation.z + parent2.translation.z) / 2.0,
                ),
                rotation: parent1.rotation.slerp(parent2.rotation, 0.5),
                ..Default::default()
            };

            new_population.push(child);
        }

        // Mutation (randomly perturb new population)
        new_population.par_iter_mut().for_each(|individual| {
            let mut rng = thread_rng();
            if rng.gen::<f32>() < mutation_rate {
                individual.translation += Vec3::new(
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                );
                individual.rotation *= Quat::from_euler(
                    bevy::math::EulerRot::XYZ,
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                );
            }
        });

        population = new_population;
    }

    match best_transform {
        Some(t) => Ok(t),
        None => Err("Failed to find a suitable transformation.".to_string()),
    }
}
