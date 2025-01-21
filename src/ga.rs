use bevy::prelude::*;
use rand::prelude::*;
use rand::distributions::Uniform;

use crate::utils::convert_vec;

pub fn genetic_algorithm(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    generations: usize,
    mutation_rate: f32,
    convergence_threshold: f32,
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty".to_string());
    }

    let source = convert_vec(source);
    let target = convert_vec(target);

    // Define the bounds for the transformation parameters
    let rotation_bounds = Uniform::new(-std::f32::consts::PI, std::f32::consts::PI);
    let translation_bounds = Uniform::new(-10.0, 10.0); // Adjust as needed for your data.

    let mut rng = rand::thread_rng();

    // Initialize population
    let mut population: Vec<Transform> = (0..population_size)
        .map(|_| Transform {
            rotation: random_quaternion(&mut rng, &rotation_bounds),
            translation: [
                rng.sample(translation_bounds),
                rng.sample(translation_bounds),
                rng.sample(translation_bounds),
            ],
        })
        .collect();

    for generation in 0..generations {
        // Evaluate fitness
        let fitness: Vec<f32> = population
            .iter()
            .map(|transform| {
                let transformed_source: Vec<[f32; 3]> = source
                    .iter()
                    .map(|&point| transform.transform_point(point))
                    .collect();
                let correspondences = find_correspondences(&transformed_source, &target);
                compute_residual_error(&correspondences)
            })
            .collect();

        // Check for convergence
        if let Some(&best_fitness) = fitness.iter().min() {
            if best_fitness < convergence_threshold {
                let best_index = fitness.iter().position(|&f| f == best_fitness).unwrap();
                return Ok(population[best_index].clone());
            }
        }

        // Selection: Select the top N individuals
        let selected_indices = select_top_n(&fitness, population_size / 2);
        let selected_population: Vec<Transform> = selected_indices
            .into_iter()
            .map(|i| population[i].clone())
            .collect();

        // Crossover and mutation
        population = evolve_population(
            selected_population,
            population_size,
            mutation_rate,
            &mut rng,
            &rotation_bounds,
            &translation_bounds,
        );

        println!("Generation {}: Best fitness = {}", generation, fitness.iter().cloned().reduce(f32::min).unwrap());
    }

    Err("Failed to converge to a solution within the given generations.".to_string())
}