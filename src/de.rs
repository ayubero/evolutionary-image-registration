use rand::prelude::*;
use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform;

pub fn differential_evolution(
    source: &Vec<[f32; 3]>,
    target: &Vec<[f32; 3]>,
    population_size: usize,
    generations: usize,
    crossover_probability: f32,
    differential_weight: f32,
    convergence_threshold: f32
) -> Result<Transform, String> {
    if source.is_empty() || target.is_empty() {
        return Err("Source or target point cloud is empty.".to_string());
    }

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
        .map(|_| {
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
    for _ in 0..generations {
        for i in 0..population_size {
            // Select three random, distinct individuals (not including `i`)
            let mut indices: Vec<usize> = (0..population_size).filter(|&idx| idx != i).collect();
            indices.shuffle(&mut rng);
            let (a, b, c) = (indices[0], indices[1], indices[2]);

            // Perform mutation: Create a trial vector
            let mutant_translation = population[a].transform.translation
                + differential_weight
                    * (population[b].transform.translation - population[c].transform.translation);

            let mutant_rotation = Quat::from_euler(
                bevy::math::EulerRot::XYZ,
                population[a].transform.rotation.to_euler(bevy::math::EulerRot::XYZ).0
                    + differential_weight
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
                    + differential_weight
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
                    + differential_weight
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
