use bevy::prelude::*;
use rand::Rng;
use std::cmp::Ordering;
use crate::problem;

#[derive(Clone)]
struct Element {
    encoding: Transform,
    cost: f32,
}

impl Element {
    fn new(encoding: Transform, cost: f32) -> Self {
        Element { encoding, cost }
    }

    fn copy(&self) -> Self {
        Element {
            encoding: self.encoding.clone(),
            cost: self.cost,
        }
    }
}

impl std::fmt::Debug for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[el={:?}, cost={:?}]", self.encoding, self.cost)
    }
}

fn population_initialization(prob: &dyn Problem, pop_size: usize) -> Vec<Element> {
    let mut rng = rand::thread_rng();
    (0..pop_size)
        .map(|_| {
            let encoding = prob.generate_random_elem(&mut rng);
            Element::new(encoding, 0.0)
        })
        .collect()
}

fn population_evaluation(prob: &dyn Problem, population: &mut [Element]) {
    for el in population.iter_mut() {
        el.cost = prob.eval(&el.encoding);
    }
}

fn tournament_selection(population: &mut [Element], pop_size: usize) -> usize {
    let mut rng = rand::thread_rng();
    let i1 = rng.gen_range(0..pop_size);
    let i2 = rng.gen_range(0..pop_size);
    if population[i1].cost < population[i2].cost {
        i1
    } else {
        i2
    }
}

fn population_recombination(
    recombination_func: &str,
    prob: &dyn Problem,
    population: &[Element],
    index_p1: usize,
    index_p2: usize,
) -> Transform {
    prob.recombine(&population[index_p1].encoding, &population[index_p2].encoding, recombination_func)
}

fn new_population_truncation(population: &mut [Element], children: Vec<Element>) -> Vec<Element> {
    let mut joined_population = population.to_vec();
    joined_population.extend(children);

    joined_population.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal));
    joined_population.truncate(population.len());

    joined_population
}

pub trait Problem {
    fn generate_random_elem(&self, rng: &mut rand::rngs::ThreadRng) -> Transform;
    fn eval(&self, encoding: &Transform) -> f32;
    fn recombine(&self, parent1: &Transform, parent2: &Transform, recombination_func: &str) -> Transform;
    fn mutate(&self, encoding: Transform, mutation_func: &str) -> Transform;
}

fn evolution_strategy(
    prob: &dyn Problem,
    recombination_func: Option<&str>,
    mutation_func: Option<&str>,
    pop_size: usize,
    max_iter: usize,
    new_population_build: fn(&mut [Element], Vec<Element>) -> Vec<Element>,
) -> Element {
    let mut population = population_initialization(prob, pop_size);
    population_evaluation(prob, &mut population);

    let mut it = 0;
    while it < max_iter {
        let mut children = Vec::new();

        // Recombination
        for _ in 0..pop_size {
            let mut rng = rand::thread_rng();
            let i1 = rng.gen_range(0..pop_size);
            let i2 = rng.gen_range(0..pop_size);
            if let Some(recomb_func) = recombination_func {
                let c = population_recombination(recomb_func, prob, &population, i1, i2);
                children.push(Element::new(c, 0.0));
            }
        }

        // Mutation
        for child in children.iter_mut() {
            if let Some(mut_func) = mutation_func {
                child.encoding = prob.mutate(child.encoding.clone(), mut_func);
            }
        }

        population_evaluation(prob, &mut children);
        population = new_population_build(&mut population, children);
        it += 1;
    }

    population.into_iter().min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal)).unwrap()
}