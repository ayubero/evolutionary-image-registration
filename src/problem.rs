use bevy::prelude::*;
use rand::Rng;
use crate::evolutionary::Problem;
use crate::render::NO_VALUE;

pub struct PointCloudRegistration {
    transform1: Transform,
    keypoints1: Vec<[f32; 3]>,
    keypoints2: Vec<[f32; 3]>,
    matches: Vec<[usize; 2]>
}

impl PointCloudRegistration {
    pub fn new(transform1: Transform, keypoints1: Vec<[f32; 3]>, keypoints2: Vec<[f32; 3]>, matches: Vec<[usize; 2]>) -> Self {
        Self {
            transform1: transform1,
            keypoints1: keypoints1,
            keypoints2: keypoints2,
            matches: matches
        }
    }
}

impl Problem for PointCloudRegistration {
    fn generate_random_elem(&self, rng: &mut rand::rngs::ThreadRng) -> Transform {
        // Generate random position, rotation for the Transform
        let position = Vec3::new(
            rng.gen_range(-10.0..10.0), 
            rng.gen_range(-10.0..10.0), 
            rng.gen_range(-10.0..10.0)
        );
        let rotation = Quat::from_xyzw(
            gen_rnd_angle(rng),
            gen_rnd_angle(rng),
            gen_rnd_angle(rng),
            gen_rnd_angle(rng),
        ).normalize();
        Transform::from_translation(position).with_rotation(rotation)
    }

    fn eval(&self, encoding: &Transform) -> f32 {
        let mut error = 0.0; // The error is the sum of all the distances
        for dmatch in self.matches.iter() {
            // Get keypoints coordinates
            let query_idx = dmatch[0];
            let keypoint1 = self.keypoints1[query_idx];
    
            let train_idx = dmatch[1];
            let keypoint2 = self.keypoints2[train_idx];
            
            // Check if keypoint has depth
            if keypoint1[2] != NO_VALUE && keypoint2[2] != NO_VALUE {
                // Compute the distance between two matched keypoints
                let position1 = self.transform1.transform_point(Vec3::from_array(keypoint1));
                let position2 = encoding.transform_point(Vec3::from_array(keypoint2));
                let mut distance = position1.distance(position2);
                error += distance;
                //println!("Distance {} Match {:?}", distance, dmatch);
            }
        }
        error
    }

    fn recombine(&self, parent1: &Transform, parent2: &Transform, _recombination_func: &str) -> Transform {
        // Average the positions and rotations
        let position = (parent1.translation + parent2.translation) / 2.0;
        // Spherical linear interpolation
        let rotation = parent1.rotation.slerp(parent2.rotation, 0.5); 
        Transform::from_translation(position).with_rotation(rotation)
    }

    fn mutate(&self, encoding: Transform, _mutation_func: &str) -> Transform {
        // Apply a small random change to the position, rotation
        let mut rng = rand::thread_rng();
        const TRANS_RANGE: f32 = 0.0000001;
        const ROT_RANGE: f32 = 0.000000001;
        let translation_mutation = Vec3::new(
            gen_rnd_number(&mut rng, -TRANS_RANGE, TRANS_RANGE), 
            gen_rnd_number(&mut rng, -TRANS_RANGE, TRANS_RANGE),
            gen_rnd_number(&mut rng, -TRANS_RANGE, TRANS_RANGE)
        );
        let rotation_mutation = Quat::from_xyzw(
            gen_rnd_number(&mut rng, -ROT_RANGE, ROT_RANGE),
            gen_rnd_number(&mut rng, -ROT_RANGE, ROT_RANGE),
            gen_rnd_number(&mut rng, -ROT_RANGE, ROT_RANGE),
            gen_rnd_number(&mut rng, -ROT_RANGE, ROT_RANGE)
        );

        let resulting_rotation = (encoding.rotation * rotation_mutation).normalize();

        encoding.with_translation(encoding.translation + translation_mutation)
                .with_rotation(resulting_rotation)
    }
}

fn gen_rnd_angle(rng: &mut rand::rngs::ThreadRng) -> f32 {
    rng.gen_range(0.0..std::f32::consts::PI)
}

fn gen_rnd_number(rng: &mut rand::rngs::ThreadRng, min: f32, max: f32) -> f32 {
    rng.gen_range(min..max)
}