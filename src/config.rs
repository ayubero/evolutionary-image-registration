use bevy::prelude::*;

pub const POSE1: Transform = Transform {
    rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
    translation: Vec3::new(0.0, 0.0, 0.0),
    scale: Vec3::new(1.0, 1.0, 1.0)
};

// True POSE2
/*pub const POSE2: Transform = Transform {
    rotation: Quat::from_xyzw(0.0059421, -0.0373319, 0.0209614, 0.999065),
    translation: Vec3::new(0.067494*10.0, 0.058187*10.0, 0.0369303*10.0),
    scale: Vec3::new(1.0, 1.0, 1.0)
};*/

// Initial POSE2 (It's actually the POSE4)
pub const POSE2: Transform = Transform {
    rotation: Quat::from_xyzw(0.0610943, -0.324556, 0.149797, 0.931926),
    translation: Vec3::new(0.649504*10.0, 0.394082*10.0, 0.590801*10.0),
    scale: Vec3::new(1.0, 1.0, 1.0)
};