use opencv::{
    core::{DMatch, KeyPoint, Mat, Vector}, 
    features2d::{draw_matches, BFMatcher, DrawMatchesFlags, ORB_ScoreType, ORB}, 
    imgcodecs::{imread, imwrite, IMREAD_GRAYSCALE}, 
    prelude::*, 
    Result
};

pub fn extract_features(img1_path: &str, img2_path: &str) -> 
    Result<(Vector<KeyPoint>, Vector<KeyPoint>, Vector<DMatch>)> {
    let img1 = imread(img1_path, IMREAD_GRAYSCALE)?;
    let img2 = imread(img2_path, IMREAD_GRAYSCALE)?;
    if img1.empty() || img2.empty() {
        panic!("Could not load one or both images");
    }

    // Create ORB detector
    let mut orb = ORB::create(
        500, // Number of keypoints
        1.2, // Scale factor
        8,   // Number of levels
        31,  // Edge threshold
        0,   // First level
        2,   // WTA_K
        ORB_ScoreType::HARRIS_SCORE,   // Score type (HARRIS_SCORE or FAST_SCORE)
        31,  // Patch size
        20,  // Fast threshold
    )?;

    // Detect ORB features and compute descriptors
    let mut keypoints1 = Vector::new();
    let mut descriptors1 = Mat::default();
    orb.detect_and_compute(&img1, &Mat::default(), &mut keypoints1, &mut descriptors1, false)?;

    let mut keypoints2 = Vector::new();
    let mut descriptors2 = Mat::default();
    orb.detect_and_compute(&img2, &Mat::default(), &mut keypoints2, &mut descriptors2, false)?;

    // Match features using the BFMatcher
    let bf = BFMatcher::create(opencv::core::NORM_HAMMING, true)?;
    let mut matches = Vector::new();
    bf.train_match(&descriptors1, &descriptors2, &mut matches, &Mat::default())?;

    // Sort matches by distance and retain the top 100
    let mut matches_vec: Vec<_> = matches.to_vec();
    matches_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    let top_matches = &matches_vec[..70]; // Take the top 100 matches

    // Convert top matches back to OpenCV VectorOfDMatch
    let mut top_matches_dmatch: Vector<DMatch> = Vector::new();
    for m in top_matches {
        top_matches_dmatch.push(*m);
    }

    println!("Number of matches: {}", matches.len());
    /*if let Ok(top_match) = top_matches_dmatch.get(0) {
        println!("Top Match 1: {:?}", top_match);
    }*/

    // Draw matches on the images
    let mut matched_image = Mat::default();
    draw_matches(
        &img1,
        &keypoints1,
        &img2,
        &keypoints2,
        &top_matches_dmatch,
        &mut matched_image,
        opencv::core::Scalar::all(-1.0),
        opencv::core::Scalar::all(-1.0),
        &opencv::core::Vector::new(),
        DrawMatchesFlags::DEFAULT,
    )?;

    // Save the matched image
    let output_path = "assets/matches.png";
    imwrite(output_path, &matched_image, &opencv::core::Vector::new())?;
    println!("Matched image saved to {}", output_path);

    Ok((keypoints1, keypoints2, matches))
}