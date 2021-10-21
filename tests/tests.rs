use std::time::Instant;

use all_of_matrix_multiplication::Matrix2D;

#[test]
fn test_multiplications() {
    let size = 3200;
    let a = Matrix2D::new_uniform_rand(size, size, -1., 1.);
    let b = Matrix2D::new_uniform_rand(size, size, -1., 1.);

    let start = Instant::now();
    let c1 = Matrix2D::naive_multiply(&a, &b);
    println!(
        "naive:                     {:>6} ms",
        start.elapsed().as_millis(),
    );

    let start = Instant::now();
    let c2 = Matrix2D::naive_multiply_optimized(&a, &b);
    println!(
        "naive optimized:           {:>6} ms",
        start.elapsed().as_millis()
    );
    println!("error: {:+e}", Matrix2D::sub(&c2, &c1).norm_inf());

    let start = Instant::now();
    let c3 = Matrix2D::naive_multiply_optimized_block(&a, &b);
    println!(
        "naive optimized with block: {:>6} ms",
        start.elapsed().as_millis()
    );
    println!("error: {:+e}", Matrix2D::sub(&c3, &c1).norm_inf());

    let start = Instant::now();
    let c4 = Matrix2D::strassen_multiply(&a, &b);
    println!(
        "strassen:                   {:>6} ms",
        start.elapsed().as_millis()
    );
    println!("error: {:+e}", Matrix2D::sub(&c4, &c1).norm_inf());

    let start = Instant::now();
    let c5 = Matrix2D::modified_strassen_multiply(&a, &b);
    println!(
        "modified strassen:          {:>6} ms",
        start.elapsed().as_millis()
    );
    println!("error: {:+e}", Matrix2D::sub(&c5, &c1).norm_inf());
}
