// #[repr(C)]
// pub struct State {
//     pub counter: usize,
// }

#[unsafe(no_mangle)]
pub fn step(counter: &mut usize) {
    *counter += 1;
    println!("doing stuff in iteration {}", counter);
}
