extern crate blindlib;
extern crate env_logger;

fn main() {
    env_logger::init();
    blindlib::run_reps();
}
