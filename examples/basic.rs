use clap::Parser;
use gvf_snakes::run_gvf_snakes;

#[derive(Parser)]
struct Args {
    input: String,
    output: String,
}

fn main() {
    let args = Args::parse();

    run_gvf_snakes(
        &args.input,
        &args.output,
        50,
        (128.0, 128.0),
        50.0,
        0.2,
        100,
        0.1,
        0.3,
        2.0,
        50,
    );
}
