use ndarray::s;
use ndarray_image::{open_image, save_image, Colors};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "image", about = "Loads image and puts red dots on it")]
struct Opt {
    /// File to put red dots on
    #[structopt(parse(from_os_str))]
    file: PathBuf,
    /// Output file with red dots
    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let mut image = open_image(opt.file, Colors::Rgb).expect("unable to open input image");
    for n in image.slice_mut(s![..;10, ..;2, 0]) {
        *n = 255;
    }
    save_image(opt.output, image.view(), Colors::Rgb).expect("failed to write output");
}
