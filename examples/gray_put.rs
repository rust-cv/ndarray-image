use image::open;
use ndarray::s;
use ndarray_image::{ImgLuma, NdGray, NdImage};
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
    let image = open(opt.file).expect("unable to open input image");
    let image = image.to_luma();
    let ndimage: NdGray = NdImage(&image).into();
    let mut ndimage = ndimage.to_owned();
    let slice = ndimage.slice_mut(s![..;10, ..;2]);
    for n in slice {
        *n = 255;
    }
    let image: Option<ImgLuma> = NdImage(ndimage.view()).into();
    let image = image.expect("failed to convert ndarray to image");
    image.save(opt.output).expect("failed to write output");
}
