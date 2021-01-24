use image::{
    Bgr, Bgra, ImageBuffer, ImageError, ImageResult, Luma, LumaA, Pixel, Primitive, Rgb, Rgba,
};
use ndarray::ShapeBuilder;
use ndarray::{Array2, Array3, ArrayView, ArrayViewMut, Ix2, Ix3};
use std::ops::Deref;
use std::path::Path;

/// This newtype struct can wrap an image from either the `ndarray` or `image` crates to
/// automatically allow them to be turned `into()` the equivalents in the other crate.
/// This works without copying.
pub struct NdImage<T>(pub T);

pub type NdGray<'a, A = u8> = ArrayView<'a, A, Ix2>;
pub type NdGrayMut<'a, A = u8> = ArrayViewMut<'a, A, Ix2>;
pub type NdColor<'a, A = u8> = ArrayView<'a, A, Ix3>;
pub type NdColorMut<'a, A = u8> = ArrayViewMut<'a, A, Ix3>;

pub type ImgLuma<'a, A = u8> = ImageBuffer<Luma<A>, &'a [A]>;
pub type ImgLumaA<'a, A = u8> = ImageBuffer<LumaA<A>, &'a [A]>;
pub type ImgRgb<'a, A = u8> = ImageBuffer<Rgb<A>, &'a [A]>;
pub type ImgRgba<'a, A = u8> = ImageBuffer<Rgba<A>, &'a [A]>;
pub type ImgBgr<'a, A = u8> = ImageBuffer<Bgr<A>, &'a [A]>;
pub type ImgBgra<'a, A = u8> = ImageBuffer<Bgra<A>, &'a [A]>;

pub enum Colors {
    Luma,
    LumaA,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
}

/// Opens a gray image using the `image` crate and loads it into a 2d array.
/// This performs a copy.
pub fn open_gray_image(path: impl AsRef<Path>) -> ImageResult<Array2<u8>> {
    let image = image::open(path)?;
    let image = image.to_luma8();
    let image: NdGray = NdImage(&image).into();
    Ok(image.to_owned())
}

/// Opens a color image using the `image` crate and loads it into a 3d array.
/// This performs a copy.
pub fn open_image(path: impl AsRef<Path>, colors: Colors) -> ImageResult<Array3<u8>> {
    let image = image::open(path)?;
    let image = match colors {
        Colors::Luma => {
            let image = image.to_luma8();
            let image: NdColor = NdImage(&image).into();
            image.to_owned()
        }
        Colors::LumaA => {
            let image = image.to_luma_alpha8();
            let image: NdColor = NdImage(&image).into();
            image.to_owned()
        }
        Colors::Rgb => {
            let image = image.to_rgb8();
            let image: NdColor = NdImage(&image).into();
            image.to_owned()
        }
        Colors::Rgba => {
            let image = image.to_rgba8();
            let image: NdColor = NdImage(&image).into();
            image.to_owned()
        }
        Colors::Bgr => {
            let image = image.to_bgr8();
            let image: NdColor = NdImage(&image).into();
            image.to_owned()
        }
        Colors::Bgra => {
            let image = image.to_bgra8();
            let image: NdColor = NdImage(&image).into();
            image.to_owned()
        }
    };
    Ok(image)
}

/// Saves a gray image using the `image` crate from a 3d array.
pub fn save_gray_image(path: impl AsRef<Path>, image: NdGray<'_, u8>) -> ImageResult<()> {
    let image: Option<ImgLuma> = NdImage(image.view()).into();
    let image = image.ok_or_else(|| {
        ImageError::Decoding(image::error::DecodingError::new(
            image::error::ImageFormatHint::Unknown,
            "non-contiguous ndarray Array",
        ))
    })?;
    image.save(path)?;
    Ok(())
}

/// Saves a color image using the `image` crate from a 3d array.
pub fn save_image(
    path: impl AsRef<Path>,
    image: NdColor<'_, u8>,
    colors: Colors,
) -> ImageResult<()> {
    match colors {
        Colors::Luma => {
            let image: Option<ImgLuma> = NdImage(image.view()).into();
            let image = image.ok_or_else(|| {
                ImageError::Decoding(image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "non-contiguous ndarray Array",
                ))
            })?;
            image.save(path)?;
        }
        Colors::LumaA => {
            let image: Option<ImgLumaA> = NdImage(image.view()).into();
            let image = image.ok_or_else(|| {
                ImageError::Decoding(image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "non-contiguous ndarray Array",
                ))
            })?;
            image.save(path)?;
        }
        Colors::Rgb => {
            let image: Option<ImgRgb> = NdImage(image.view()).into();
            let image = image.ok_or_else(|| {
                ImageError::Decoding(image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "non-contiguous ndarray Array",
                ))
            })?;
            image.save(path)?;
        }
        Colors::Rgba => {
            let image: Option<ImgRgba> = NdImage(image.view()).into();
            let image = image.ok_or_else(|| {
                ImageError::Decoding(image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "non-contiguous ndarray Array",
                ))
            })?;
            image.save(path)?;
        }
        Colors::Bgr => {
            let image: Option<ImgBgr> = NdImage(image.view()).into();
            let image = image.ok_or_else(|| {
                ImageError::Decoding(image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "non-contiguous ndarray Array",
                ))
            })?;
            image.save(path)?;
        }
        Colors::Bgra => {
            let image: Option<ImgBgra> = NdImage(image.view()).into();
            let image = image.ok_or_else(|| {
                ImageError::Decoding(image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "non-contiguous ndarray Array",
                ))
            })?;
            image.save(path)?;
        }
    }
    Ok(())
}

/// Turn grayscale images into 2d array views.
impl<'a, C, A: 'static> Into<NdGray<'a, A>> for NdImage<&'a ImageBuffer<Luma<A>, C>>
where
    A: Primitive,
    C: Deref<Target = [A]> + AsRef<[A]>,
{
    fn into(self) -> NdGray<'a, A> {
        let NdImage(image) = self;
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let slice: &'a [A] = unsafe { std::mem::transmute(image.as_flat_samples().as_slice()) };
        ArrayView::from_shape((height, width).strides((width, 1)), slice).unwrap()
    }
}

/// Turn grayscale images into mutable 2d array views.
impl<'a, C, A: 'static> Into<NdGrayMut<'a, A>> for NdImage<&'a mut ImageBuffer<Luma<A>, C>>
where
    A: Primitive,
    C: Deref<Target = [A]> + AsRef<[A]>,
{
    fn into(self) -> NdGrayMut<'a, A> {
        let NdImage(image) = self;
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        #[allow(clippy::transmute_ptr_to_ref)]
        let slice: &'a mut [A] =
            unsafe { std::mem::transmute(image.as_flat_samples().as_slice() as *const [A]) };
        ArrayViewMut::from_shape((height, width).strides((width, 1)), slice).unwrap()
    }
}

/// Turn arbitrary images into 3d array views with one dimension for the color channel.
impl<'a, C, P: 'static, A: 'static> Into<NdColor<'a, A>> for NdImage<&'a ImageBuffer<P, C>>
where
    A: Primitive,
    P: Pixel<Subpixel = A>,
    C: Deref<Target = [P::Subpixel]> + AsRef<[A]>,
{
    fn into(self) -> NdColor<'a, A> {
        let NdImage(image) = self;
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let channels = P::CHANNEL_COUNT as usize;
        let slice: &'a [A] = unsafe { std::mem::transmute(image.as_flat_samples().as_slice()) };
        ArrayView::from_shape(
            (height, width, channels).strides((width * channels, channels, 1)),
            slice,
        )
        .unwrap()
    }
}

/// Turn arbitrary images into mutable 3d array views with one dimension for the color channel.
impl<'a, C, P: 'static, A: 'static> Into<NdColorMut<'a, A>> for NdImage<&'a mut ImageBuffer<P, C>>
where
    A: Primitive,
    P: Pixel<Subpixel = A>,
    C: Deref<Target = [P::Subpixel]> + AsRef<[A]>,
{
    fn into(self) -> NdColorMut<'a, A> {
        let NdImage(image) = self;
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let channels = P::CHANNEL_COUNT as usize;
        #[allow(clippy::transmute_ptr_to_ref)]
        let slice: &'a mut [A] =
            unsafe { std::mem::transmute(image.as_flat_samples().as_slice() as *const [A]) };
        ArrayViewMut::from_shape(
            (height, width, channels).strides((width * channels, channels, 1)),
            slice,
        )
        .unwrap()
    }
}

/// Turn 2d `ArrayView` into a `Luma` image.
///
/// Can fail if the `ArrayView` is not contiguous.
impl<'a, A: 'static> Into<Option<ImgLuma<'a, A>>> for NdImage<NdGray<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgLuma<'a, A>> {
        let NdImage(image) = self;
        if let [height, width] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from slice")
            })
        } else {
            unreachable!("the ndarray had more than 2 dimensions");
        }
    }
}

/// Turn 3d `ArrayView` into a `Luma` image.
///
/// Can fail if the `ArrayView` is not contiguous or has the wrong number of channels.
impl<'a, A: 'static> Into<Option<ImgLuma<'a, A>>> for NdImage<NdColor<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgLuma<'a, A>> {
        let NdImage(image) = self;
        if let [height, width, 1] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}

/// Turn 3d `ArrayView` into a `LumaA` image.
///
/// Can fail if the `ArrayView` is not contiguous or has the wrong number of channels.
impl<'a, A: 'static> Into<Option<ImgLumaA<'a, A>>> for NdImage<NdColor<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgLumaA<'a, A>> {
        let NdImage(image) = self;
        if let [height, width, 2] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}

/// Turn 3d `ArrayView` into a `Rgb` image.
///
/// Can fail if the `ArrayView` is not contiguous or has the wrong number of channels.
impl<'a, A: 'static> Into<Option<ImgRgb<'a, A>>> for NdImage<NdColor<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgRgb<'a, A>> {
        let NdImage(image) = self;
        if let [height, width, 3] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}

/// Turn 3d `ArrayView` into a `Rgba` image.
///
/// Can fail if the `ArrayView` is not contiguous or has the wrong number of channels.
impl<'a, A: 'static> Into<Option<ImgRgba<'a, A>>> for NdImage<NdColor<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgRgba<'a, A>> {
        let NdImage(image) = self;
        if let [height, width, 4] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}

/// Turn 3d `ArrayView` into a `Bgr` image.
///
/// Can fail if the `ArrayView` is not contiguous or has the wrong number of channels.
impl<'a, A: 'static> Into<Option<ImgBgr<'a, A>>> for NdImage<NdColor<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgBgr<'a, A>> {
        let NdImage(image) = self;
        if let [height, width, 3] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}

/// Turn 3d `ArrayView` into a `Bgra` image.
///
/// Can fail if the `ArrayView` is not contiguous or has the wrong number of channels.
impl<'a, A: 'static> Into<Option<ImgBgra<'a, A>>> for NdImage<NdColor<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgBgra<'a, A>> {
        let NdImage(image) = self;
        if let [height, width, 4] = *image.shape() {
            image.to_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}
