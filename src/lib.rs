use image::{Bgr, Bgra, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use ndarray::ShapeBuilder;
use ndarray::{ArrayView, Ix2, Ix3};
use std::ops::Deref;

/// This newtype struct can wrap an image from either the `ndarray` or `image` crates to
/// automatically allow them to be turned `into()` the equivalents in the other crate.
pub struct NdImage<T>(pub T);

pub type NdGray<'a, A = u8> = ArrayView<'a, A, Ix2>;
pub type NdColor<'a, A = u8> = ArrayView<'a, A, Ix3>;

pub type ImgLuma<'a, A = u8> = ImageBuffer<Luma<A>, &'a [A]>;
pub type ImgLumaA<'a, A = u8> = ImageBuffer<LumaA<A>, &'a [A]>;
pub type ImgRgb<'a, A = u8> = ImageBuffer<Rgb<A>, &'a [A]>;
pub type ImgRgba<'a, A = u8> = ImageBuffer<Rgba<A>, &'a [A]>;
pub type ImgBgr<'a, A = u8> = ImageBuffer<Bgr<A>, &'a [A]>;
pub type ImgBgra<'a, A = u8> = ImageBuffer<Bgra<A>, &'a [A]>;

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

/// Turn 2d `ArrayView` into a `Luma` image.
///
/// Can fail if the `ArrayView` is not contiguous.
impl<'a, A: 'static> Into<Option<ImgLuma<'a, A>>> for NdImage<NdGray<'a, A>>
where
    A: Primitive,
{
    fn into(self) -> Option<ImgLuma<'a, A>> {
        let NdImage(image) = self;
        if let &[height, width] = image.shape() {
            image.into_slice().map(|slice| {
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
        if let &[height, width, 1] = image.shape() {
            image.into_slice().map(|slice| {
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
        if let &[height, width, 2] = image.shape() {
            image.into_slice().map(|slice| {
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
        if let &[height, width, 3] = image.shape() {
            image.into_slice().map(|slice| {
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
        if let &[height, width, 4] = image.shape() {
            image.into_slice().map(|slice| {
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
        if let &[height, width, 3] = image.shape() {
            image.into_slice().map(|slice| {
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
        if let &[height, width, 4] = image.shape() {
            image.into_slice().map(|slice| {
                ImageBuffer::from_raw(width as u32, height as u32, slice)
                    .expect("failed to create image from raw vec")
            })
        } else {
            None
        }
    }
}
