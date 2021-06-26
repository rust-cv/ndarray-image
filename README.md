# ndarray-image

[![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo]

[ci]: https://img.shields.io/crates/v/ndarray-image.svg
[cl]: https://crates.io/crates/ndarray-image/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/ndarray-image/badge.svg
[dl]: https://docs.rs/ndarray-image/

[lo]: https://tokei.rs/b1/github/rust-cv/ndarray-image?category=code

Allows conversion between ndarray's types and image's types

## Deprecated

WARNING: This crate is currently deprecated in favor of https://github.com/rust-cv/nshare. Use `nshare` instead.
This crate may eventually be repurposed, or if someone else wants to take the name, just reach out on the Rust CV Discord.

This crate allows zero-copy conversion between `ArrayView` from `ndarray` and `ImageBuffer` from `image`.

## Output of `red_put` example

![red_put](http://vadixidav.github.io/ndarray-image/red_put.png)
