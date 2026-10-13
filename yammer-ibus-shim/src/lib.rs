//! Yammer's IBus input-method engine, compiled as a `cdylib`.
//!
//! This crate is the only place `librush` (LGPL) is named as a dependency —
//! see CLAUDE.local.md for why, and `ffi.rs` for the boundary the rest of
//! Yammer talks to via `libloading` rather than a normal Cargo dependency.

mod bus;
mod engine;
mod ffi;

pub use ffi::*;
