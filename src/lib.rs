//! [![Crates.io](https://img.shields.io/crates/v/block-mesh.svg)](https://crates.io/crates/block-mesh)
//! [![Docs.rs](https://docs.rs/block-mesh/badge.svg)](https://docs.rs/block-mesh)
//!
//! Fast algorithms for generating voxel block meshes.
//!
//! ![Mesh Examples](https://raw.githubusercontent.com/bonsairobo/block-mesh-rs/main/examples-crate/render/mesh_examples.png)
//!
//! Two algorithms are included:
//! - [`visible_block_faces`](crate::visible_block_faces): very fast but suboptimal meshes
//! - [`greedy_quads`](crate::greedy_quads): not quite as fast, but far fewer triangles are generated
//!
//! Benchmarks show that [`visible_block_faces`](crate::visible_block_faces) generates about 40 million quads per second on a
//! single core of a 2.5 GHz Intel Core i7. Assuming spherical input data, [`greedy_quads`](crate::greedy_quads) can generate a
//! more optimal version of the same mesh with 1/3 of the quads, but it takes about 3 times longer. To run the benchmarks
//! yourself, `cd bench/ && cargo bench`.
//!
//! # Example Code
//!
//! ```
//! use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! use block_mesh::{greedy_quads, GreedyQuadsBuffer, MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
//! use block_mesh::DefaultVoxelContext;
//!
//! #[derive(Clone, Copy, Eq, PartialEq)]
//! struct BoolVoxel(bool);
//!
//! const EMPTY: BoolVoxel = BoolVoxel(false);
//! const FULL: BoolVoxel = BoolVoxel(true);
//!
//! impl Voxel for BoolVoxel {
//!     fn get_visibility(&self) -> VoxelVisibility {
//!         if *self == EMPTY {
//!             VoxelVisibility::Empty
//!         } else {
//!             VoxelVisibility::Opaque
//!         }
//!     }
//! }
//!
//! impl MergeVoxel for BoolVoxel {
//!     type MergeValue = Self;
//!     type MergeValueFacingNeighbour = Self;
//!
//!     fn merge_value(&self) -> Self::MergeValue {
//!         *self
//!     }
//!
//!     fn merge_value_facing_neighbour(&self) -> Self::MergeValueFacingNeighbour {
//!         *self
//!     }
//! }
//!
//! // A 16^3 chunk with 1-voxel boundary padding.
//! type ChunkShape = ConstShape3u32<18, 18, 18>;
//!
//! // This chunk will cover just a single octant of a sphere SDF (radius 15).
//! let mut voxels = [EMPTY; ChunkShape::SIZE as usize];
//! for i in 0..ChunkShape::SIZE {
//!     let [x, y, z] = ChunkShape::delinearize(i);
//!     voxels[i as usize] = if ((x * x + y * y + z * z) as f32).sqrt() < 15.0 {
//!         FULL
//!     } else {
//!         EMPTY
//!     };
//! }
//!
//! let mut buffer = GreedyQuadsBuffer::new(voxels.len());
//! greedy_quads(
//!     &voxels,
//!     &ChunkShape {},
//!     [0; 3],
//!     [17; 3],
//!     &RIGHT_HANDED_Y_UP_CONFIG.faces,
//!     &mut buffer,
//!     &DefaultVoxelContext,
//! );
//!
//! // Some quads were generated.
//! assert!(buffer.quads.num_quads() > 0);
//! ```

mod bounds;
mod buffer;
pub mod geometry;
mod greedy;
mod simple;

pub use buffer::*;
#[doc(inline)]
pub use geometry::*;
pub use greedy::*;
pub use simple::*;

pub use ilattice;
pub use ndshape;

/// Describes how this voxel influences mesh generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelVisibility {
    /// This voxel should not produce any geometry.
    Empty,
    /// Should produce geometry, and also light can pass through.
    Translucent,
    /// Light cannot pass through this voxel.
    Opaque,
}

/// Implement on your voxel types to inform the library
/// how to generate geometry for this voxel.
pub trait Voxel {
    fn get_visibility(&self) -> VoxelVisibility;
}

pub trait VoxelContext<T> {
    fn get_visibility(&self, voxel: &T) -> VoxelVisibility;
}

pub trait MergeVoxelContext<T>: VoxelContext<T> {
    type MergeValue: Eq;
    type MergeValueFacingNeighbour: Eq;

    fn merge_value(&self, voxel: &T) -> Self::MergeValue;

    fn merge_value_facing_neighbour(&self, voxel: &T) -> Self::MergeValueFacingNeighbour;
}

pub struct DefaultVoxelContext;

impl<T: Voxel> VoxelContext<T> for DefaultVoxelContext {
    fn get_visibility(&self, voxel: &T) -> VoxelVisibility {
        voxel.get_visibility()
    }
}

impl<T: MergeVoxel> MergeVoxelContext<T> for DefaultVoxelContext {
    type MergeValue = T::MergeValue;
    type MergeValueFacingNeighbour = T::MergeValueFacingNeighbour;

    fn merge_value(&self, voxel: &T) -> Self::MergeValue {
        voxel.merge_value()
    }

    fn merge_value_facing_neighbour(&self, voxel: &T) -> Self::MergeValueFacingNeighbour {
        voxel.merge_value_facing_neighbour()
    }
}
