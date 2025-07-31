use crate::{
    bounds::assert_in_bounds, OrientedBlockFace, UnitQuadBuffer, UnorientedUnitQuad, VoxelVisibility,
};
use crate::VoxelContext;

use ilattice::glam::UVec3;
use ilattice::prelude::Extent;
use ndshape::Shape;

/// A fast and simple meshing algorithm that produces a single quad for every visible face of a block.
///
/// This is faster than [`greedy_quads`](crate::greedy_quads) but it produces many more quads.
pub fn visible_block_faces<T, S, C>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut UnitQuadBuffer,
    ctx: &C
) where
    S: Shape<3, Coord = u32>,
    C: VoxelContext<T>
{
    visible_block_faces_with_voxel_view(
        voxels,
        voxels_shape,
        min,
        max,
        faces,
        output,
        ctx,
    )
}

/// Same as [`visible_block_faces`](visible_block_faces),
/// with the additional ability to interpret the array as some other type.
/// Use this if you want to mesh the same array multiple times
/// with different sets of voxels being visible.
pub fn visible_block_faces_with_voxel_view<'a, T, S, C>(
    voxels: &'a [T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut UnitQuadBuffer,
    ctx: &C,
) where
    C: VoxelContext<T>,
    S: Shape<3, Coord = u32>,
{
    assert_in_bounds(voxels, voxels_shape, min, max);

    let min = UVec3::from(min).as_ivec3();
    let max = UVec3::from(max).as_ivec3();
    let extent = Extent::from_min_and_max(min, max);
    let interior = extent.padded(-1); // Avoid accessing out of bounds with a 3x3x3 kernel.
    let interior =
        Extent::from_min_and_shape(interior.minimum.as_uvec3(), interior.shape.as_uvec3());

    let kernel_strides =
        faces.map(|face| voxels_shape.linearize(face.signed_normal().as_uvec3().to_array()));

    for p in interior.iter3() {
        let p_array = p.to_array();
        let p_index = voxels_shape.linearize(p_array);
        let p_voxel = unsafe { voxels.get_unchecked(p_index as usize) };

        if let VoxelVisibility::Empty = ctx.get_visibility(&p_voxel) {
            continue;
        }

        for (face_index, face_stride) in kernel_strides.into_iter().enumerate() {
            let neighbor_index = p_index.wrapping_add(face_stride);
            let neighbor_voxel = unsafe { voxels.get_unchecked(neighbor_index as usize) };

            // TODO: If the face lies between two transparent voxels, we choose not to mesh it. We might need to extend the
            // IsOpaque trait with different levels of transparency to support this.
            let face_needs_mesh = match ctx.get_visibility(&neighbor_voxel) {
                VoxelVisibility::Empty => true,
                VoxelVisibility::Translucent => {
                    ctx.get_visibility(&p_voxel) == VoxelVisibility::Opaque
                }
                VoxelVisibility::Opaque => false,
            };

            if face_needs_mesh {
                output.groups[face_index].push(UnorientedUnitQuad { minimum: p_array });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultVoxelContext, Voxel, RIGHT_HANDED_Y_UP_CONFIG};
    use ndshape::{ConstShape, ConstShape3u32};

    #[test]
    #[should_panic]
    fn panics_with_max_out_of_bounds_access() {
        let samples = [EMPTY; SampleShape::SIZE as usize];
        let mut buffer = UnitQuadBuffer::new();
        visible_block_faces(
            &samples,
            &SampleShape {},
            [0; 3],
            [34, 33, 33],
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
            &mut buffer,
            &DefaultVoxelContext
        );
    }

    #[test]
    #[should_panic]
    fn panics_with_min_out_of_bounds_access() {
        let samples = [EMPTY; SampleShape::SIZE as usize];
        let mut buffer = UnitQuadBuffer::new();
        visible_block_faces(
            &samples,
            &SampleShape {},
            [0, 34, 0],
            [33; 3],
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
            &mut buffer,
            &DefaultVoxelContext
        );
    }

    type SampleShape = ConstShape3u32<34, 34, 34>;

    /// Basic voxel type with one byte of texture layers
    #[derive(Default, Clone, Copy, Eq, PartialEq)]
    struct BoolVoxel(bool);

    const EMPTY: BoolVoxel = BoolVoxel(false);

    impl Voxel for BoolVoxel {
        fn get_visibility(&self) -> VoxelVisibility {
            if *self == EMPTY {
                VoxelVisibility::Empty
            } else {
                VoxelVisibility::Opaque
            }
        }
    }
}
