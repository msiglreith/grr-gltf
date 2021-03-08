use spirv_std::SampledImage;
use flink::{f32x3, f32x4};

#[spirv(image_type(
    // sampled_type is hardcoded to f32 for now
    dim = "DimCube",
    depth = 0,
    arrayed = 0,
    multisampled = 0,
    sampled = 1,
    image_format = "Unknown"
))]
#[derive(Copy, Clone)]
pub struct ImageCube {
    _x: u32,
}

pub trait CubeImage {
    fn sample(&self, coord: f32x3) -> f32x4;
    fn sample_lod(&self, coord: f32x3, lod: f32) -> f32x4;
}

impl CubeImage for SampledImage<ImageCube> {
    #[spirv_std_macros::gpu_only]
    fn sample(&self, coord: f32x3) -> f32x4 {
        unsafe {
            let mut result = f32x4::ZERO;
            asm!(
                "%sampledImage = OpLoad typeof*{1} {1}",
                "%coord = OpLoad typeof*{2} {2}",
                "%result = OpImageSampleImplicitLod typeof*{0} %sampledImage %coord",
                "OpStore {0} %result",
                in(reg) &mut result,
                in(reg) self,
                in(reg) &coord
            );
            result
        }
    }

    #[spirv_std_macros::gpu_only]
    fn sample_lod(&self, coord: f32x3, lod: f32) -> f32x4 {
        unsafe {
            let mut result = f32x4::ZERO;
            asm!(
                "%sampledImage = OpLoad typeof*{1} {1}",
                "%coord = OpLoad typeof*{2} {2}",
                "%lod = OpLoad typeof*{3} {3}",
                "%result = OpImageSampleExplicitLod typeof*{0} %sampledImage %coord Lod %lod",
                "OpStore {0} %result",
                in(reg) &mut result,
                in(reg) self,
                in(reg) &coord,
                in(reg) &lod,
            );
            result
        }
    }
}
