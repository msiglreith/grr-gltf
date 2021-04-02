use glace::{f32x3, f32x4};
use spirv_std::{Cubemap, SampledImage};

#[spirv_std_macros::gpu_only]
pub fn cubemap_sample(cubemap: &SampledImage<Cubemap>, coord: f32x3) -> f32x4 {
    unsafe {
        let mut result = f32x4::default();
        asm!(
            "%sampledImage = OpLoad typeof*{1} {1}",
            "%coord = OpLoad typeof*{2} {2}",
            "%result = OpImageSampleImplicitLod typeof*{0} %sampledImage %coord",
            "OpStore {0} %result",
            in(reg) &mut result,
            in(reg) cubemap,
            in(reg) &coord,
        );
        result
    }
}

#[spirv_std_macros::gpu_only]
pub fn cubemap_sample_lod(cubemap: &SampledImage<Cubemap>, coord: f32x3, lod: f32) -> f32x4 {
    unsafe {
        let mut result = f32x4::default();
        asm!(
            "%sampledImage = OpLoad typeof*{1} {1}",
            "%coord = OpLoad typeof*{2} {2}",
            "%lod = OpLoad typeof*{3} {3}",
            "%result = OpImageSampleExplicitLod typeof*{0} %sampledImage %coord Lod %lod",
            "OpStore {0} %result",
            in(reg) &mut result,
            in(reg) cubemap,
            in(reg) &coord,
            in(reg) &lod,
        );
        result
    }
}
