#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm)]
#![register_attr(spirv)]

use spirv_std::storage_class::{Input, Output, Uniform, UniformConstant};
use spirv_std::{SampledImage, Image2d};
use flink::{f32x3x3, f32x4x4, f32x2, f32x3, vec3, vec4, f32x4};

mod cube;
use self::cube::*;

#[spirv(block)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocalsPbr {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
    eye_world: f32x4,
    specular_mipmaps: u32,
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[spirv(fragment)]
pub fn main_fs(
    f_normal_world: Input<f32x3>,
    f_texcoord: Input<f32x2>,
    f_tangent_world: Input<f32x4>,
    f_position_world: Input<f32x3>,
    mut output: Output<f32x4>,
    #[spirv(binding = 0)] u_locals_fs: Uniform<LocalsPbr>,
    #[spirv(binding = 0)] u_albedo: UniformConstant<SampledImage<Image2d>>,
    #[spirv(binding = 1)] u_normals: UniformConstant<SampledImage<Image2d>>,
    #[spirv(binding = 2)] u_metal_roughness: UniformConstant<SampledImage<Image2d>>,
    #[spirv(binding = 3)] u_ambient_occlusion: UniformConstant<SampledImage<Image2d>>,
    #[spirv(binding = 4)] u_diffuse_map: UniformConstant<SampledImage<ImageCube>>,
    #[spirv(binding = 5)] u_specular_map: UniformConstant<SampledImage<ImageCube>>,
    #[spirv(binding = 6)] u_lut_ggx: UniformConstant<SampledImage<Image2d>>,
) {
    let f_texcoord = spirv_std::glam::Vec2::new(f_texcoord.x, f_texcoord.y);

    let normal = vec3(f_normal_world.x, f_normal_world.y, f_normal_world.z).normalize();
    let tangent = vec3(f_tangent_world.x, f_tangent_world.y, f_tangent_world.z).normalize();
    let bitangent = normal.cross(tangent).normalize();

    let u_normals = u_normals;
    let normal_tangent = u_normals.sample(f_texcoord);
    let normal_tangent = vec3(2.0 * normal_tangent.x - 1.0, 2.0 * normal_tangent.y - 1.0, 2.0 * normal_tangent.z - 1.0).normalize();
    let tangent_to_world = f32x3x3 {
        x: vec3(tangent.x, bitangent.x, normal.x),
        y: vec3(tangent.y, bitangent.y, normal.y),
        z: vec3(tangent.z, bitangent.z, normal.z),
    };
    let normal_world = (normal_tangent * tangent_to_world).normalize();

    // Diffuse IBL
    let albedo = u_albedo.sample(f_texcoord);

    let irradiance = u_diffuse_map.sample(normal_world);
    let light_diffuse = vec3(irradiance.x * albedo.x, irradiance.y * albedo.y, irradiance.z * albedo.z);

    // Specular IBL
    let metal_roughness = u_metal_roughness.sample(f_texcoord);

    let metalness = metal_roughness.z;
    let roughness = metal_roughness.y;

    let specular_color = vec3(mix(0.04, albedo.x, metalness), mix(0.04, albedo.y, metalness), mix(0.04, albedo.z, metalness));

    let u_locals = u_locals_fs;
    let eye_world = vec3(u_locals.eye_world.x, u_locals.eye_world.y, u_locals.eye_world.z);
    let view_world = (eye_world - *f_position_world).normalize();

    let n_dot_v = normal_world.dot(view_world);
    let reflect = 2.0 * n_dot_v * normal_world - view_world;
    let lod = roughness * u_locals.specular_mipmaps as f32;

    let brdf_ggx = u_lut_ggx.sample(spirv_std::glam::Vec2::new(n_dot_v.max(0.0), roughness));
    let specular_ibl = u_specular_map.sample_lod(reflect, lod);

    let light_specular = vec3(
        specular_ibl.x * (specular_color.x * brdf_ggx.x + brdf_ggx.y),
        specular_ibl.y * (specular_color.y * brdf_ggx.x + brdf_ggx.y),
        specular_ibl.z * (specular_color.z * brdf_ggx.x + brdf_ggx.y),
    );

    // AO
    let ambient_occlusion = u_ambient_occlusion.sample(f_texcoord);

    let color = vec3(
        (light_diffuse.x + light_specular.x) * ambient_occlusion.x,
        (light_diffuse.y + light_specular.y) * ambient_occlusion.y,
        (light_diffuse.z + light_specular.z) * ambient_occlusion.z,
    );

    *output = vec4(color.x, color.y, color.z, 1.0);
}

#[spirv(vertex)]
pub fn main_vs(
    v_position_obj: Input<f32x3>,
    v_normal_obj: Input<f32x3>,
    v_texcoord: Input<f32x2>,
    v_tangent_obj: Input<f32x4>,
    #[spirv(position)] mut a_position: Output<f32x4>,
    mut a_normal_world: Output<f32x3>,
    mut a_texcoord: Output<f32x2>,
    mut a_tangent_world: Output<f32x4>,
    mut a_position_world: Output<f32x3>,
    #[spirv(binding = 0)] u_locals_vs: Uniform<LocalsPbr>,
) {
    *a_normal_world = *v_normal_obj;
    *a_texcoord = *v_texcoord;
    *a_tangent_world = *v_tangent_obj;

    let locals = u_locals_vs;

    let pos_obj = v_position_obj;
    let pos_world = vec4(pos_obj.x, pos_obj.y, pos_obj.z, 1.0);
    *a_position_world = vec3(pos_world.x, pos_world.y, pos_world.z);

    let pos_view = pos_world * locals.world_to_view;
    let pos_clip = pos_view * locals.view_to_clip;
    *a_position = pos_clip;
}

#[spirv(block)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocalsSkybox {
    view_to_world: f32x4x4,
    clip_to_view: f32x4x4,
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn skybox_vs(
    #[spirv(vertex_id)] vert_id: Input<i32>,
    #[spirv(position)] mut a_position: Output<f32x4>,
    mut a_view_dir: Output<f32x3>,
    #[spirv(binding = 0)] u_locals: Uniform<LocalsSkybox>,
) {
    let position_uv = flink::geometry::Fullscreen::position(*vert_id);
    let position_clip = vec4(position_uv.x, position_uv.y, 0.0, 1.0);
    let mut position_view = position_clip * u_locals.clip_to_view;
    position_view.w = 0.0;
    let position_world = position_view * u_locals.view_to_world;

    *a_view_dir = vec3(position_world.x, position_world.y, position_world.z);
    *a_position = position_clip;
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn skybox_fs(
    f_view_dir: Input<f32x3>,
    #[spirv(binding = 0)] u_diffuse_map: UniformConstant<SampledImage<ImageCube>>,
    mut output: Output<f32x4>,
) {
    let sky = u_diffuse_map.sample(*f_view_dir);
    *output = vec4(sky.x, sky.y, sky.z, 1.0);
}
