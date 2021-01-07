use std::fs;
use std::mem;
use std::path::Path;
use std::time::Instant;

use self::camera::{Camera, InputMap};
use flink::{f32x3, f32x4, f32x4x4, vec3, vec4};
use glutin::event::{DeviceEvent, Event, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};

mod camera;
mod ktx;

#[repr(C)]
#[derive(Debug)]
struct LocalsPbr {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
    eye_world: f32x4,
    specular_mipmaps: u32,
}

#[repr(C)]
#[derive(Debug)]
struct LocalsSkybox {
    view_to_world: f32x4x4,
    clip_to_view: f32x4x4,
}

fn max_mip_levels_2d(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2() as u32 + 1
}

fn main() -> anyhow::Result<()> {
    unsafe {
        let begin = Instant::now();
        let el = EventLoop::new();
        println!("{:?}", begin.elapsed());
        let wb = glutin::window::WindowBuilder::new().with_title("grr - cyberpunk");
        let window = glutin::ContextBuilder::new()
            .with_srgb(true)
            .with_multisampling(4)
            .build_windowed(wb, &el)?
            .make_current()
            .unwrap();
        println!("{:?}", begin.elapsed());

        let grr = grr::Device::new(
            |symbol| window.get_proc_address(symbol) as *const _,
            grr::Debug::Disable,
        );

        let directory = Path::new("assets");

        let load_png =
            |name: &str, format: grr::Format, downsample: bool| -> anyhow::Result<grr::Image> {
                let start = Instant::now();
                let path = directory.join(name);
                let img = image::open(&Path::new(&path)).unwrap().to_rgba8();
                let img_width = img.width();
                let img_height = img.height();
                let img_data = img.into_raw();
                println!("{:?}", start.elapsed());

                let texture = grr.create_image(
                    grr::ImageType::D2 {
                        width: img_width,
                        height: img_height,
                        layers: 1,
                        samples: 1,
                    },
                    format,
                    if downsample {
                        max_mip_levels_2d(img_width, img_height)
                    } else {
                        1
                    },
                )?;
                grr.copy_host_to_image(
                    &img_data,
                    texture,
                    grr::HostImageCopy {
                        host_layout: grr::MemoryLayout {
                            base_format: grr::BaseFormat::RGBA,
                            format_layout: grr::FormatLayout::U8,
                            row_length: img_width,
                            image_height: img_height,
                            alignment: 4,
                        },
                        image_subresource: grr::SubresourceLayers {
                            level: 0,
                            layers: 0..1,
                        },
                        image_offset: grr::Offset { x: 0, y: 0, z: 0 },
                        image_extent: grr::Extent {
                            width: img_width,
                            height: img_height,
                            depth: 1,
                        },
                    },
                );
                if downsample {
                    grr.generate_mipmaps(texture);
                }

                Ok(texture)
            };

        let num_indices = 70_074;
        let bin = std::fs::read(directory.join("SciFiHelmet.bin"))?;
        let index_buffer =
            grr.create_buffer_from_host(&bin[..num_indices * 4], grr::MemoryFlags::DEVICE_LOCAL)?;
        let vertex_buffer =
            grr.create_buffer_from_host(&bin[num_indices * 4..], grr::MemoryFlags::DEVICE_LOCAL)?;

        let albedo = load_png(
            "SciFiHelmet_BaseColor.png",
            grr::Format::R8G8B8A8_SRGB,
            true,
        )?;
        let normals = load_png("SciFiHelmet_Normal.png", grr::Format::R8G8B8_UNORM, true)?;
        let metal_roughness = load_png(
            "SciFiHelmet_MetallicRoughness.png",
            grr::Format::R8G8B8_UNORM,
            true,
        )?;
        let ambient_occlusion = load_png(
            "SciFiHelmet_AmbientOcclusion.png",
            grr::Format::R8G8B8_UNORM,
            true,
        )?;
        let lut_ggx = load_png("lut_ggx.png", grr::Format::R8G8B8_UNORM, false)?;

        let sampler = grr.create_sampler(grr::SamplerDesc {
            min_filter: grr::Filter::Linear,
            mag_filter: grr::Filter::Linear,
            mip_map: None,
            address: (
                grr::SamplerAddress::ClampBorder,
                grr::SamplerAddress::ClampBorder,
                grr::SamplerAddress::ClampBorder,
            ),
            lod_bias: 0.0,
            lod: 0.0..10.0,
            compare: None,
            border_color: [0.0, 0.0, 0.0, 0.0],
        })?;

        let vertex_array = grr.create_vertex_array(&[
            grr::VertexAttributeDesc {
                location: 0,
                binding: 0,
                format: grr::VertexFormat::Xyz32Float,
                offset: 0,
            },
            grr::VertexAttributeDesc {
                location: 1,
                binding: 1,
                format: grr::VertexFormat::Xyz32Float,
                offset: 0,
            },
            grr::VertexAttributeDesc {
                location: 2,
                binding: 2,
                format: grr::VertexFormat::Xy32Float,
                offset: 0,
            },
            grr::VertexAttributeDesc {
                location: 3,
                binding: 3,
                format: grr::VertexFormat::Xyzw32Float,
                offset: 0,
            },
        ])?;

        let empty_array = grr.create_vertex_array(&[])?;

        let spirv = include_bytes!(env!("shader.spv"));

        let pbr_vs = grr.create_shader(
            grr::ShaderStage::Vertex,
            grr::ShaderSource::Spirv {
                entrypoint: "main_vs",
            },
            &spirv[..],
            grr::ShaderFlags::VERBOSE,
        )?;
        let pbr_fs = grr.create_shader(
            grr::ShaderStage::Fragment,
            grr::ShaderSource::Spirv {
                entrypoint: "main_fs",
            },
            &spirv[..],
            grr::ShaderFlags::VERBOSE,
        )?;

        let pbr_pipeline = grr.create_graphics_pipeline(
            grr::VertexPipelineDesc {
                vertex_shader: pbr_vs,
                tessellation_control_shader: None,
                tessellation_evaluation_shader: None,
                geometry_shader: None,
                fragment_shader: Some(pbr_fs),
            },
            grr::PipelineFlags::VERBOSE,
        )?;

        let pbr_state_ds = grr::DepthStencil {
            depth_test: true,
            depth_write: true,
            depth_compare_op: grr::Compare::LessEqual,
            stencil_test: false,
            stencil_front: grr::StencilFace::KEEP,
            stencil_back: grr::StencilFace::KEEP,
        };

        let skybox_vs = grr.create_shader(
            grr::ShaderStage::Vertex,
            grr::ShaderSource::Spirv {
                entrypoint: "skybox_vs",
            },
            &spirv[..],
            grr::ShaderFlags::VERBOSE,
        )?;
        let skybox_fs = grr.create_shader(
            grr::ShaderStage::Fragment,
            grr::ShaderSource::Spirv {
                entrypoint: "skybox_fs",
            },
            &spirv[..],
            grr::ShaderFlags::VERBOSE,
        )?;

        let skybox_pipeline = grr.create_graphics_pipeline(
            grr::VertexPipelineDesc {
                vertex_shader: skybox_vs,
                tessellation_control_shader: None,
                tessellation_evaluation_shader: None,
                geometry_shader: None,
                fragment_shader: Some(skybox_fs),
            },
            grr::PipelineFlags::VERBOSE,
        )?;

        let skybox_state_ds = grr::DepthStencil {
            depth_test: false,
            depth_write: false,
            depth_compare_op: grr::Compare::LessEqual,
            stencil_test: false,
            stencil_front: grr::StencilFace::KEEP,
            stencil_back: grr::StencilFace::KEEP,
        };

        let specular = fs::read(directory.join("specular.ktx2"))?;
        let specular_raw = ktx::Image::new(&specular)?;
        let specular_map = grr.create_image(
            grr::ImageType::D2 {
                width: specular_raw.header.pixel_width,
                height: specular_raw.header.pixel_height,
                layers: 6,
                samples: 1,
            },
            grr::Format::R16G16B16A16_SFLOAT,
            specular_raw.header.level_count,
        )?;
        let specular_view = grr.create_image_view(
            specular_map,
            grr::ImageViewType::Cube,
            grr::Format::R16G16B16A16_SFLOAT,
            grr::SubresourceRange {
                levels: 0..specular_raw.header.level_count,
                layers: 0..6,
            },
        )?;
        println!("Uploading specular image into GPU memory");
        for (i, level) in specular_raw.levels.iter().enumerate() {
            grr.copy_host_to_image(
                level,
                specular_map,
                grr::HostImageCopy {
                    host_layout: grr::MemoryLayout {
                        base_format: grr::BaseFormat::RGBA,
                        format_layout: grr::FormatLayout::F16,
                        row_length: specular_raw.header.pixel_width >> i,
                        image_height: specular_raw.header.pixel_height >> i,
                        alignment: 4,
                    },
                    image_subresource: grr::SubresourceLayers {
                        level: i as _,
                        layers: 0..6,
                    },
                    image_offset: grr::Offset { x: 0, y: 0, z: 0 },
                    image_extent: grr::Extent {
                        width: specular_raw.header.pixel_width >> i,
                        height: specular_raw.header.pixel_height >> i,
                        depth: 1,
                    },
                },
            );
        }

        let diffuse = fs::read(directory.join("diffuse.ktx2"))?;
        let diffuse_raw = ktx::Image::new(&diffuse)?;
        let diffuse_map = grr.create_image(
            grr::ImageType::D2 {
                width: diffuse_raw.header.pixel_width,
                height: diffuse_raw.header.pixel_height,
                layers: 6,
                samples: 1,
            },
            grr::Format::R16G16B16A16_SFLOAT,
            diffuse_raw.header.level_count,
        )?;
        let diffuse_view = grr.create_image_view(
            diffuse_map,
            grr::ImageViewType::Cube,
            grr::Format::R16G16B16A16_SFLOAT,
            grr::SubresourceRange {
                levels: 0..diffuse_raw.header.level_count,
                layers: 0..6,
            },
        )?;
        println!("Uploading diffuse image into GPU memory");
        for (i, level) in diffuse_raw.levels.iter().enumerate() {
            grr.copy_host_to_image(
                level,
                diffuse_map,
                grr::HostImageCopy {
                    host_layout: grr::MemoryLayout {
                        base_format: grr::BaseFormat::RGBA,
                        format_layout: grr::FormatLayout::F16,
                        row_length: diffuse_raw.header.pixel_width >> i,
                        image_height: diffuse_raw.header.pixel_height >> i,
                        alignment: 4,
                    },
                    image_subresource: grr::SubresourceLayers {
                        level: i as _,
                        layers: 0..6,
                    },
                    image_offset: grr::Offset { x: 0, y: 0, z: 0 },
                    image_extent: grr::Extent {
                        width: diffuse_raw.header.pixel_width >> i,
                        height: diffuse_raw.header.pixel_height >> i,
                        depth: 1,
                    },
                },
            );
        }

        let mut camera = Camera::new(vec3(0.0, 0.0, 0.0), 0.0, 0.0);
        let mut input = InputMap::new();

        el.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::LoopDestroyed => return,
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(physical_size) => {
                        window.resize(physical_size);
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                Event::DeviceEvent { event, .. } => match event {
                    DeviceEvent::MouseMotion { delta } => {
                        input.update_mouse_motion((delta.0 as _, delta.1 as _));
                    }
                    DeviceEvent::Button { state, .. } => {
                        input.update_mouse1(state);
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    camera.update(&input);
                    input.reset_delta();

                    let size = window.window().inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    let eye = camera.position() + camera.view_dir() * 4.5;
                    let center = camera.position();

                    let locals = LocalsPbr {
                        world_to_view: f32x4x4::look_at_inv(eye, eye - center),
                        view_to_clip: f32x4x4::perspective(
                            std::f32::consts::PI * 0.25,
                            aspect,
                            0.1,
                            10000.0,
                        ),
                        eye_world: vec4(eye.x, eye.y, eye.z, 0.0),
                        specular_mipmaps: specular_raw.header.level_count,
                    };
                    let u_locals = grr
                        .create_buffer_from_host(
                            grr::as_u8_slice(&[locals]),
                            grr::MemoryFlags::DEVICE_LOCAL,
                        )
                        .unwrap();
                    let locals_inv = LocalsSkybox {
                        view_to_world: f32x4x4::look_at(eye, eye - center),
                        clip_to_view: f32x4x4::perspective_inv(
                            std::f32::consts::PI * 0.25,
                            aspect,
                            0.1,
                            10000.0,
                        ),
                    };
                    let u_locals_inv = grr
                        .create_buffer_from_host(
                            grr::as_u8_slice(&[locals_inv]),
                            grr::MemoryFlags::DEVICE_LOCAL,
                        )
                        .unwrap();

                    grr.bind_framebuffer(grr::Framebuffer::DEFAULT);
                    grr.set_viewport(
                        0,
                        &[grr::Viewport {
                            x: 0.0,
                            y: 0.0,
                            w: size.width as _,
                            h: size.height as _,
                            n: 0.0,
                            f: 1.0,
                        }],
                    );
                    grr.set_scissor(
                        0,
                        &[grr::Region {
                            x: 0,
                            y: 0,
                            w: size.width as _,
                            h: size.height as _,
                        }],
                    );

                    grr.clear_attachment(
                        grr::Framebuffer::DEFAULT,
                        grr::ClearAttachment::ColorFloat(0, [0.0, 0.0, 0.0, 1.0]),
                    );
                    grr.clear_attachment(
                        grr::Framebuffer::DEFAULT,
                        grr::ClearAttachment::Depth(1.0),
                    );

                    grr.bind_pipeline(skybox_pipeline);
                    grr.bind_vertex_array(empty_array);
                    grr.bind_depth_stencil_state(&skybox_state_ds);
                    grr.bind_uniform_buffers(
                        0,
                        &[grr::BufferRange {
                            buffer: u_locals_inv,
                            offset: 0,
                            size: std::mem::size_of::<LocalsSkybox>() as _,
                        }],
                    );
                    grr.bind_image_views(0, &[diffuse_view]);
                    grr.bind_samplers(0, &[sampler]);
                    grr.draw(grr::Primitive::Triangles, 0..3, 0..1);

                    grr.bind_vertex_array(vertex_array);
                    grr.bind_pipeline(pbr_pipeline);
                    grr.bind_depth_stencil_state(&pbr_state_ds);
                    grr.bind_vertex_buffers(
                        vertex_array,
                        0,
                        &[
                            // position
                            grr::VertexBufferView {
                                buffer: vertex_buffer,
                                offset: 0,
                                stride: (3 * mem::size_of::<f32>()) as _,
                                input_rate: grr::InputRate::Vertex,
                            },
                            // normal
                            grr::VertexBufferView {
                                buffer: vertex_buffer,
                                offset: 840_888,
                                stride: (3 * mem::size_of::<f32>()) as _,
                                input_rate: grr::InputRate::Vertex,
                            },
                            // texcoord
                            grr::VertexBufferView {
                                buffer: vertex_buffer,
                                offset: 2_802_960,
                                stride: (2 * mem::size_of::<f32>()) as _,
                                input_rate: grr::InputRate::Vertex,
                            },
                            // tangent
                            grr::VertexBufferView {
                                buffer: vertex_buffer,
                                offset: 1_681_776,
                                stride: (4 * mem::size_of::<f32>()) as _,
                                input_rate: grr::InputRate::Vertex,
                            },
                        ],
                    );
                    grr.bind_index_buffer(vertex_array, index_buffer);
                    grr.bind_uniform_buffers(
                        0,
                        &[grr::BufferRange {
                            buffer: u_locals,
                            offset: 0,
                            size: std::mem::size_of::<LocalsPbr>() as _,
                        }],
                    );
                    grr.bind_image_views(
                        0,
                        &[
                            albedo.as_view(),
                            normals.as_view(),
                            metal_roughness.as_view(),
                            ambient_occlusion.as_view(),
                            diffuse_view,
                            specular_view,
                            lut_ggx.as_view(),
                        ],
                    );
                    grr.bind_samplers(0, &[sampler]);

                    grr.draw_indexed(
                        grr::Primitive::Triangles,
                        grr::IndexTy::U32,
                        0..num_indices as _,
                        0..1,
                        0,
                    );
                    grr.delete_buffer(u_locals);
                    grr.delete_buffer(u_locals_inv);

                    window.swap_buffers().unwrap();
                }
                _ => (),
            }
        })
    }
}
