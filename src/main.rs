use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, AttachmentImage, ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::{self, FenceSignalFuture, FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;

use winit::window::Window;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use nalgebra_glm::{identity, TMat4, TVec3};

use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Default, Clone)]
struct MVP {
    model: TMat4<f32>,
    view: TMat4<f32>,
    proj: TMat4<f32>,
}

#[allow(dead_code)]
impl MVP {
    fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            proj: identity(),
        }
    }
}

vulkano::impl_vertex!(Vertex, position, normal);

const BG_COL: [f32; 4] = [0.40, 0.40, 0.40, 1.0];

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    colour_buffer: Arc<ImageView<AttachmentImage>>,
    normal_buffer: Arc<ImageView<AttachmentImage>>,
    depth_buffer: Arc<ImageView<AttachmentImage>>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        normal_buffer.clone(),
                        colour_buffer.clone(),
                        depth_buffer.clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");

    (physical_device, queue_family)
}

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain<Window>>) -> Arc<RenderPass> {
    vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
                final_colour: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),  // set the format the same as the swapchain
                    samples: 1,
                },

                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16_SFLOAT,  // set the format the same as the swapchain
                    samples: 1,
                },
                colour: {
                    load: Clear,
                    store: DontCare,
                    format: Format::A2B10G10R10_UNORM_PACK32,  // set the format the same as the swapchain
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
        passes: [
            {
                color: [normals, colour],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_colour],
                depth_stencil: {},
                input: [normals, colour]
            }
        ]
    )
    .unwrap()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    subpass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(subpass)
        .build(device.clone())
        .unwrap()
}

fn get_pipeline_with_depth(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    subpass: Subpass,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(subpass)
        .build(device.clone())
        .unwrap()
}

fn get_command_buffers(
    device: Arc<Device>,
    queue: Arc<Queue>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    deferred_set: Arc<PersistentDescriptorSet>,
    lighting_pipeline: Arc<GraphicsPipeline>,
    lighting_set: Arc<PersistentDescriptorSet>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit, // don't forget to write the correct buffer usage
            )
            .unwrap();

            builder
                .begin_render_pass(
                    framebuffer.clone(),
                    SubpassContents::Inline,
                    vec![BG_COL.into(), BG_COL.into(), BG_COL.into(), 1f32.into()], // Use 1f32 for depth clear to give unique colour
                )
                .unwrap()
                .bind_pipeline_graphics(deferred_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    deferred_pipeline.layout().clone(),
                    0,
                    deferred_set.clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .bind_index_buffer(index_buffer.clone())
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap()
                .next_subpass(SubpassContents::Inline)
                .unwrap()
                .bind_pipeline_graphics(lighting_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    lighting_pipeline.layout().clone(),
                    0,
                    lighting_set.clone(),
                )
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn get_mvp(dimensions: winit::dpi::PhysicalSize<u32>, dt: Duration) -> MVP {
    let rotation = nalgebra_glm::rotation(
        dt.as_millis() as f32 * 0.002,
        &TVec3::new(0.5f32, -0.5f32, 0.5f32),
    );

    let model: TMat4<f32> = nalgebra_glm::translation(&TVec3::new(0f32, 0f32, 1f32)) * rotation;
    let view = nalgebra_glm::look_at_rh(
        &TVec3::new(0f32, 0f32, -1f32),
        &TVec3::new(0f32, 0f32, 1f32),
        &TVec3::<f32>::new(0f32, -1f32, 0f32),
    );
    let proj = nalgebra_glm::perspective(
        (dimensions.width as f32) / (dimensions.height as f32),
        ((dt.as_secs_f32() * 0.8f32).sin()).abs() + 80f32,
        0.05f32,
        1000f32,
    );

    MVP { model, view, proj }
}

fn make_square_indices(vertices: &Vec<Vertex>) -> Vec<u32> {
    assert_eq!(vertices.len() % 4, 0);

    let mut indices: Vec<u32> = Vec::new();

    for i in 0..vertices.len() / 4 {
        for j in 0..6 {
            if j < 4 {
                indices.push((i * 4 + j) as u32);
            } else {
                indices.push((i * 4 + 5 - j) as u32);
            }
        }
    }

    indices
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .expect("failed to create instance");

    let event_loop = EventLoop::new(); // ignore this for now
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, surface.clone(), &device_extensions);

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions), // new
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("failed to get surface capabilities");

    let mut dimensions = surface.window().inner_size();
    let composite_alpha = capabilities
        .supported_composite_alpha
        .iter()
        .next()
        .unwrap();
    let image_format = Some(
        physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1, // How many buffers to use in the swapchain
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::color_attachment(), // What the images are going to be used for
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    let mut render_pass = get_render_pass(device.clone(), swapchain.clone());

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions.clone().into(), Format::D16_UNORM)
            .unwrap(),
    )
    .unwrap();

    let mut normal_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions.clone().into(),
            Format::R16G16B16A16_SFLOAT,
        )
        .unwrap(),
    )
    .unwrap();

    let mut colour_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions.clone().into(),
            Format::A2B10G10R10_UNORM_PACK32,
        )
        .unwrap(),
    )
    .unwrap();

    let mut framebuffers = get_framebuffers(
        &images,
        render_pass.clone(),
        colour_buffer.clone(),
        normal_buffer.clone(),
        depth_buffer.clone(),
    );

    let vertices = vec![
        // Front Face
        Vertex {
            position: [-0.5, 0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
        },
        Vertex {
            position: [-0.5, -0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
        },
        // Back Face
        Vertex {
            position: [-0.5, 0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
        }, // Left Face
        Vertex {
            position: [-0.5, 0.5, -0.5],
            normal: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            normal: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5, -0.5],
            normal: [-1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            normal: [-1.0, 0.0, 0.0],
        },
        // Right Face
        Vertex {
            position: [0.5, 0.5, -0.5],
            normal: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            normal: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            normal: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            normal: [1.0, 0.0, 0.0],
        },
        // Top Face
        Vertex {
            position: [-0.5, -0.5, -0.5],
            normal: [0.0, -1.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            normal: [0.0, -1.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            normal: [0.0, -1.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            normal: [0.0, -1.0, 0.0],
        },
        // Bottom Face
        Vertex {
            position: [-0.5, 0.5, -0.5],
            normal: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            normal: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            normal: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            normal: [0.0, 1.0, 0.0],
        },
    ];

    let indices = make_square_indices(&vertices); // [0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 5]

    // println!("{:?}\n{}", indices, indices.len());

    // mod vs {
    //     vulkano_shaders::shader! {
    //         ty: "vertex",
    //         src: "
    //             #version 450

    //             layout(location = 0) in vec3 position;
    //             layout(location = 1) in vec3 normal;

    //             layout(location = 0) out vec3 o_normal;
    //             layout(location = 1) out vec3 o_colour;
    //             layout(location = 2) out vec3 o_fragPos;

    //             layout(set = 0, binding = 0) uniform MVP_Data {
    //                 mat4 model;
    //                 mat4 view;
    //                 mat4 proj;
    //             } uniforms;

    //             void main() {
    //                         o_normal = mat3(uniforms.model) * normal;
    //                         o_colour = normalize(normal) * 0.5 + 0.5;
    //                         o_fragPos = vec3(uniforms.model * vec4(position, 1.0));

    //                         mat4 mvp = uniforms.proj * uniforms.view * uniforms.model;

    //                         gl_Position = mvp * vec4(position, 1.0);
    //             }",
    //         types_meta: {
    //                 use bytemuck::{Zeroable, Pod};

    //                 #[derive(Clone, Copy, Zeroable, Pod)]
    //             },
    //     }
    // }

    // mod fs {
    //     vulkano_shaders::shader! {
    //         ty: "fragment",
    //         src: "#version 450

    //             layout(location = 0) in vec3 normal;
    //             layout(location = 1) in vec3 colour;
    //             layout(location = 2) in vec3 fragPos;

    //             layout(location = 0) out vec4 f_color;

    //             void main() {
    //                 vec3 lightCol = vec3(1.0);
    //                 vec3 lightPos = vec3(0.0, 0.0, -10.0);

    //                 vec3 lightDir = normalize(lightPos - fragPos);

    //                 vec3 ambient = vec3(0.1);

    //                 vec3 dir_colour = max(dot(normal, lightDir), 0.0) * lightCol;

    //                 f_color = vec4((dir_colour + ambient) * colour, 1.0);
    //             }",
    //     }
    // }

    // let vs = vs::load(device.clone()).expect("failed to create vertex shader module");
    // let fs = fs::load(device.clone()).expect("failed to create vertex shader module");

    mod deferred_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/deferred.vert.glsl",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    mod deferred_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/deferred.frag.glsl",
        }
    }

    mod lighting_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/lighting.vert.glsl",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    mod lighting_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/lighting.frag.glsl",
            // types_meta: {
            //     use bytemuck::{Pod, Zeroable};

            //     #[derive(Clone, Copy, Zeroable, Pod)]
            // },
        }
    }

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag::load(device.clone()).unwrap();

    let uniform_buffer =
        CpuBufferPool::<deferred_vert::ty::MVP_Data>::uniform_buffer(device.clone());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    let mut past_time = Instant::now();
    let time = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
        }
        Event::RedrawEventsCleared => {
            #[allow(unused_variables)]
            let dt = past_time.elapsed();
            past_time = Instant::now();

            if recreate_swapchain {
                recreate_swapchain = false;

                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;

                    let new_dimensions = surface.window().inner_size();
                    dimensions = new_dimensions;

                    let new_depth_buffer = ImageView::new_default(
                        AttachmentImage::transient(
                            device.clone(),
                            new_dimensions.into(),
                            Format::D16_UNORM,
                        )
                        .unwrap(),
                    )
                    .unwrap();

                    normal_buffer = ImageView::new_default(
                        AttachmentImage::transient_input_attachment(
                            device.clone(),
                            dimensions.clone().into(),
                            Format::R16G16B16A16_SFLOAT,
                        )
                        .unwrap(),
                    )
                    .unwrap();

                    colour_buffer = ImageView::new_default(
                        AttachmentImage::transient_input_attachment(
                            device.clone(),
                            dimensions.clone().into(),
                            Format::A2B10G10R10_UNORM_PACK32,
                        )
                        .unwrap(),
                    )
                    .unwrap();

                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: new_dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };
                    swapchain = new_swapchain.clone();

                    render_pass = get_render_pass(device.clone(), new_swapchain.clone());

                    let new_framebuffers = get_framebuffers(
                        &new_images,
                        render_pass.clone(),
                        colour_buffer.clone(),
                        normal_buffer.clone(),
                        new_depth_buffer.clone(),
                    );
                    framebuffers = new_framebuffers;
                }
            };

            let vertex_buffer_e = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                vertices.clone().into_iter(),
            )
            .unwrap();

            let index_buffer_e = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::index_buffer(),
                false,
                indices.clone().into_iter(),
            )
            .unwrap();

            viewport.dimensions = dimensions.into();

            let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
            let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

            let deferred_pipeline = get_pipeline_with_depth(
                device.clone(),
                deferred_vert.clone(),
                deferred_frag.clone(),
                deferred_pass.clone(),
                viewport.clone(),
            );

            let lighting_pipeline = get_pipeline(
                device.clone(),
                lighting_vert.clone(),
                lighting_frag.clone(),
                lighting_pass.clone(),
                viewport.clone(),
            );

            let uniform_buffer_subbuffer = {
                let mvp = get_mvp(dimensions, time.elapsed());
                let uniform_data = deferred_vert::ty::MVP_Data {
                    model: mvp.model.into(),
                    view: mvp.view.into(),
                    proj: mvp.proj.into(),
                };

                uniform_buffer.next(uniform_data).unwrap()
            };

            let deferred_layout = deferred_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .clone()
                .unwrap();
            // let mut deferred_set_builder = PersistentDescriptorSet::start(deferred_layout.clone());

            let deferred_set = PersistentDescriptorSet::new(
                deferred_layout.clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    uniform_buffer_subbuffer.clone(),
                )],
            )
            .unwrap();

            let lighting_layout = lighting_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .clone()
                .unwrap();
            let lighting_set = PersistentDescriptorSet::new(
                lighting_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, normal_buffer.clone()),
                    WriteDescriptorSet::image_view(1, colour_buffer.clone()),
                    WriteDescriptorSet::buffer(2, uniform_buffer_subbuffer.clone()),
                ],
            )
            .unwrap();

            // let owned_pipeline = new_pipeline.clone().to_owned();

            // let layout = owned_pipeline.layout().set_layouts().get(0).unwrap();

            // let set = PersistentDescriptorSet::new(
            //     layout.clone(),
            //     [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
            // )
            // .unwrap();

            let command_buffers = get_command_buffers(
                device.clone(),
                queue.clone(),
                deferred_pipeline.clone(),
                deferred_set.clone(),
                lighting_pipeline.clone(),
                lighting_set.clone(),
                &framebuffers,
                vertex_buffer_e.clone(),
                index_buffer_e.clone(),
            );

            let (image_i, suboptimal, acquire_future) =
                match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            if let Some(image_fence) = &fences[image_i] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match fences[previous_fence_i].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing FenceSignalFuture
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i].clone())
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_i)
                .then_signal_fence_and_flush();

            fences[image_i] = match future {
                Ok(value) => Some(Arc::new(value)),
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    None
                }
            };

            previous_fence_i = image_i;
        }
        Event::MainEventsCleared => {}
        _ => (),
    });
}
