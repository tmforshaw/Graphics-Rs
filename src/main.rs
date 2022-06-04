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

use std::time::Instant;

mod mvp;
mod vertex;

use vertex::Vertex;

const BG_COL: [f32; 4] = [0.40, 0.40, 0.40, 1.0];

#[repr(C)]
#[derive(Default, Clone)]
struct Light {
    position: [f32; 3],
    colour: [f32; 3],
    intensity: f32,
}

impl Light {
    fn new(position: [f32; 3], colour: [f32; 3], intensity: f32) -> Self {
        Self {
            position,
            colour,
            intensity,
        }
    }
}

#[repr(C)]
#[derive(Default, Clone)]
struct Camera {
    position: [f32; 3],
    dt: u32,
}

impl Camera {
    fn new(position: [f32; 3], dt: u32) -> Self {
        Self { position, dt }
    }
}

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
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

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

fn new_attachment_image(
    device: Arc<Device>,
    dimensions: winit::dpi::PhysicalSize<u32>,
    format: Format,
) -> Arc<ImageView<AttachmentImage>> {
    ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions.clone().into(),
            format,
        )
        .unwrap(),
    )
    .unwrap()
}

fn recreate_swapchain(
    dimensions: winit::dpi::PhysicalSize<u32>,
    device: Arc<Device>,
    swapchain: Arc<Swapchain<Window>>,
) -> Option<(
    Arc<Swapchain<Window>>,
    winit::dpi::PhysicalSize<u32>,
    Vec<Arc<Framebuffer>>,
    Arc<RenderPass>,
    Arc<ImageView<AttachmentImage>>,
    Arc<ImageView<AttachmentImage>>,
)> {
    // Recreate attachment image buffers
    let depth_buffer = new_attachment_image(device.clone(), dimensions, Format::D16_UNORM);
    let normal_buffer =
        new_attachment_image(device.clone(), dimensions, Format::R16G16B16A16_SFLOAT);
    let colour_buffer =
        new_attachment_image(device.clone(), dimensions, Format::A2B10G10R10_UNORM_PACK32);

    let (new_swapchain, images) = match swapchain.recreate(SwapchainCreateInfo {
        image_extent: dimensions.into(),
        ..swapchain.create_info()
    }) {
        Ok(r) => r,
        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return Option::None,
        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
    };

    let render_pass = get_render_pass(device, swapchain);

    let framebuffers = get_framebuffers(
        &images,
        render_pass.clone(),
        colour_buffer.clone(),
        normal_buffer.clone(),
        depth_buffer.clone(),
    );

    Option::from((
        new_swapchain,
        dimensions,
        framebuffers,
        render_pass,
        normal_buffer,
        colour_buffer,
    ))
}

fn new_swapchain_images(
    device: Arc<Device>,
    physical_device: PhysicalDevice,
    surface: &Arc<Surface<Window>>,
) -> (
    Arc<Swapchain<Window>>,
    Vec<Arc<SwapchainImage<Window>>>,
    winit::dpi::PhysicalSize<u32>,
) {
    let capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("failed to get surface capabilities");

    let dimensions = surface.window().inner_size();
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

    let (swapchain, images) = Swapchain::new(
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

    (swapchain, images, dimensions)
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

    let (mut swapchain, images, mut dimensions) =
        new_swapchain_images(device.clone(), physical_device, &surface);

    let mut render_pass = get_render_pass(device.clone(), swapchain.clone());

    // Create attachment image buffers
    let depth_buffer = new_attachment_image(device.clone(), dimensions, Format::D16_UNORM);
    let mut normal_buffer =
        new_attachment_image(device.clone(), dimensions, Format::R16G16B16A16_SFLOAT);
    let mut colour_buffer =
        new_attachment_image(device.clone(), dimensions, Format::A2B10G10R10_UNORM_PACK32);

    let mut framebuffers = get_framebuffers(
        &images,
        render_pass.clone(),
        colour_buffer.clone(),
        normal_buffer.clone(),
        depth_buffer.clone(),
    );

    let vertices = vertex::CUBE_VERTICES.clone().to_vec();

    let indices = make_square_indices(&vertices.clone());

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag::load(device.clone()).unwrap();

    let mvp_buffer = CpuBufferPool::<deferred_vert::ty::MvpData>::uniform_buffer(device.clone());
    let lighting_buffer =
        CpuBufferPool::<lighting_frag::ty::LightData>::uniform_buffer(device.clone());
    let camera_buffer =
        CpuBufferPool::<lighting_frag::ty::CameraData>::uniform_buffer(device.clone());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let mut window_resized = false;
    let mut recreate_swapchain_b = false;

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

            if recreate_swapchain_b {
                recreate_swapchain_b = false;

                if window_resized || recreate_swapchain_b {
                    recreate_swapchain_b = false;

                    dimensions = surface.clone().window().inner_size();

                    (
                        swapchain,
                        dimensions,
                        framebuffers,
                        render_pass,
                        normal_buffer,
                        colour_buffer,
                    ) = recreate_swapchain(dimensions.clone(), device.clone(), swapchain.clone())
                        .unwrap();
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

            let mvp_buffer_subbuffer = {
                let mvp = mvp::get_mvp(dimensions, time.clone());
                let mvp_data = deferred_vert::ty::MvpData {
                    model: mvp.model.into(),
                    view: mvp.view.into(),
                    proj: mvp.proj.into(),
                };

                mvp_buffer.next(mvp_data).unwrap()
            };

            let lighting_buffer_subbuffer = {
                let light = Light::new([0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 1.0);
                let light_data = lighting_frag::ty::LightData {
                    _dummy0: [0; 4],
                    position: light.position.into(),
                    colour: light.colour.into(),
                    intensity: light.intensity.into(),
                };

                lighting_buffer.next(light_data).unwrap()
            };

            let camera_buffer_subbuffer = {
                let camera = Camera::new(
                    [0.0, 0.0, 0.0],
                    time.elapsed().as_secs().try_into().unwrap(),
                );
                let camera_data = lighting_frag::ty::CameraData {
                    position: camera.position.into(),
                    dt: camera.dt.into(),
                };

                camera_buffer.next(camera_data).unwrap()
            };

            let deferred_layout = deferred_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .clone()
                .unwrap();
            let deferred_set = PersistentDescriptorSet::new(
                deferred_layout.clone(),
                [WriteDescriptorSet::buffer(0, mvp_buffer_subbuffer.clone())],
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
                    WriteDescriptorSet::buffer(2, mvp_buffer_subbuffer),
                    WriteDescriptorSet::buffer(3, lighting_buffer_subbuffer),
                    WriteDescriptorSet::buffer(4, camera_buffer_subbuffer),
                ],
            )
            .unwrap();

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
                        recreate_swapchain_b = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain_b = true;
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
                    recreate_swapchain_b = true;
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
