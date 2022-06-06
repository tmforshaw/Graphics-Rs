use vulkano::buffer::{CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::PersistentDescriptorSet;
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
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError};

use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;

use crate::vertex::{Index, Vertex};

pub fn create_instance() -> Arc<Instance> {
    Instance::new(InstanceCreateInfo {
        enabled_extensions: vulkano_win::required_extensions(),
        ..Default::default()
    })
    .expect("failed to create instance")
}

pub fn select_physical_device<'a>(
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

pub fn get_devices_surface_queue<'a>(
    event_loop: &EventLoop<()>,
    instance: &'a Arc<Instance>,
) -> (
    PhysicalDevice<'a>,
    Arc<Device>,
    Arc<Queue>,
    Arc<Surface<Window>>,
) {
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

    (physical_device, device, queue, surface)
}

pub fn new_swapchain_images(
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

pub fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain<Window>>) -> Arc<RenderPass> {
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

pub fn new_attachment_image(
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

pub fn get_framebuffers(
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

pub fn recreate_swapchain(
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

pub fn get_pipeline(
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

pub fn get_pipeline_with_depth(
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

const BG_COL: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

pub fn get_command_buffers(
    device: Arc<Device>,
    queue: Arc<Queue>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    deferred_set: Arc<PersistentDescriptorSet>,
    lighting_pipeline: Arc<GraphicsPipeline>,
    lighting_set: Arc<PersistentDescriptorSet>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[Index]>>,
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
