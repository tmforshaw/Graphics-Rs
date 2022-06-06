use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::pipeline::{graphics::viewport::Viewport, Pipeline};
use vulkano::render_pass::Subpass;
use vulkano::swapchain::AcquireError;
use vulkano::sync::{self, FenceSignalFuture, FlushError, GpuFuture};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use std::sync::Arc;
use std::time::Instant;

// Modules

mod camera;
mod light;
mod model;
mod pipeline_commands;
mod shader;
pub mod vertex;
mod vp;

use camera::Camera;
use light::Light;
use model::{Model, ModelCollection};
use pipeline_commands::{
    create_instance, get_command_buffers, get_devices_surface_queue, get_framebuffers,
    get_pipeline, get_pipeline_with_depth, get_render_pass, new_attachment_image,
    new_swapchain_images, recreate_swapchain,
};
use shader::{deferred_frag, deferred_vert, lighting_frag, lighting_vert};

fn main() {
    let event_loop = EventLoop::new();

    let instance = create_instance();
    let (physical_device, device, queue, surface) =
        get_devices_surface_queue(&event_loop, &instance);

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

    let cube = Model::new_cube(vertex::CUBE_VERTICES.clone().to_vec());

    let model_vec = vec![cube.clone(), cube.clone()];

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag::load(device.clone()).unwrap();

    let vp_buffer = CpuBufferPool::<deferred_vert::ty::VpData>::uniform_buffer(device.clone());
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

            let mut model_vec_clone = model_vec.clone();

            model_vec_clone[0].set_matrix(vp::get_model(time.clone()));
            model_vec_clone[1].set_matrix(vp::get_model_2(time.clone()));

            let models = ModelCollection::from_vec(model_vec_clone.clone());

            let vertex_buffer_e = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                models.vertices().into_iter(),
            )
            .unwrap();

            let index_buffer_e = CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::index_buffer(),
                false,
                models.indices().into_iter(),
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

            let vp_buffer_subbuffer = {
                let vp = vp::get_vp(dimensions);
                let vp_data = deferred_vert::ty::VpData {
                    view: vp.view.into(),
                    proj: vp.proj.into(),
                };

                vp_buffer.next(vp_data).unwrap()
            };

            let lighting_buffer_subbuffer = {
                let light = Light::new(
                    // Position
                    [0.0, 0.0, -1.0],
                    // Colour
                    [
                        ((time.elapsed().as_secs_f32() * 3f32).sin() + 1.0) * 0.5,
                        ((time.elapsed().as_secs_f32()).cos() + 1.0) * 0.,
                        1.0,
                    ],
                    // Intensity
                    1.0,
                );
                let light_data = lighting_frag::ty::LightData {
                    _dummy0: [0; 4],
                    position: light.position().into(),
                    colour: light.colour().into(),
                    intensity: light.intensity().into(),
                };

                lighting_buffer.next(light_data).unwrap()
            };

            let camera_buffer_subbuffer = {
                let camera = Camera::new(
                    [0.0, 0.0, 0.0],
                    time.elapsed().as_secs().try_into().unwrap(),
                );
                let camera_data = lighting_frag::ty::CameraData {
                    position: camera.position().into(),
                    dt: camera.dt().into(),
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
                [WriteDescriptorSet::buffer(0, vp_buffer_subbuffer.clone())],
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
                    WriteDescriptorSet::buffer(2, vp_buffer_subbuffer),
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
