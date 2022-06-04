use nalgebra_glm::{identity, TMat4, TVec3};

use std::time::Instant;

#[derive(Default, Clone)]
pub struct MVP {
    pub model: TMat4<f32>,
    pub view: TMat4<f32>,
    pub proj: TMat4<f32>,
}

#[allow(dead_code)]
impl MVP {
    pub fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            proj: identity(),
        }
    }
}

pub fn get_mvp(dimensions: winit::dpi::PhysicalSize<u32>, time: Instant) -> MVP {
    let rotation = nalgebra_glm::rotation(
        time.elapsed().as_secs_f32() * 2f32,
        &TVec3::new(0.5f32, -0.5f32, 0.5f32),
    );

    let model: TMat4<f32> = nalgebra_glm::translation(&TVec3::new(
        0f32,
        0f32,
        1.25f32 + (time.elapsed().as_secs_f32() * 2f32).sin(),
    )) * rotation;
    let view = nalgebra_glm::look_at_rh(
        &TVec3::new(0f32, 0f32, -1f32),
        &TVec3::new(0f32, 0f32, 1f32),
        &TVec3::<f32>::new(0f32, -1f32, 0f32),
    );

    let proj = nalgebra_glm::perspective(
        (dimensions.width as f32) / (dimensions.height as f32),
        90f32,
        0.05f32,
        100f32,
    );

    MVP { model, view, proj }
}
