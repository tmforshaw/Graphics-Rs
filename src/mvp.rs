use nalgebra_glm::{identity, TMat4, TVec3};

use std::time::Duration;

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

pub fn get_mvp(dimensions: winit::dpi::PhysicalSize<u32>, dt: Duration) -> MVP {
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
        100f32,
    );

    MVP { model, view, proj }
}
