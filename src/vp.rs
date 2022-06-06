use nalgebra_glm::{identity, TMat4, TVec3};

use std::time::Instant;

const PI: f32 = 3.1415926535f32;

#[derive(Default, Clone)]
pub struct VP {
    pub view: TMat4<f32>,
    pub proj: TMat4<f32>,
}

#[allow(dead_code)]
impl VP {
    pub fn new() -> Self {
        Self {
            view: identity(),
            proj: identity(),
        }
    }
}

pub fn get_vp(dimensions: winit::dpi::PhysicalSize<u32>) -> VP {
    let view = nalgebra_glm::look_at_rh(
        &TVec3::new(0f32, 0f32, -1f32),
        &TVec3::new(0f32, 0f32, 1f32),
        &TVec3::<f32>::new(0f32, -1f32, 0f32),
    );

    let proj = nalgebra_glm::perspective(
        (dimensions.width as f32) / (dimensions.height as f32),
        PI * 0.5f32,
        0.05f32,
        100f32,
    );

    VP { view, proj }
}

pub fn get_model(time: Instant) -> TMat4<f32> {
    let rotation = nalgebra_glm::rotation(
        time.elapsed().as_secs_f32() * 2f32,
        &TVec3::new(0.5f32, -0.5f32, 0.5f32),
    );

    let translation = nalgebra_glm::translation(&TVec3::new(
        0f32,
        -5f32,
        20f32 + 15f32 * (time.elapsed().as_secs_f32() * 2f32).sin(),
    ));

    translation * rotation
}

pub fn get_model_2(time: Instant) -> TMat4<f32> {
    let rotation = nalgebra_glm::rotation(
        time.elapsed().as_secs_f32() * 10f32,
        &TVec3::new(0.2f32, 0.0f32, 0.5f32),
    );

    let translation = nalgebra_glm::translation(&TVec3::new(
        3f32 * (time.elapsed().as_secs_f32() * 1f32).sin(),
        2f32 + 3f32 * (time.elapsed().as_secs_f32() * 1f32).cos(),
        15f32,
    ));

    translation * rotation
}
