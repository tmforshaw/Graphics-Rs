#[repr(C)]
#[derive(Default, Clone)]
pub struct Camera {
    position: [f32; 3],
    dt: u32,
}

impl Camera {
    pub fn new(position: [f32; 3], dt: u32) -> Self {
        Self { position, dt }
    }

    pub fn position(self: &Self) -> [f32; 3] {
        self.position.clone()
    }

    pub fn dt(self: &Self) -> u32 {
        self.dt.clone()
    }
}
