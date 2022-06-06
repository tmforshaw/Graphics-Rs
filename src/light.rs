use crate::vertex::CompactVec3;

#[repr(C)]
#[derive(Default, Clone)]
pub struct Light {
    position: CompactVec3,
    colour: CompactVec3,
    intensity: f32,
}

impl Light {
    pub fn new(position: CompactVec3, colour: CompactVec3, intensity: f32) -> Self {
        Self {
            position,
            colour,
            intensity,
        }
    }

    pub fn position(self: &Self) -> CompactVec3 {
        self.position.clone()
    }

    pub fn colour(self: &Self) -> CompactVec3 {
        self.colour.clone()
    }

    pub fn intensity(self: &Self) -> f32 {
        self.intensity.clone()
    }
}
