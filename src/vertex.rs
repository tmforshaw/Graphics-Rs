use bytemuck::{Pod, Zeroable};

type CompactVec3 = [f32; 3];

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: CompactVec3,
    pub normal: CompactVec3,
}

impl Vertex {
    pub const fn new(position: CompactVec3, normal: CompactVec3) -> Self {
        Vertex { position, normal }
    }
}

vulkano::impl_vertex!(Vertex, position, normal);

pub const CUBE_VERTICES: [Vertex; 4 * 6] = [
    // Front Face
    Vertex::new([-0.5, 0.5, -0.5], [0.0, 0.0, -1.0]),
    Vertex::new([0.5, -0.5, -0.5], [0.0, 0.0, -1.0]),
    Vertex::new([0.5, 0.5, -0.5], [0.0, 0.0, -1.0]),
    Vertex::new([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0]),
    // Back Face
    Vertex::new([-0.5, 0.5, 0.5], [0.0, 0.0, 1.0]),
    Vertex::new([0.5, -0.5, 0.5], [0.0, 0.0, 1.0]),
    Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0]),
    Vertex::new([0.5, 0.5, 0.5], [0.0, 0.0, 1.0]),
    // Left Face
    Vertex::new([-0.5, 0.5, -0.5], [-1.0, 0.0, 0.0]),
    Vertex::new([-0.5, -0.5, 0.5], [-1.0, 0.0, 0.0]),
    Vertex::new([-0.5, -0.5, -0.5], [-1.0, 0.0, 0.0]),
    Vertex::new([-0.5, 0.5, 0.5], [-1.0, 0.0, 0.0]),
    // Right Face
    Vertex::new([0.5, 0.5, -0.5], [1.0, 0.0, 0.0]),
    Vertex::new([0.5, -0.5, 0.5], [1.0, 0.0, 0.0]),
    Vertex::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]),
    Vertex::new([0.5, -0.5, -0.5], [1.0, 0.0, 0.0]),
    // Top Face
    Vertex::new([-0.5, -0.5, -0.5], [0.0, -1.0, 0.0]),
    Vertex::new([0.5, -0.5, 0.5], [0.0, -1.0, 0.0]),
    Vertex::new([0.5, -0.5, -0.5], [0.0, -1.0, 0.0]),
    Vertex::new([-0.5, -0.5, 0.5], [0.0, -1.0, 0.0]),
    // Bottom Face
    Vertex::new([-0.5, 0.5, -0.5], [0.0, 1.0, 0.0]),
    Vertex::new([0.5, 0.5, 0.5], [0.0, 1.0, 0.0]),
    Vertex::new([-0.5, 0.5, 0.5], [0.0, 1.0, 0.0]),
    Vertex::new([0.5, 0.5, -0.5], [0.0, 1.0, 0.0]),
];
