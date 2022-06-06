pub mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/deferred.vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/deferred.frag.glsl",
    }
}

pub mod lighting_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/lighting.vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod lighting_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/lighting.frag.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
