use crate::vertex::{make_square_indices, Index, Vertex};

use nalgebra_glm::{identity, TMat4, TVec3, TVec4};

#[derive(Default, Clone)]
pub struct Model {
    vertices: Vec<Vertex>,
    indices: Vec<Index>,
    matrix: TMat4<f32>,
}

#[allow(dead_code)]
impl Model {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<Index>) -> Self {
        Self {
            vertices,
            indices,
            matrix: identity(),
        }
    }

    pub fn new_cube(vertices: Vec<Vertex>) -> Self {
        Self {
            vertices: vertices.clone(),
            indices: make_square_indices(&vertices),
            matrix: identity(),
        }
    }

    pub fn indices(self: &Self) -> Vec<Index> {
        self.indices.clone()
    }

    pub fn vertices(self: &Self) -> Vec<Vertex> {
        self.vertices.clone()
    }

    pub fn matrix(self: &Self) -> TMat4<f32> {
        self.matrix.clone()
    }

    pub fn set_matrix(self: &mut Self, matrix: TMat4<f32>) {
        self.matrix = matrix;
    }
}

#[derive(Default, Clone)]
pub struct ModelCollection {
    vertices: Vec<Vertex>,
    indices: Vec<Index>,
}

impl ModelCollection {
    pub fn from_vec(models: Vec<Model>) -> Self {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<Index> = Vec::new();

        for i in 0..models.len() {
            let mut model_vertices: Vec<Vertex> = models[i]
                .vertices()
                .iter_mut()
                .map(|v| {
                    // Must use w coordinate so translation is applied
                    let new_vertex_position = models[i].matrix()
                        * TVec4::<f32>::new(v.position[0], v.position[1], v.position[2], 1.0);

                    let new_vertex_normals = models[i].matrix().transform_vector(
                        &TVec3::<f32>::new(v.normal[0], v.normal[1], v.normal[2]),
                    );

                    Vertex::new(
                        [
                            new_vertex_position.x,
                            new_vertex_position.y,
                            new_vertex_position.z,
                        ],
                        [
                            new_vertex_normals.x,
                            new_vertex_normals.y,
                            new_vertex_normals.z,
                        ],
                    )
                })
                .collect();

            let mut model_indices: Vec<Index> = models[i]
                .indices()
                .iter_mut()
                .map(|&mut i| i + vertices.len() as Index)
                .collect();

            // let model_vertices: Vec<Vertex> = Vec::new();

            // for v in models[i].vertices().iter() {
            //     let new_vertex_position = models[i].matrix().transform_vector(&TVec3::<f32>::new(
            //         v.position[0],
            //         v.position[1],
            //         v.position[2],
            //     ));

            //     let new_vertex_normals = models[i].matrix().transform_vector(&TVec3::<f32>::new(
            //         v.normal[0],
            //         v.normal[1],
            //         v.normal[2],
            //     ));

            //     model_vertices.push(Vertex::new(
            //         [
            //             new_vertex_position.x,
            //             new_vertex_position.y,
            //             new_vertex_position.z,
            //         ],
            //         [
            //             new_vertex_normals.x,
            //             new_vertex_normals.y,
            //             new_vertex_normals.z,
            //         ],
            //     ));
            // }

            // let mut model_indices: Vec<Index> = Vec::new();

            // for i in models[i].indices().iter() {
            //     model_indices.push(i + vertices.len() as Index);
            // }

            vertices.append(&mut model_vertices);
            indices.append(&mut model_indices);
        }

        Self { vertices, indices }
    }

    pub fn vertices(self: &Self) -> Vec<Vertex> {
        assert!(self.vertices.len() > 0);

        self.vertices.clone()
    }

    pub fn indices(self: &Self) -> Vec<Index> {
        assert!(self.indices.len() > 0);

        self.indices.clone()
    }
}
