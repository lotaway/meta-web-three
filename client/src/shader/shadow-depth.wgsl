//  wgsl着色器
//  顶点着色器主函数
/*@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
    //  内部设置三角形顶点坐标
    var pos = array<vec2<f32>, 6>(
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(-1.0, 1.0),
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}*/

@group(2) @binding(0) var<uniform> model_view: mat4x4<f32>;
@group(2) @binding(1) var<uniform> lightProjection: mat4x4<f32>;

struct Input {
    @builtin(instance_index) instance_index : u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn main(input: Input) -> @builtin(position) vec4<f32> {
    let modelview = model_view[input.instance_index];
    let pos = vec4(input.position, 1.0);
    return lightProjection * modelview * pos;
}