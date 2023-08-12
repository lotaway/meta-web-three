//  wgsl着色器
//  顶点着色器主函数
/*@vertex
fn vt_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
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

@group(0) @binding(0) var<uniform> m4s: mat4x4<f32>;
@group(0) @binding(1) var<uniform> m4t: mat4x4<f32>;
@group(2) @binding(0) var<uniform> animate: mat4x4<f32>;

struct Out {
    @builtin(position) v_pos: vec4<f32>,
    @location(0) f_pos: vec3<f32>,
    @location(1) v_colors: vec3<f32>,
}

@vertex
fn vt_main(@location(0) position: vec3<f32>, @location(1) colors: vec3<f32>) -> Out {
    var out: Out;
    var model_matrix = animate * m4s * m4t;
    out.v_pos = model_matrix * vec4<f32>(position, 1.0);
    out.f_pos = out.v_pos.xyz;
    out.v_colors = colors;
//    var m4s = default_m4s();
//    var m4t = default_m4t();
    return out;
}

fn matrix_create() -> mat4x4<f32> {
    return mat4x4<f32>(
        1.0,0.0,0.0,0.0,
        0.0,1.0,0.0,0.0,
        0.0,0.0,1.0,0.0,
        0.0,0.0,0.0,1.0,
    );
}

fn default_m4s() -> mat4x4<f32> {
    return mat4x4<f32>(
        0.5,0.0,0.0,0.0,
        0.0,0.5,0.0,0.0,
        0.0,0.0,0.5,0.0,
        0.0,0.0,0.0,1.0,
    );
}

fn default_m4t() -> mat4x4<f32> {
    return mat4x4<f32>(
        1.0,0.0,0.0,-0.5,
        0.0,1.0,0.0,-0.5,
        0.0,0.0,1.0,0.0,
        0.0,0.0,0.0,1.0,
    );
}