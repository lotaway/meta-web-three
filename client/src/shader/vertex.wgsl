//  wgsl着色器
//  顶点着色器主函数
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
    //  内部设置三角形顶点坐标
    var pos = array<vec2<f32>, 3>(
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}