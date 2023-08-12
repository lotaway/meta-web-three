//  片元着色器
@group(1) @binding(0) var<uniform> colors: vec3<f32>;

@fragment
fn fs_main(@location(0) f_pos: vec3<f32>, @location(1) v_colors: vec3<f32>) -> @location(0) vec4<f32> {
//    return vec4(f_pos.x, colors.y, 1 - f_pos.y, 1.0);
    return vec4(v_colors.x, v_colors.y, f_pos.y - colors.z, 1.0);
}

@fragment
fn fs_color(@builtin(position) fragCode: vec4<f32>) -> @location(0) vec4<f32> {
//colors.x
    return vec4((fragCode.x - 50.0) / 150.0, colors.y, colors.z, 1.0);
}

/*
关于WGSL的内置方法、函数和类，以下是一些常用的内置函数和类：
- 数学函数：abs、sin、cos、tan、sqrt、pow等。
- 向量和矩阵操作：dot、cross、normalize、length、transpose等。
- 采样器和纹理操作：textureSample、textureSampleLevel、textureLoad、textureStore等。
- 条件和循环：if、else、for、while等。
- 数据类型：bool、int、uint、float、vec2、vec3、vec4、mat2、mat3、mat4等。
*/
fn random() -> f32 {
  var seed: u32 = 0; // 设置一个种子值
  seed = (seed * 1664525u + 1013904223u) & 0xFFFFFFFFu; // 更新种子值
  return f32(seed) / 4294967296.0; // 将种子值转换为浮点数并返回
}