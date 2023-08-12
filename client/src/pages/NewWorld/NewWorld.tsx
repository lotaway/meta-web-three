import {memo, createRef, useEffect} from "react"
import {Buffer} from "buffer"
import {GPUShaderModuleBuilder, ThreeGPU, VertexBuilder, WebGPUBuilder} from "../../system/webGPU"
import positionVertWGSL from "../../shader/position.vert.wgsl?raw"
import triangleVertexWGSL from "../../shader/triangle.vert.wgsl?raw"
import pinkFragmentWGSL from "../../shader/pink.frag.wgsl?raw"
import colorFragmentWGSL from "../../shader/color.frag.wgsl?raw"
import style from "./NewWorld.module.less"
import {mat4, vec3} from "gl-matrix"

//  创建渲染管线
async function initPipeline(device: GPUDevice, format: GPUTextureFormat) {
    const vertexModule = GPUShaderModuleBuilder
        .fromWGSL(triangleVertexWGSL)
        .setDevice(device)
        .build()
    /*const vertex = {
        data: Float32Array.from([
            // x, y, z三个一组，并无硬性规定，只要方便下方buffers使用即可
            0.0, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.0, 0.5, 0.0,
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.5, 0.0,
        ]),
        size: 6
    }*/
    const vertex = VertexBuilder.fromVex([
        {x: 0.0, y: 0.0, z: 0.0},
        {x: 0.5, y: 0.0, z: 0.0},
        {x: 0.0, y: 0.5, z: 0.0},
        {x: 0.0, y: 0.5, z: 0.0},
        {x: 0.5, y: 0.0, z: 0.0},
        {x: 0.5, y: 0.5, z: 0.0}
    ])
    const vertexBuffer = device.createBuffer({
        size: vertex.data.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        // mappedAtCreation: true,
    })
    // await vertexBuffer.mapAsync(GPUMapMode.WRITE)
    // const vertexBMap = new Float32Array(vertexBuffer.getMappedRange())
    // vertexBMap.set(vertex.data)
    // vertexBuffer.unmap()
    device.queue.writeBuffer(vertexBuffer, 0, vertex.data)

    const vertexColors = Float32Array.from([
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        0.2, 1.0, 0.0,
        1.0, 1.0, 0.0,
        0.8, 0.0, 1.0,
        0.5, 1.0, 1.0,
    ])
    const vertexColorsBuffer = device.createBuffer({
        size: vertexColors.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(vertexColorsBuffer, 0, vertexColors)

    const fragmentModule = GPUShaderModuleBuilder.fromWGSL(colorFragmentWGSL).setDevice(device).build()
    const xy: GPUVertexBufferLayout = {
        arrayStride: 3 * 4,
        attributes: [{
            shaderLocation: 0, //   对应顶点缓冲区存储的位置值
            offset: 0,  //  每组要跳过的长度（字节）
            format: "float32x2",
        }]
    }
    const z: GPUVertexBufferLayout = {
        arrayStride: 3 * 4,
        attributes: [{
            shaderLocation: 0,
            offset: 2 * 4,
            format: "float32",
        }]
    }
    // const buffers: Iterable<GPUVertexBufferLayout> = [xy, z]

    const xyz: GPUVertexAttribute = {
        shaderLocation: 0,
        offset: 0,
        format: "float32x3",
    }
    const vertexBufferLayout: GPUVertexBufferLayout = {
        arrayStride: 3 * 4, //  表示一组为3个值乘以（float32）4个字节的长度
        attributes: [xyz],
    }
    const colorsBufferLayout: GPUVertexBufferLayout = {
        arrayStride: 3 * 4,
        attributes: [{
            shaderLocation: 1,
            offset: 0,
            format: "float32x3",
        }]
    }
    const pDescriptor: GPURenderPipelineDescriptor = {
        //  布局，用于着色器中资源的绑定
        layout: "auto",
        //  指定顶点着色器
        vertex: {
            //  着色器文件
            module: vertexModule,
            //  入口函数
            entryPoint: "vt_main",
            buffers: [
                vertexBufferLayout,
                colorsBufferLayout,
            ],
        },
        //  指定片元着色器
        fragment: {
            //  着色器文件
            module: fragmentModule,
            //  入口函数
            entryPoint: "fs_main",
            targets: [{
                format
            }]
        },
        //  图元类型
        primitive: {
            //  指定是三角形
            topology: 'triangle-list'
        },
        //  多重采样，用于抗锯齿
        /*multisample: {
            count: 4
        }*/
    }
    const pipeline = device.createRenderPipeline(pDescriptor)
    return {vert: {vertex, vertexBuffer, vertexColorsBuffer}, pipeline}
}

//  Uniform负责实现变形矩阵，bindGroup可以实现对顶点着色器和片元着色器传递值
function initUniform(device: GPUDevice, pipeline: GPURenderPipeline) {
    const index = new Proxy({
        value: -1,
        next: -1,
    }, {
        get(target: {
            value: number;
            next: number;
            [key: string | symbol]: any
        }, p: string | symbol, receiver: any): any {
            if (p == "next") {
                ++target.value
                return target.value
            }
            return target[p]
        }
    })
    /*const transformArray = Float32Array.from([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])*/
    const m4s = mat4.create() as Float32Array
    mat4.scale(m4s, m4s, [0.5, 1.0, 1.0])
    const bDescriptor: GPUBufferDescriptor = {
        size: m4s.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }
    const m4sBuffer = device.createBuffer(bDescriptor)
    device.queue.writeBuffer(m4sBuffer, 0, m4s)

    const m4t = mat4.create() as Float32Array
    mat4.translate(m4t, m4t, [0.2, 0.2, 0])
    const m4tBuffer = device.createBuffer({
        size: m4t.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })
    device.queue.writeBuffer(m4tBuffer, 0, m4t)

    const vertexGroupIndex = index.next
    const vertexEntries: Iterable<GPUBindGroupEntry> = [{
        binding: 0,
        resource: {
            buffer: m4sBuffer
        }
    }, {
        binding: 1,
        resource: {
            buffer: m4tBuffer
        }
    }]
    const vDescriptor: GPUBindGroupDescriptor = {
        layout: pipeline.getBindGroupLayout(vertexGroupIndex),
        entries: vertexEntries
    }
    const vertexBindGroup = device.createBindGroup(vDescriptor)

    const colorsArray = Float32Array.from([Math.random(), Math.random(), Math.random()])
    const cDescriptor: GPUBufferDescriptor = {
        size: colorsArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }
    const colorsBuffer = device.createBuffer(cDescriptor)
    device.queue.writeBuffer(colorsBuffer, 0, colorsArray)

    const fragmentGroupIndex = index.next
    const fragmentEntries: Iterable<GPUBindGroupEntry> = [{
        binding: 0,
        resource: {
            buffer: colorsBuffer
        }
    }]
    const fDescriptor: GPUBindGroupDescriptor = {
        layout: pipeline.getBindGroupLayout(fragmentGroupIndex),
        entries: fragmentEntries
    }
    const fragmentBindGroup = device.createBindGroup(fDescriptor)

    return {
        uniformGroups: [{
            slot: vertexGroupIndex,
            bindGroup: vertexBindGroup
        }, {
            slot: fragmentGroupIndex,
            bindGroup: fragmentBindGroup
        }]
    }
}

//  创建纹理
function initTexture(device: GPUDevice, format: GPUTextureFormat, canvas: HTMLCanvasElement): GPUTextureView {
    const descriptor: GPUTextureDescriptor = {
        size: [canvas.width, canvas.height],
        sampleCount: 4,
        format,
        //  纹理用途：最终渲染
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    }
    const texture = device.createTexture(descriptor)
    return texture.createView()
}

function render(device: GPUDevice, context: GPUCanvasContext, pipeline: GPURenderPipeline, bindGroups: Array<{
    slot: number,
    bindGroup: GPUBindGroup,
}>, textureView: GPUTextureView | null, vert: {
    vertex: ThreeGPU.Vertex<Float32Array>,
    vertexBuffer: GPUBuffer,
    vertexColorsBuffer: GPUBuffer,
}) {
    function loop() {
        //  创建命令编码，用于执行各种指令，独立于JS之外的线程
        const commandEncoder = device.createCommandEncoder()

        const isPlus = Math.random() * 2 > 1 ? 1 : -1
        baseBg.r = baseBg.r + Math.random() * changeRedis * isPlus
        baseBg.r = baseBg.r > 0.5 ? 0.47 : baseBg.r
        baseBg.r = baseBg.r < 0.4 ? 0.43 : baseBg.r
        baseBg.g = baseBg.g + Math.random() * changeRedis * isPlus
        baseBg.g = baseBg.g > 0.5 ? 0.47 : baseBg.g
        baseBg.g = baseBg.g < 0.4 ? 0.43 : baseBg.g
        baseBg.b = baseBg.b + Math.random() * changeRedis * isPlus
        baseBg.b = baseBg.b > 0.5 ? 0.47 : baseBg.b
        baseBg.b = baseBg.b < 0.4 ? 0.43 : baseBg.b

        const rDescriptor: GPURenderPassDescriptor = {
            //  颜色附件数组
            colorAttachments: [{
                //  最终解析的纹理
                view: textureView ?? context.getCurrentTexture().createView(),
                //  中途解析的纹理
                // resolveTarget: context.getCurrentTexture().createView(),
                clearValue: baseBg,
                //  是否保留原有内容，这里表示清空
                loadOp: "clear",
                //  保存操作
                storeOp: "store",
            }]
        }
        //  开启一个渲染通道（类似图层）
        const renderPass = commandEncoder.beginRenderPass(rDescriptor)
        //  设置渲染管线
        renderPass.setPipeline(pipeline)
        //  插入顶点值，slot对应buffers所在的index
        renderPass.setVertexBuffer(0, vert.vertexBuffer)
        renderPass.setVertexBuffer(1, vert.vertexColorsBuffer)
        //  设置uniform
        for (const item of bindGroups) {
            renderPass.setBindGroup(item.slot, item.bindGroup)
        }

        angle += 0.05
        const m4r = mat4.create() as Float32Array
        mat4.rotateZ(m4r, m4r, angle)
        const m4rBuffer = device.createBuffer({
            size: m4r.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })
        device.queue.writeBuffer(m4rBuffer, 0, m4r)
        const m4rBindGroupIndex = 2
        const m4rBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(m4rBindGroupIndex),
            entries: [{
                binding: 0,
                resource: {
                    buffer: m4rBuffer
                }
            }]
        })
        renderPass.setBindGroup(m4rBindGroupIndex, m4rBindGroup)

        //  绘制三角形，根据wgsl定义好的顶点数
        // renderPass.draw(6, 1, 0, 0)
        //  绘制图形，根据js里自定义的vertex
        renderPass.draw(vert.vertex.size)
        //  结束渲染通道
        renderPass.end()
        //  将定义好的一连串命令生成GPU缓冲区对象
        const cmdBuffer = commandEncoder.finish()
        //  提交命令
        device.queue.submit([cmdBuffer])
        requestAnimationFrame(() => loop())
    }

    const changeRedis = 0.001
    const baseBg: GPUColorDict = {
        r: 0.0,
        g: 0.0,
        // b: 0.5,
        // b: Math.random() * 0.1,
        b: 0.0,
        a: 1.0,
    }
    let angle = 0
    loop()
}

class Web3d {
    static async getInstance(element: ThreeGPU.CreateGPUParams) {
        try {
            const {adapter, device, format, canvas, context} = await ThreeGPU.createGPU(element)
            const webGpu = new WebGPUBuilder({
                adapter,
                device,
                format,
            })
            const {pipeline, vert} = await initPipeline(device, format)
            const {uniformGroups} = initUniform(device, pipeline)
            const textureView = initTexture(device, format, canvas)
            render(device, context, pipeline, uniformGroups, null, vert)
        } catch (err) {
            console.log(`creating gpu error: ${err}`)
        }
    }
}

function NewWorld() {
    const canvasRef = createRef<HTMLCanvasElement>()
    useEffect(() => {
        function dispose() {
            if (timer)
                clearTimeout(timer)
        }

        if (!canvasRef?.current)
            return
        let timer = setTimeout((canvas: HTMLCanvasElement) => {
            Web3d.getInstance(canvas)
        }, 300, canvasRef.current)
        return dispose
    }, [])
    return (
        <canvas ref={canvasRef} width="200" height="200" className={style.newWorld}/>
    )
}

export default memo(NewWorld, () => true)
