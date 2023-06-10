import {memo, createRef, useEffect} from "react"
import {createGPU} from "../../utils/webGPU"
import vertexCodeWGSL from "../../shader/vertex.wgsl?raw"
import fragmentWGSL from "../../shader/frag.wgsl?raw"
import style from "./NewWorld.module.less"

function NewWorld() {
    const canvasRef = createRef<HTMLCanvasElement>()
    //  创建渲染管线
    const initPipeline = (device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline => {
        const vertex = device.createShaderModule({
            code: vertexCodeWGSL
            // code: vWGSL
        })
        const shader = device.createShaderModule({
            code: fragmentWGSL
            // code: fWGSL
        })
        let pipeline = device.createRenderPipeline({
            //  布局，用于着色器中资源的绑定
            layout: "auto",
            //  指定顶点着色器
            vertex: {
                //  着色器文件
                module: vertex,
                //  入口函数
                entryPoint: "vs_main"
            },
            //  指定片元着色器
            fragment: {
                //  着色器文件
                module: shader,
                //  入口函数
                entryPoint: "fs_main",
                targets: [{
                    format
                }]
            },
            //  图元类型
            primitive: {
                //  指定是三角形
                topology: "triangle-list"
            },
            //  多重采样，用于抗锯齿
            multisample: {
                count: 4
            }
        })
        return pipeline
    }
    //  创建纹理
    const initTexture = (device: GPUDevice, format: GPUTextureFormat, canvas: HTMLCanvasElement): GPUTextureView => {
        const texture = device.createTexture({
            size: [canvas.width, canvas.height],
            sampleCount: 4,
            format,
            //  纹理用途：最终渲染
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        })
        return texture.createView()
    }
    const render = (device: GPUDevice, context: GPUCanvasContext, pipeline: GPURenderPipeline, view: GPUTextureView) => {
        //  创建命令编码
        let commandEncoder = device.createCommandEncoder()
        //  开始渲染通道
        let renderPass = commandEncoder.beginRenderPass({
            //  颜色附件数组
            colorAttachments: [{
                //  最终解析的纹理
                view,
                //  中途解析的纹理
                resolveTarget: context.getCurrentTexture().createView(),
                clearValue: {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0
                },
                //  清除操作
                loadOp: "clear",
                //  保存操作
                storeOp: "store"
            }]
        })
        //  设置渲染管线
        renderPass.setPipeline(pipeline)
        //  绘制三角形
        renderPass.draw(3, 1, 0, 0)
        //  结束渲染通道
        renderPass.end();
        //  提交命令
        device.queue.submit([commandEncoder.finish()])
        //  结束命令编码
        requestAnimationFrame(() => render(device, context, pipeline, view))
    }
    useEffect(() => {
        if (canvasRef?.current)
            createGPU(canvasRef.current)
                .then(({adapter, device, format, canvas, context}) => {
                    const pipeline = initPipeline(device, format)
                    const textureView = initTexture(device, format, canvas)
                    render(device, context, pipeline, textureView)
                })
                .catch(err => {

                })
    }, [])
    return (
        <canvas ref={canvasRef} className={style.newWorld}/>
    )
}

export default memo(NewWorld, () => true)
