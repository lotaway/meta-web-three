type GPUId = string
type CreateGPUParams = GPUId | HTMLCanvasElement
interface CanvasRes {
    canvas: HTMLCanvasElement
    context: GPUCanvasContext
}

function getCanvas(_targetCanvas: Readonly<CreateGPUParams>): CanvasRes {
    let canvas
    if (typeof _targetCanvas === "string")
        canvas = document.getElementById(_targetCanvas as string) as HTMLCanvasElement
    else
        canvas = _targetCanvas as HTMLCanvasElement
    if (!canvas)
        throw new Error("No canvas found.")
    const context = canvas.getContext("webgpu")
    if (!context)
        throw new Error("No canvas gpu context")
    return {canvas, context}
}

export async function createGPU(_targetCanvas: CreateGPUParams) {
    if (!navigator?.gpu)
        throw new Error("No gpu found.")
    //  获取最合适的GPU
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter)
        throw new Error("No gpu adapter.")
    // adapter.features:array
    // adapter.limits:object
    //  获取设备并请求分配一定资源
    const device = await adapter.requestDevice({
        requiredFeatures: ["texture-compression-bc"],
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
        }
    })
    const {canvas, context} = getCanvas(_targetCanvas)
    //  自行指定
    // const format = "rgba8unorm"
    //  根据显卡支持的格式里选最佳的
    const format = navigator.gpu.getPreferredCanvasFormat()
    const size = [canvas.clientWidth * window.devicePixelRatio, canvas.clientHeight * window.devicePixelRatio]
    //  配置上下文
    context.configure({
        device,
        //  绘制画布的格式，可手动定义，这里是使用自动获取最佳格式
        format,
        size,
        //  设置不透明
        alphaMode: "opaque",
        compositingAlphaMode: "opaque"
    })
    return {
        adapter,
        device,
        format,
        canvas,
        context
    }
}
