type CreateGPUParams = string | HTMLCanvasElement

function getCanvas(_targetCanvas: Readonly<CreateGPUParams>) {
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
    if (!navigator || !navigator.gpu)
        throw new Error("No gpu found.")
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter)
        throw new Error("No gpu adapter.")
    // adapter.features:array
    // adapter.limits:object
    const device = await adapter.requestDevice({
        requiredFeatures: ["texture-compression-bc"],
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
        }
    })
    const {canvas, context} = getCanvas(_targetCanvas)
    const format = context.getPreferredFormat(adapter)
    const size = [canvas.clientWidth * window.devicePixelRatio, canvas.clientHeight * window.devicePixelRatio]
    context.configure({
        device,
        format,
        size,
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
