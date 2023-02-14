import {memo, createRef, useEffect} from "react"
import {createGPU} from "../utils/webGPU"
// import vertexCode from "../shader/vertex.wgsl?raw"

function NewWorld() {
    const canvasRef = createRef<HTMLCanvasElement>()
    const initPipeline = (device: GPUDevice) => {
        const vertex = device.createShaderModule({
            // code: vertexCode
            code: ""
        })
        const s = device.createShaderModule({
            code: ""
        })
    }
    useEffect(() => {
        if (canvasRef?.current)
            createGPU(canvasRef.current).then(({adapter, device}) => {
                initPipeline(device)
            })
    }, [])
    return (
        <canvas ref={canvasRef}/>
    )
}

export default memo(NewWorld, () => true)
