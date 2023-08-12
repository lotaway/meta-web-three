import {mat4, vec2, vec3, vec4} from "gl-matrix"

export namespace ThreeGPU {

    type GPUId = string

    export type CreateGPUParams = GPUId | HTMLCanvasElement

    interface CanvasRes<T extends RenderingContext | GPUCanvasContext = RenderingContext | GPUCanvasContext> {
        canvas: HTMLCanvasElement
        context: T
    }

    export type TypedArray =
        Int16ArrayConstructor
        | Int32ArrayConstructor
        | Float32ArrayConstructor
        | Float64ArrayConstructor

    export interface Vertex<DataType = TypedArray> {
        data: DataType,
        size: number
    }

    export function getCanvas<T extends CanvasRes["context"]>(_targetCanvas: Readonly<CreateGPUParams>, contextId: string = "webgpu"): CanvasRes<T> {
        let canvas
        if (typeof _targetCanvas === "string")
            canvas = document.getElementById(_targetCanvas) as HTMLCanvasElement
        else
            canvas = _targetCanvas as HTMLCanvasElement
        if (!canvas)
            throw new Error("No canvas found.")
        const context = canvas.getContext(contextId) as T
        if (!context)
            throw new Error(`No canvas ${contextId}} context`)
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
        const {canvas, context} = getCanvas<GPUCanvasContext>(_targetCanvas)
        //  自行指定
        // const format = "rgba8unorm"
        //  根据显卡支持的格式里选最佳的
        const format = navigator.gpu.getPreferredCanvasFormat()
        const size = [canvas.clientWidth * window.devicePixelRatio, canvas.clientHeight * window.devicePixelRatio]
        const configure: GPUCanvasConfiguration = {
            device,
            //  绘制画布的格式，可手动定义，这里是使用自动获取最佳格式
            format,
            // size,
            //  设置不透明
            alphaMode: "opaque",
            // compositingAlphaMode: "opaque"
        }
        //  配置上下文
        context.configure(configure)
        return {
            adapter,
            device,
            format,
            canvas,
            context
        }
    }
}

export interface ToArray<T> {
    toArray(): Array<T>
}

export type EasyCreate<T> = {
    create(value?: any): T
}

export interface Vec {
    x: number
}

export interface Vec2 extends Vec {
    y: number
}

export interface Vec3 extends Vec2 {
    z: number
    // get xy(): Vec2
}

export interface Vec4 extends Vec3 {
    w: number
    // get xyz(): Vec3
}

export interface WebGPUOptions {
    shaderLocation?: number
    adapter: GPUAdapter
    device: GPUDevice
    format: GPUTextureFormat
}

export class WebGPUBuilder {
    private _shaderLocation: number

    constructor(options: WebGPUOptions) {
        this._shaderLocation = options.shaderLocation ?? 0
    }

    get shaderLocation() {
        return {
            next: () => {
                this._shaderLocation++
                return this
            },
            value: this._shaderLocation
        }
    }
}

export class GPUImpl implements GPU {
    __brand: "GPU"
    wgslLanguageFeatures: WGSLLanguageFeatures
    _requestAdapter: (options?: GPURequestAdapterOptions) => Promise<GPUAdapter | null>
    _getPreferredCanvasFormat: () => GPUTextureFormat

    constructor(gpu: GPU) {
        this.__brand = gpu.__brand
        this.wgslLanguageFeatures = gpu.wgslLanguageFeatures
        this._requestAdapter = gpu.requestAdapter
        this._getPreferredCanvasFormat = gpu.getPreferredCanvasFormat
    }

    static fromBrowser() {
        return new GPUImpl(window.navigator.gpu)
    }

    getPreferredCanvasFormat(): GPUTextureFormat {
        return this._getPreferredCanvasFormat()
    }

    async requestAdapter(options?: GPURequestAdapterOptions) {
        return await this._requestAdapter(options)
    }
}

export class GPUBufferDescriptorBuilder implements GPUBufferDescriptor {
    constructor(readonly size: number, public usage: number, readonly mappedAtCreation?: boolean | undefined, readonly label?: string | undefined) {
    }

    setUsage(usage: keyof GPUBufferUsage) {
        this.usage = Number(usage)
        return this
    }

    from(descriptor: GPUBufferDescriptor) {
        return new GPUBufferDescriptorBuilder(descriptor.size, descriptor.usage, descriptor.mappedAtCreation, descriptor.label)
    }
}

export class GPUColorIterator implements Iterator<number, null> {
    readonly list: ArrayLike<number>
    index: number = -1

    constructor(readonly color: GPUColorDict) {
        this.list = [color.r, color.g, color.b, color.a]
        if (this.list.length)
            this.index = 0
    }

    get isDone() {
        return this.index == this.list.length - 1
    }

    next(...args: [] | [undefined]): IteratorResult<number, any> {
        const done = this.isDone
        if (done) {
            return this.return(null)
        }
        return {
            done,
            value: this.list[this.index++]
        }
    }

    return(value: null): IteratorResult<number, null> {
        return {
            done: true,
            value
        }
    }

}

export class GPUColorDircWrapper implements GPUColorDict, Iterable<number> {
    constructor(
        public readonly r: number,
        public readonly g: number,
        public readonly b: number,
        public readonly a: number = 1.0,
    ) {

    }

    static create() {
        return this.createWhite()
    }

    static createBlack() {
        return new GPUColorDircWrapper(0.0, 0.0, 0.0)
    }

    static createWhite() {
        return new GPUColorDircWrapper(1.0, 1.0, 1.0)
    }

    [Symbol.iterator](): Iterator<number> {
        return new GPUColorIterator(this);
    }

    toList = () => [this.r, this.g, this.b, this.a]
}

export class VecIterator implements Iterator<number> {

    readonly vecs: Array<number>
    protected index = 0

    constructor(items: Array<number>) {
        this.vecs = items
    }

    get isEnd() {
        return this.vecs.length - 1 <= this.index
    }

    next(...args: [] | [undefined]): IteratorResult<number, any> {
        if (this.isEnd)
            return this.return(null)
        return {
            done: false,
            value: this.vecs[this.index++]
        }
    }

    return(value?: any): IteratorResult<number, any> {
        return {
            done: true,
            value: null
        };
    }

}

export class VecBuilder {

}

export class Vec2Builder extends VecBuilder implements Vec2, ToArray<number>, Iterable<number> {
    x: number
    y: number

    constructor(options: {
        x: number,
        y: number
    }) {
        super()
        this.x = options.x
        this.y = options.y
    }

    static create() {
        return new Vec2Builder({x: 0, y: 0})
    }

    [Symbol.iterator](): Iterator<number> {
        return new VecIterator(this.toArray());
    }

    fromArray(arr: Iterable<number>) {
        const selfArr = this.toArray()
        for (const num of arr) {
            if (selfArr.length == 0)
                break
            selfArr[0] = num
            selfArr.shift()
        }
    }

    toArray(): Array<number> {
        return [this.x, this.y]
    }

    toVec(): vec2 {
        return this.toArray() as vec2
    }
}

export class Vec3Builder extends VecBuilder implements Vec3, ToArray<number>, Iterable<number> {
    x: number
    y: number
    z: number

    constructor(options: {
        x: number,
        y: number,
        z: number
    }) {
        super()
        this.x = options.x
        this.y = options.y
        this.z = options.z
    }

    // get xy => this.proxy.xy

    static create() {
        return new Vec3Builder({x: 0, y: 0, z: 0})
    }

    get xy(): vec2 {
        return [this.x, this.y]
    }

    [Symbol.iterator](): Iterator<number> {
        return new VecIterator(this.toArray());
    }

    fromArray(arr: Iterable<number>) {
        const selfArr = this.toArray()
        for (const num of arr) {
            if (selfArr.length == 0)
                break
            selfArr[0] = num
            selfArr.shift()
        }
    }

    toArray(): Array<number> {
        return [this.x, this.y, this.z]
    }

    toVec(): vec3 {
        return this.toArray() as vec3
    }
}

export class Vec4Builder extends VecBuilder implements Vec4, ToArray<number>, Iterable<number> {
    x: number
    y: number
    z: number
    w: number

    constructor(options: {
        x: number,
        y: number,
        z: number,
        w: number
    }) {
        super()
        this.x = options.x
        this.y = options.y
        this.z = options.z
        this.w = options.w
    }

    static create() {
        return new Vec4Builder({x: 0, y: 0, z: 0, w: 0})
    }

    get xyz() {
        return [this.x, this.y, this.z]
    }

    [Symbol.iterator](): Iterator<number> {
        return new VecIterator(this.toArray());
    }

    fromArray(arr: Iterable<number>) {
        const selfArr = this.toArray()
        for (const num of arr) {
            if (selfArr.length == 0)
                break
            selfArr[0] = num
            selfArr.shift()
        }
    }

    toArray(): Array<number> {
        return [this.x, this.y, this.z]
    }

    toVec(): vec3 {
        return this.toArray() as vec3
    }
}

export class VertexBuilder<DataType = ThreeGPU.TypedArray> implements ThreeGPU.Vertex<DataType> {

    constructor(
        readonly data: DataType,
        readonly size: number
    ) {

    }

    static fromVex<DataType = Float32Array>(vexs: Iterable<Vec2 | Vec3 | Vec4>, _TypedArray: ThreeGPU.TypedArray = Float32Array): ThreeGPU.Vertex<DataType> {
        let arr: Array<number> = []
        let size = 0
        for (let v of vexs) {
            arr.push(...Object.values(v))
            size++
        }
        return new VertexBuilder<DataType>(
            _TypedArray.from(arr) as DataType,
            size,
        )
    }

}

export class GPUShaderModuleDescriptorBuilder implements GPUShaderModuleDescriptor {
    constructor(
        readonly code: string,
        readonly sourceMap?: object | undefined,
        readonly hints?: Record<string, GPUShaderModuleCompilationHint> | undefined,
        readonly label?: string | undefined,
    ) {

    }
}

export class GPUShaderModuleBuilder {

    protected device?: GPUDevice

    constructor(protected readonly descriptor: GPUShaderModuleDescriptor) {

    }

    static fromWGSL(code: string): GPUShaderModuleBuilder {
        return new GPUShaderModuleBuilder({
            code
        })
    }

    static buildModule(device: GPUDevice, code: GPUShaderModuleDescriptor["code"]): GPUShaderModule {
        return GPUShaderModuleBuilder.fromWGSL(code).setDevice(device).build()
    }

    setDevice(device: GPUDevice) {
        this.device = device
        return this
    }

    getDescriptor(): GPUShaderModuleDescriptor {
        return this.descriptor
    }

    build(): GPUShaderModule {
        if (!this.device)
            throw new Error("Device is not initial.")
        return this.device.createShaderModule(this.getDescriptor())
    }
}

function testMatrix() {

    //  创建平移矩阵
    const m4 = mat4.create()
    const m4t = mat4.create()
    //  指定x轴平移2
    mat4.translate(m4, m4t, [2, 0, 0])
    //  创建缩放矩阵
    const m4s = mat4.create()
    //  指定x轴放大10倍
    mat4.scale(m4, m4s, [10, 1, 1])
    //  模型矩阵用于执行前面的先平移后缩放的矩阵乘法
    const model = mat4.create()
    //  以下由于矩阵乘法的特性实际先后顺序是倒过来的
    mat4.multiply(model, model, m4s)    //  后缩放
    mat4.multiply(model, model, m4t)    //  先平移

    // 以上代码的简化写法
    const model2 = mat4.create()
    mat4.scale(model2, model2, [10, 1, 0])
    mat4.translate(model2, model2, [2, 0, 0])

    //  创建两个顶点
    const p1 = vec3.fromValues(2, 0, 0)
    const p2 = vec3.create()
    //  对p1使用model/model2矩阵变换，并将结果存储在p2，相当于对顶点p1执行先平移后缩放的动作
    vec3.transformMat4(p2, p1, model2)
}