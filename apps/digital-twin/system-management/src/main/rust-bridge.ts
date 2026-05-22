import * as NativeTypes from '../../native/index'

let native: typeof NativeTypes | null = null

try {
    native = require('../../native/index.node')
} catch (error) {
    console.error('Failed to load native module:', error)
    native = {
        init: () => console.log("Native module not loaded"),
        getSystemInfo: () => null as any,
        getCpuInfo: () => [],
        getMemoryInfo: () => null as any,
        getNetworkInfo: () => [],
        getDiskInfo: () => [],
        getProcessList: () => [],
        listCameras: () => [],
        startCamera: () => {},
        stopCamera: () => false,
        getLatestFrame: () => null,
    } as any
}

export function initializeSupport() {
    if (!native) return
    try {
        native.init()
    } catch (e) {
        console.error("Error initializing native support:", e)
    }
}

export function getSystemMonitor() {
    if (!native) return null
    return {
        getSystemInfo: () => native!.getSystemInfo(),
        getCpuInfo: () => native!.getCpuInfo(),
        getMemoryInfo: () => native!.getMemoryInfo(),
        getNetworkInfo: () => native!.getNetworkInfo(),
        getDiskInfo: () => native!.getDiskInfo(),
        getProcessList: () => native!.getProcessList(),
    }
}

export function getCameraManager() {
    if (!native) return null
    return {
        listCameras: () => native!.listCameras(),
        startCamera: (index: number, width: number, height: number, fps: number) =>
            native!.startCamera(index, width, height, fps),
        stopCamera: () => native!.stopCamera(),
        getLatestFrame: () => native!.getLatestFrame(),
    }
}

export function getAudioManager() {
    if (!native) return null
    return {
        listAudioDevices: () => native!.listAudioDevices(),
        startAudioCapture: (deviceIndex: number, sampleRate: number = 0) =>
            native!.startAudioCapture(deviceIndex, sampleRate),
        stopAudioCapture: () => native!.stopAudioCapture(),
        getAudioData: () => native!.getAudioData(),
    }
}

export default native
