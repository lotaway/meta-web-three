import { ipcMain, desktopCapturer } from "electron"
import { MediaService } from "./nestjs/services/media.service"
import { LLMService } from "./nestjs/services/llm.service"
import { IPC_CHANNELS } from "./constants"
import { INestApplicationContext } from "@nestjs/common"
import { getAudioManager } from "./rust-bridge"

export class IpcRegistry {
    constructor(
        private getNestApp: () => INestApplicationContext | null,
        private getAppLifecycle: () => any
    ) { }

    register() {
        ipcMain.handle(IPC_CHANNELS.READ_FILE_IN_DIRECTORY, (event, filePath) => {
            const nestApp = this.getNestApp()
            if (!nestApp) return null
            const mediaService = nestApp.get(MediaService)
            return mediaService.readFileInDirectory(filePath)
        })

        ipcMain.handle(IPC_CHANNELS.MERGE_VIDEO, async (event, videos, outputPath) => {
            const nestApp = this.getNestApp()
            if (!nestApp) return null
            const mediaService = nestApp.get(MediaService)
            return await mediaService.mergeVideo(videos, outputPath)
        })

        ipcMain.handle(IPC_CHANNELS.LLM_COMPLETION, async (event, prompt) => {
            const nestApp = this.getNestApp()
            if (!nestApp) return null
            const llmService = nestApp.get(LLMService)
            return await llmService.completion(prompt)
        })

        ipcMain.handle(IPC_CHANNELS.SUBTITLES_OPEN, () => {
            this.getAppLifecycle().getSubtitlesWindow()?.open()
        })

        ipcMain.handle(IPC_CHANNELS.SUBTITLES_CLOSE, () => {
            this.getAppLifecycle().getSubtitlesWindow()?.close()
        })

        ipcMain.handle(IPC_CHANNELS.SUBTITLES_UPDATE, (event, text) => {
            this.getAppLifecycle().getSubtitlesWindow()?.updateText(text)
        })

        // Also handle SUBTITLES_TEXT as it's used by AudioContext
        ipcMain.on(IPC_CHANNELS.SUBTITLES_TEXT, (event, text) => {
            this.getAppLifecycle().getSubtitlesWindow()?.updateText(text)
        })

        ipcMain.handle(IPC_CHANNELS.GET_AUDIO_SOURCES, async () => {
            const sources = await desktopCapturer.getSources({
                types: ['window', 'screen'],
                thumbnailSize: { width: 0, height: 0 }
            })
            return sources.map(source => ({
                id: source.id,
                name: source.name
            }))
        })

        // Rust native audio capture
        const audio = getAudioManager()

        ipcMain.handle(IPC_CHANNELS.AUDIO_LIST_DEVICES, () => {
            return audio?.listAudioDevices() ?? []
        })

        ipcMain.handle(IPC_CHANNELS.AUDIO_START_CAPTURE, (_event, deviceIndex: number, sampleRate: number = 0) => {
            try {
                audio?.startAudioCapture(deviceIndex, sampleRate)
                return true
            } catch {
                return false
            }
        })

        ipcMain.handle(IPC_CHANNELS.AUDIO_STOP_CAPTURE, () => {
            return audio?.stopAudioCapture() ?? false
        })

        ipcMain.handle(IPC_CHANNELS.AUDIO_GET_DATA, () => {
            return audio?.getAudioData() ?? null
        })
    }
}
