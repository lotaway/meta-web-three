import * as express from 'express'
import { HttpStatus } from '@nestjs/common'
import { LLMService } from '../services/llm.service'

export const DEFAULT_MODEL_INFO = {
    "name": "local",
    "version": "1.0.0",
    "object": "model",
    "owned_by": "lotaway",
    "api_version": "v1",
}

export class LLMController {
    constructor(
        private readonly llmService?: LLMService
    ) { }

    getShow(req: express.Request, res: express.Response) {
        res.json({ ok: true })
    }

    getTags(req: express.Request, res: express.Response) {
        const models = [{ ...DEFAULT_MODEL_INFO }]
        const localProvider = process.env.LOCAL_LLM_PROVIDER
        if (localProvider) {
            models.push({ ...DEFAULT_MODEL_INFO, name: 'local' })
        }
        res.json(models)
    }

    async chatCompletions(req: express.Request, res: express.Response): Promise<void> {
        try {
            const payload = req.body
            let prompt = payload.prompt
            if (!prompt && payload.messages && Array.isArray(payload.messages)) {
                const lastMsg = payload.messages[payload.messages.length - 1]
                prompt = lastMsg.content
            }

            if (!prompt) {
                res.status(HttpStatus.BAD_REQUEST).json({ error: 'Missing prompt or messages' })
                return
            }

            if (payload.stream) {
                await this.handleStream(prompt, payload, res)
            } else if (this.llmService) {
                await this.handleNonStream(prompt, res)
            } else {
                await this.handleStream(prompt, payload, res)
            }
        } catch (error: any) {
            console.error('Error in chat completions:', error)
            res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: error.message })
        }
    }

    private async handleNonStream(prompt: string, res: express.Response): Promise<void> {
        try {
            const response = await this.llmService!.completion(prompt)
            const content = typeof response === 'string' ? response : response.content || JSON.stringify(response)
            res.json({
                id: Date.now().toString(),
                object: 'chat.completion',
                created: Math.floor(Date.now() / 1000),
                model: 'local',
                choices: [{
                    index: 0,
                    message: {
                        role: 'assistant',
                        content
                    },
                    finish_reason: 'stop'
                }]
            })
        } catch (err: any) {
            console.error('[LLM] Non-stream completion error:', err)
            res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: err.message })
        }
    }

    private async handleStream(prompt: string, payload: any, res: express.Response): Promise<void> {
        const localProvider = process.env.LOCAL_LLM_PROVIDER
        if (!localProvider) {
            res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: 'LOCAL_LLM_PROVIDER not configured' })
            return
        }

        try {
            const response = await fetch(`${localProvider}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })

            if (!response.ok) {
                const errorText = await response.text()
                res.status(response.status).send(errorText)
                return
            }

            res.writeHead(response.status, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            })

            const reader = response.body?.getReader()
            if (reader) {
                while (true) {
                    const { done, value } = await reader.read()
                    if (done) break
                    res.write(value)
                }
            }
            res.end()
        } catch (err: any) {
            console.error('[Local LLM] Streaming error:', err)
            if (!res.headersSent) {
                res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: 'Failed to stream from local provider' })
            } else {
                res.end()
            }
        }
    }
}
