import * as express from 'express'
import { HttpStatus } from '@nestjs/common'

export class AuthController {
    async handleToken(req: express.Request, res: express.Response): Promise<void> {
        try {
            const { token } = req.body
            if (!token) {
                res.status(HttpStatus.BAD_REQUEST).json({ error: 'Missing token' })
                return
            }
            console.log('[API] Received auth token')
            res.json({ status: 'success' })
        } catch (error: any) {
            res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: error.message })
        }
    }
}
