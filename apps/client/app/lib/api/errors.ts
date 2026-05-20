export class ApiError extends Error {
  public readonly code: string
  public readonly status: number
  public readonly data: unknown
  public readonly isApiError: boolean = true

  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    status: number = 500,
    data: unknown = null
  ) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.data = data

    Object.setPrototypeOf(this, ApiError.prototype)
  }

  static fromResponse(response: Response, data?: unknown): ApiError {
    const message = data && typeof data === 'object' && 'message' in data
      ? (data as any).message
      : response.statusText || '请求失败'

    const code = data && typeof data === 'object' && 'code' in data
      ? (data as any).code
      : String(response.status)

    return new ApiError(message, code, response.status, data)
  }

  static fromError(error: Error): ApiError {
    if (error instanceof ApiError) {
      return error
    }
    return new ApiError(error.message, 'NETWORK_ERROR', 0, null)
  }
}

export const ErrorCodes = {
  SUCCESS: '0000',
  PARAM_ERROR: '1000',
  PARAM_VALIDATION_ERROR: '1001',
  PARAM_MISSING_ERROR: '1002',
  PARAM_TYPE_ERROR: '1003',
  METHOD_NOT_ALLOWED: '1004',
  NOT_FOUND: '1005',
  FILE_TOO_LARGE: '1006',

  USER_NOT_FOUND: '2001',
  USER_PASSWORD_ERROR: '2002',
  USER_TOKEN_EXPIRED: '2003',
  USER_TOKEN_INVALID: '2004',

  PRODUCT_NOT_FOUND: '3001',
  PRODUCT_OFF_SHELF: '3002',

  ORDER_NOT_FOUND: '4001',
  ORDER_CREATE_FAILED: '4002',
  ORDER_CANCEL_FAILED: '4003',
  ORDER_STATUS_INVALID: '4004',

  PAYMENT_INSUFFICIENT_BALANCE: '5001',
  PAYMENT_EXCHANGE_FAILED: '5002',

  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
} as const

export type ErrorCode = typeof ErrorCodes[keyof typeof ErrorCodes]

export function isApiError(error: unknown): error is ApiError {
  return typeof error === 'object' && error !== null && 'isApiError' in error
}

export function getErrorMessage(error: unknown): string {
  if (isApiError(error)) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return '未知错误'
}