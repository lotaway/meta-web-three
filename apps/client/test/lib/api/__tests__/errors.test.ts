import { ApiError, ErrorCodes, isApiError, getErrorMessage } from '../../../app/lib/api/errors'

describe('api/errors', () => {
  describe('ApiError', () => {
    it('should create error with default values', () => {
      const error = new ApiError('Test error')
      expect(error.message).toBe('Test error')
      expect(error.code).toBe('UNKNOWN_ERROR')
      expect(error.status).toBe(500)
      expect(error.data).toBeNull()
      expect(error.isApiError).toBe(true)
    })

    it('should create error with custom values', () => {
      const error = new ApiError('Test error', 'TEST_CODE', 400, { details: 'test' })
      expect(error.message).toBe('Test error')
      expect(error.code).toBe('TEST_CODE')
      expect(error.status).toBe(400)
      expect(error.data).toEqual({ details: 'test' })
    })

    it('should preserve prototype chain', () => {
      const error = new ApiError('Test error')
      expect(error).toBeInstanceOf(Error)
      expect(error).toBeInstanceOf(ApiError)
    })
  })

  describe('ApiError.fromResponse', () => {
    it('should create error from successful response with error code', () => {
      const response = new Response(
        JSON.stringify({ code: '4001', message: '订单不存在' }),
        { status: 400 }
      )

      const error = ApiError.fromResponse(response)

      expect(error.status).toBe(400)
      expect(error.code).toBe('4001')
      expect(error.message).toBe('订单不存在')
    })

    it('should create error from response with fallback message', () => {
      const response = new Response(null, { status: 404, statusText: 'Not Found' })

      const error = ApiError.fromResponse(response)

      expect(error.status).toBe(404)
      expect(error.message).toBe('Not Found')
    })

    it('should use status as code when code is not in response', () => {
      const response = new Response(JSON.stringify({ message: 'Error' }), { status: 500 })

      const error = ApiError.fromResponse(response)

      expect(error.code).toBe('500')
    })
  })

  describe('ApiError.fromError', () => {
    it('should return original ApiError', () => {
      const original = new ApiError('Test')
      const error = ApiError.fromError(original)
      expect(error).toBe(original)
    })

    it('should convert Error to ApiError', () => {
      const error = ApiError.fromError(new Error('Network error'))
      expect(error).toBeInstanceOf(ApiError)
      expect(error.message).toBe('Network error')
      expect(error.code).toBe('NETWORK_ERROR')
    })

    it('should handle non-Error values', () => {
      const error = ApiError.fromError('string error')
      expect(error).toBeInstanceOf(ApiError)
      expect(error.message).toBe('string error')
    })
  })

  describe('isApiError', () => {
    it('should return true for ApiError', () => {
      const error = new ApiError('Test')
      expect(isApiError(error)).toBe(true)
    })

    it('should return false for regular Error', () => {
      const error = new Error('Test')
      expect(isApiError(error)).toBe(false)
    })

    it('should return false for null', () => {
      expect(isApiError(null)).toBe(false)
    })

    it('should return false for undefined', () => {
      expect(isApiError(undefined)).toBe(false)
    })

    it('should return false for plain object', () => {
      expect(isApiError({ message: 'Test' })).toBe(false)
    })

    it('should return false for string', () => {
      expect(isApiError('error')).toBe(false)
    })
  })

  describe('getErrorMessage', () => {
    it('should get message from ApiError', () => {
      const error = new ApiError('API Error')
      expect(getErrorMessage(error)).toBe('API Error')
    })

    it('should get message from Error', () => {
      const error = new Error('Regular Error')
      expect(getErrorMessage(error)).toBe('Regular Error')
    })

    it('should return string as is', () => {
      expect(getErrorMessage('string error')).toBe('string error')
    })

    it('should return default message for unknown types', () => {
      expect(getErrorMessage(123)).toBe('未知错误')
      expect(getErrorMessage({})).toBe('未知错误')
    })
  })

  describe('ErrorCodes', () => {
    it('should have SUCCESS code', () => {
      expect(ErrorCodes.SUCCESS).toBe('0000')
    })

    it('should have all param error codes', () => {
      expect(ErrorCodes.PARAM_ERROR).toBe('1000')
      expect(ErrorCodes.PARAM_VALIDATION_ERROR).toBe('1001')
      expect(ErrorCodes.PARAM_MISSING_ERROR).toBe('1002')
      expect(ErrorCodes.PARAM_TYPE_ERROR).toBe('1003')
      expect(ErrorCodes.METHOD_NOT_ALLOWED).toBe('1004')
      expect(ErrorCodes.NOT_FOUND).toBe('1005')
      expect(ErrorCodes.FILE_TOO_LARGE).toBe('1006')
    })

    it('should have all user error codes', () => {
      expect(ErrorCodes.USER_NOT_FOUND).toBe('2001')
      expect(ErrorCodes.USER_PASSWORD_ERROR).toBe('2002')
      expect(ErrorCodes.USER_TOKEN_EXPIRED).toBe('2003')
      expect(ErrorCodes.USER_TOKEN_INVALID).toBe('2004')
    })

    it('should have all product error codes', () => {
      expect(ErrorCodes.PRODUCT_NOT_FOUND).toBe('3001')
      expect(ErrorCodes.PRODUCT_OFF_SHELF).toBe('3002')
    })

    it('should have all order error codes', () => {
      expect(ErrorCodes.ORDER_NOT_FOUND).toBe('4001')
      expect(ErrorCodes.ORDER_CREATE_FAILED).toBe('4002')
      expect(ErrorCodes.ORDER_CANCEL_FAILED).toBe('4003')
      expect(ErrorCodes.ORDER_STATUS_INVALID).toBe('4004')
    })

    it('should have all payment error codes', () => {
      expect(ErrorCodes.PAYMENT_INSUFFICIENT_BALANCE).toBe('5001')
      expect(ErrorCodes.PAYMENT_EXCHANGE_FAILED).toBe('5002')
    })

    it('should have network error codes', () => {
      expect(ErrorCodes.NETWORK_ERROR).toBe('NETWORK_ERROR')
      expect(ErrorCodes.TIMEOUT_ERROR).toBe('TIMEOUT_ERROR')
      expect(ErrorCodes.UNKNOWN_ERROR).toBe('UNKNOWN_ERROR')
    })
  })
})