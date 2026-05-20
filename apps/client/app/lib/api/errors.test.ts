import { ApiError, ErrorCodes, isApiError, getErrorMessage } from '../api/errors'

describe('ApiError', () => {
  it('should create error with all properties', () => {
    const error = new ApiError('Test error', 'TEST_CODE', 400, { details: 'test' })

    expect(error.message).toBe('Test error')
    expect(error.code).toBe('TEST_CODE')
    expect(error.status).toBe(400)
    expect(error.data).toEqual({ details: 'test' })
    expect(error.isApiError).toBe(true)
  })

  it('should create error from response', () => {
    const response = new Response(JSON.stringify({ code: '400', message: 'Bad Request' }), {
      status: 400,
    })

    const error = ApiError.fromResponse(response)

    expect(error.status).toBe(400)
    expect(error.code).toBe('400')
  })

  it('should identify ApiError correctly', () => {
    const apiError = new ApiError('test')
    const regularError = new Error('test')

    expect(isApiError(apiError)).toBe(true)
    expect(isApiError(regularError)).toBe(false)
    expect(isApiError(null)).toBe(false)
    expect(isApiError(undefined)).toBe(false)
  })

  it('should get error message correctly', () => {
    const apiError = new ApiError('API Error')
    const regularError = new Error('Regular Error')

    expect(getErrorMessage(apiError)).toBe('API Error')
    expect(getErrorMessage(regularError)).toBe('Regular Error')
    expect(getErrorMessage('string error')).toBe('string error')
  })
})

describe('ErrorCodes', () => {
  it('should have correct success code', () => {
    expect(ErrorCodes.SUCCESS).toBe('0000')
  })

  it('should have correct param error codes', () => {
    expect(ErrorCodes.PARAM_ERROR).toBe('1000')
    expect(ErrorCodes.PARAM_VALIDATION_ERROR).toBe('1001')
    expect(ErrorCodes.PARAM_MISSING_ERROR).toBe('1002')
  })

  it('should have correct user error codes', () => {
    expect(ErrorCodes.USER_NOT_FOUND).toBe('2001')
    expect(ErrorCodes.USER_TOKEN_EXPIRED).toBe('2003')
  })
})