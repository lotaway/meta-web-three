import { dayjs } from 'element-plus'

// DateTime formatting functions - updated to accept undefined/null
export const formatDateTime = (time: string | undefined | null): string => {
  if (!time) {
    return 'N/A'
  }
  return dayjs(time).format('YYYY-MM-DD HH:mm:ss')
}

export const formatDate = (time: string | undefined | null): string => {
  if (!time) {
    return 'N/A'
  }
  return dayjs(time).format('YYYY-MM-DD')
}

export const formatTime = (time: string | undefined | null): string => {
  if (!time) {
    return 'N/A'
  }
  return dayjs(time).format('HH:mm:ss')
}

// Number formatting utilities
export const formatNumber = (value: number | string | undefined | null, decimals: number = 2): string => {
  if (value === undefined || value === null || value === '') {
    return 'N/A'
  }
  const num = typeof value === 'string' ? parseFloat(value) : value
  if (isNaN(num)) {
    return 'N/A'
  }
  return num.toFixed(decimals)
}

// Format currency
export const formatCurrency = (value: number | string | undefined | null, currency: string = '¥'): string => {
  const formatted = formatNumber(value, 2)
  return formatted === 'N/A' ? 'N/A' : `${currency}${formatted}`
}

// Format percentage
export const formatPercentage = (value: number | string | undefined | null, decimals: number = 2): string => {
  const formatted = formatNumber(value, decimals)
  return formatted === 'N/A' ? 'N/A' : `${formatted}%`
}