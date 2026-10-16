export interface BlobInfo {
  blob: () => Blob
  filename: () => string
}

export type UploadProgressCallback = (percent: number) => void

type ResolveUrl = (url: string) => void
type RejectWith = (reason: Error) => void

const toPercent = (event: ProgressEvent): number => {
  if (event.total <= 0) {
    return 100
  }
  return (event.loaded / event.total) * 100
}

const httpErrorMessage = (status: number): string =>
  `Image upload failed with HTTP status ${status}`

const parseUploadedUrl = (responseText: string): string => {
  const payload = JSON.parse(responseText)
  if (payload?.code !== 200 || !payload?.data?.url) {
    throw new Error('Image upload failed: unexpected response payload')
  }
  return payload.data.url as string
}

const settleUpload = (
  request: XMLHttpRequest,
  resolve: ResolveUrl,
  reject: RejectWith
): void => {
  if (request.status < 200 || request.status >= 300) {
    reject(new Error(httpErrorMessage(request.status)))
    return
  }
  try {
    resolve(parseUploadedUrl(request.responseText))
  } catch (error) {
    reject(error instanceof Error ? error : new Error(String(error)))
  }
}

export const createImageUploadHandler =
  (uploadUrl: string) =>
  (blobInfo: BlobInfo, progress: UploadProgressCallback): Promise<string> =>
    new Promise((resolve, reject) => {
      const request = new XMLHttpRequest()
      request.open('POST', uploadUrl)
      request.upload.onprogress = (event) => progress(toPercent(event))
      request.onload = () => settleUpload(request, resolve, reject)
      request.onerror = () => reject(new Error(httpErrorMessage(request.status)))
      const formData = new FormData()
      formData.append('file', blobInfo.blob(), blobInfo.filename())
      request.send(formData)
    })
