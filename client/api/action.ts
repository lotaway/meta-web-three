import { API_BASE_URL } from './generated'

async function request<T>(path: string, options: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE_URL}${path}`, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
    })
    if (!res.ok) throw new Error(`Request failed: ${res.status}`)
    return res.json() as Promise<T>
}

export async function addProductCollection(xUserID: number, payload: { productId: number; productName: string; productPic?: string }) {
    return request(`/v1/action/productCollection/add`, {
        method: 'POST',
        headers: { 'X-User-ID': String(xUserID) },
        body: JSON.stringify(payload),
    })
}

export async function listProductCollections<T = any[]>(xUserID: number) {
    return request<{ code: number; data: T }>(`/v1/action/productCollection/list`, {
        method: 'GET',
        headers: { 'X-User-ID': String(xUserID) },
    })
}

export async function deleteProductCollection(xUserID: number, productId: number) {
    const url = new URL(`${API_BASE_URL}/v1/action/productCollection/delete`)
    url.searchParams.set('productId', String(productId))
    const res = await fetch(url.toString(), {
        method: 'DELETE',
        headers: { 'X-User-ID': String(xUserID) },
    })
    if (!res.ok) throw new Error(`Request failed: ${res.status}`)
    return res.json()
}

export async function addReadHistory(xUserID: number, payload: { productId: number; productName: string; productPic?: string }) {
    return request(`/v1/action/readHistory/create`, {
        method: 'POST',
        headers: { 'X-User-ID': String(xUserID) },
        body: JSON.stringify(payload),
    })
}

export async function listReadHistory<T = any[]>(xUserID: number) {
    return request<{ code: number; data: T }>(`/v1/action/readHistory/list`, {
        method: 'GET',
        headers: { 'X-User-ID': String(xUserID) },
    })
}

export async function addBrandAttention(xUserID: number, payload: { brandId: number; brandName: string; brandLogo?: string }) {
    return request(`/v1/action/brandAttention/add`, {
        method: 'POST',
        headers: { 'X-User-ID': String(xUserID) },
        body: JSON.stringify(payload),
    })
}

export async function listBrandAttention<T = any[]>(xUserID: number) {
    return request<{ code: number; data: T }>(`/v1/action/brandAttention/list`, {
        method: 'GET',
        headers: { 'X-User-ID': String(xUserID) },
    })
}

