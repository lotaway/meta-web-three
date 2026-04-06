import { MallClient } from './client';

export interface ProductDetail {
  id: number;
  name: string;
  pic: string;
  albumPics: string;
  price: number;
  originalPrice: number;
  subTitle: string;
  description?: string;
  detailMobileHtml: string;
  brandName?: string;
  categoryName?: string;
  sale: number;
  stock: number;
}

export interface ProductDetailResponse {
  product: ProductDetail;
}

export interface ProductListParams {
  pageSize: number;
  pageNum: number;
  categoryId?: number;
  keyword?: string;
}

export interface ProductListItem {
  id: number;
  name: string;
  pic: string;
  price: number;
  subTitle?: string;
}

export function fetchProductDetail(id: number) {
  return MallClient.get<ProductDetailResponse>(`/product/detail/${id}`);
}

export function fetchProductList(params: ProductListParams) {
  return MallClient.get<{ list: ProductListItem[]; total: number }>('/product/queryProductList', { params });
}
