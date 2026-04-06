import { MallClient } from './client';

export interface AdvertiseItem {
  id: number;
  pic: string;
  name: string;
  link?: string;
}

export interface BrandItem {
  id: number;
  name: string;
  logo: string;
  productCount: number;
}

export interface FlashProductItem {
  id: number;
  name: string;
  pic: string;
  price: number;
  originalPrice?: number;
}

export interface FlashPromotion {
  productList: FlashProductItem[];
  startTime?: string;
  endTime?: string;
}

export interface ProductItem {
  id: number;
  name: string;
  pic: string;
  price: number;
  subTitle?: string;
  originalPrice?: number;
}

export interface MallHomeContent {
  advertiseList: AdvertiseItem[];
  brandList: BrandItem[];
  homeFlashPromotion: FlashPromotion | null;
  newProductList: ProductItem[];
  hotProductList: ProductItem[];
  subjectList: any[];
}

export function fetchMallHomeContent() {
  return MallClient.get<MallHomeContent>('/home/content');
}

export interface ProductListParams {
  pageSize: number;
  pageNum: number;
}

export interface ProductListResponse {
  list: ProductItem[];
  total: number;
}

export function fetchRecommendMallProductList(params: ProductListParams) {
  return MallClient.get<ProductListResponse>('/home/recommendProductList', { params });
}

export interface CategoryItem {
  id: number;
  name: string;
  parentId: number;
  children?: CategoryItem[];
}

export function fetchProductCategoryList(parentId: number) {
  return MallClient.get<CategoryItem[]>(`/home/productCateList/${parentId}`);
}

export function fetchNewMallProductList(params: ProductListParams) {
  return MallClient.get<ProductListResponse>('/home/newProductList', { params });
}

export function fetchHotMallProductList(params: ProductListParams) {
  return MallClient.get<ProductListResponse>('/home/hotProductList', { params });
}
