import { categoryApi, productApi, brandApi } from './generated';
import { ProductCategory } from '@/src/generated/api/models/ProductCategory';
import { Brand } from '@/src/generated/api/models/Brand';

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

function mapBrandItem(brand: Brand): BrandItem {
  return {
    id: brand.id ?? 0,
    name: brand.name ?? '',
    logo: brand.logo ?? '',
    productCount: brand.productCount ?? 0,
  };
}

export async function fetchMallHomeContent(): Promise<{ data: MallHomeContent }> {
  const [brandRes] = await Promise.all([
    brandApi.list(),
  ]);
  
  const brandList = (brandRes.data ?? []).map(mapBrandItem);
  
  return {
    data: {
      advertiseList: [],
      brandList,
      homeFlashPromotion: null,
      newProductList: [],
      hotProductList: [],
      subjectList: [],
    },
  };
}

export interface ProductListParams {
  pageSize: number;
  pageNum: number;
}

export interface ProductListResponse {
  list: ProductItem[];
  total: number;
}

export function fetchRecommendMallProductList(params: ProductListParams): Promise<ProductListResponse> {
  return Promise.resolve({ list: [], total: 0 });
}

export interface CategoryItem {
  id: number;
  name: string;
  parentId: number;
  children?: CategoryItem[];
}

function mapProductCategory(pc: ProductCategory): CategoryItem {
  return {
    id: pc.id ?? 0,
    name: pc.name ?? '',
    parentId: pc.parentId ?? 0,
  };
}

export async function fetchProductCategoryList(parentId: number): Promise<{ data: CategoryItem[] }> {
  const response = await categoryApi.viewChildren({ parentId });
  const categories = response.data?.map(mapProductCategory) ?? [];
  return { data: categories };
}

export async function fetchNewMallProductList(params: ProductListParams): Promise<{ data: ProductListResponse }> {
  const response = await productApi.listProducts({ keyword: 'new' });
  const list = (response.data ?? []).map(p => ({
    id: p.id ?? 0,
    name: p.name ?? '',
    pic: p.imageUrl ?? '',
    price: parseFloat(p.price ?? '0'),
    subTitle: undefined,
    originalPrice: undefined,
  }));
  return { data: { list, total: list.length } };
}

export async function fetchHotMallProductList(params: ProductListParams): Promise<{ data: ProductListResponse }> {
  const response = await productApi.listProducts({ keyword: 'hot' });
  const list = (response.data ?? []).map(p => ({
    id: p.id ?? 0,
    name: p.name ?? '',
    pic: p.imageUrl ?? '',
    price: parseFloat(p.price ?? '0'),
    subTitle: undefined,
    originalPrice: undefined,
  }));
  return { data: { list, total: list.length } };
}