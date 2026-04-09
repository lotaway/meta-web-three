import { Brand } from '@/src/generated/api/models/Brand';
import { ProductDTO } from '@/src/generated/api/models/ProductDTO';

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

export interface ProductItem {
  id: number;
  name: string;
  pic: string;
  price: number;
  subTitle?: string;
  originalPrice?: number;
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

export interface MallHomeContent {
  advertiseList: AdvertiseItem[];
  brandList: BrandItem[];
  homeFlashPromotion: FlashPromotion | null;
  newProductList: ProductItem[];
  hotProductList: ProductItem[];
  subjectList: unknown[];
}

export interface CategoryItem {
  id: number;
  name: string;
  parentId: number;
  children?: CategoryItem[];
}

export function toBrandItem(brand: Brand): BrandItem {
  return {
    id: brand.id ?? 0,
    name: brand.name ?? '',
    logo: brand.logo ?? '',
    productCount: brand.productCount ?? 0,
  };
}

export function toAdvertiseItem(input: any): AdvertiseItem {
  return {
    id: input?.id ?? 0,
    name: input?.name ?? '',
    pic: input?.pic ?? '',
    link: input?.url ?? undefined,
  };
}

export function toProductItem(dto: ProductDTO): ProductItem {
  return {
    id: dto.id ?? 0,
    name: dto.name ?? '',
    pic: dto.imageUrl ?? '',
    price: parseFloat(dto.price ?? '0'),
    subTitle: undefined,
    originalPrice: undefined,
  };
}
