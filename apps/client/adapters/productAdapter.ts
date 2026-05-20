import { ProductDTO } from '@/src/generated/api/models/ProductDTO';
import { ProductDetailDTO } from '@/src/generated/api/models/ProductDetailDTO';

export interface ProductDetailVM {
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

export interface ProductListItemVM {
  id: number;
  name: string;
  pic: string;
  price: number;
  subTitle?: string;
}

export function toProductDetailVM(dto: ProductDetailDTO): ProductDetailVM {
  return {
    id: dto.id ?? 0,
    name: dto.goodsName ?? '',
    pic: dto.imageUrl ?? '',
    albumPics: (dto.pictures ?? []).join(','),
    price: dto.salePrice ?? 0,
    originalPrice: dto.marketPrice ?? 0,
    subTitle: dto.goodsRemark ?? '',
    detailMobileHtml: '',
    sale: parseInt(String(dto.saleCount ?? '0'), 10),
    stock: dto.inventory ?? 0,
  };
}

export function toProductListItemVM(dto: ProductDTO): ProductListItemVM {
  return {
    id: dto.id ?? 0,
    name: dto.name ?? '',
    pic: dto.imageUrl ?? '',
    price: parseFloat(dto.price ?? '0'),
    subTitle: undefined,
  };
}
