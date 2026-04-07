import { productApi } from './generated';
import { ProductDTO } from '@/src/generated/api/models/ProductDTO';
import { ProductDetailDTO } from '@/src/generated/api/models/ProductDetailDTO';

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

function mapProductDetailDTO(dto: ProductDetailDTO): ProductDetail {
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

function mapProductDTO(dto: ProductDTO): ProductListItem {
  return {
    id: dto.id ?? 0,
    name: dto.name ?? '',
    pic: dto.imageUrl ?? '',
    price: parseFloat(dto.price ?? '0'),
    subTitle: undefined,
  };
}

export async function fetchProductDetail(id: number): Promise<{ data: ProductDetailResponse }> {
  const response = await productApi.getProduct({ id });
  const product = mapProductDetailDTO(response.data!);
  return { data: { product } };
}

export async function fetchProductList(params: ProductListParams): Promise<{ data: { list: ProductListItem[]; total: number } }> {
  const response = await productApi.listProducts({
    categoryId: params.categoryId,
    keyword: params.keyword,
  });
  const list = (response.data ?? []).map(mapProductDTO);
  return { data: { list, total: list.length } };
}