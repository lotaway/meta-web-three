import { Configuration } from '@/src/generated/api/runtime';
import { 
  ProductManagementApi, 
  ProductCategoryManagementApi, 
  ProductBrandManagementApi,
  OrderControllerApi,
  CartManagementApi
} from '@/src/generated/api';

export const API_BASE_URL = process.env.NEXT_PUBLIC_BACK_API_HOST ?? 'http://localhost:10081';

export const apiConfig = new Configuration({
  basePath: API_BASE_URL,
  credentials: 'include',
});

export const productApi = new ProductManagementApi(apiConfig);
export const categoryApi = new ProductCategoryManagementApi(apiConfig);
export const brandApi = new ProductBrandManagementApi(apiConfig);
export const orderApi = new OrderControllerApi(apiConfig);
export const cartApi = new CartManagementApi(apiConfig);