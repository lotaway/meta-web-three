import { Configuration } from '@/src/generated/api/runtime';
import { 
  ProductManagementApi, 
  ProductCategoryManagementApi, 
  ProductBrandManagementApi,
  OrderControllerApi,
  CartManagementApi,
  PasskeyApi,
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
export const passkeyApi = new PasskeyApi(apiConfig);

/**
 * Relying Party ID — 应与后端配置及 iOS associated domains 保持一致。
 * 开发期间可使用 localhost；生产环境改为实际域名（如 "example.com"）。
 */
export const RP_ID: string = process.env.EXPO_PUBLIC_RP_ID ?? 'localhost';