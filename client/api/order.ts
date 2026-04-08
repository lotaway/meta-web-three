import { orderApi } from './generated';
import { OrderCreateRequest } from '@/src/generated/api/models/OrderCreateRequest';

export interface OrderItem {
  productId: number;
  productQuantity: number;
  productPrice: number;
  productName: string;
}

export interface OrderCreateData {
  items: OrderItem[];
  remark?: string;
}

export async function createOrder(xUserID: number, data: OrderCreateData) {
  return orderApi.create({
    xUserID,
    orderCreateRequest: {
      remark: data.remark,
      items: data.items,
    },
  });
}

export async function fetchOrderList(xUserID: number, pageNum: number = 1, pageSize: number = 10) {
  return orderApi.list({
    xUserID,
    pageNum,
    pageSize,
  });
}

export async function fetchOrderDetail(xUserID: number, id: number) {
  return orderApi.detail({
    xUserID,
    id,
  });
}

export async function cancelOrder(xUserID: number, id: number) {
  return orderApi.cancel({
    xUserID,
    id,
  });
}

export async function confirmReceiveOrder(xUserID: number, id: number) {
  return orderApi.confirmReceive({
    xUserID,
    id,
  });
}
