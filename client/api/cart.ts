import { cartApi } from './generated';
import { CartItemDTO } from '@/src/generated/api/models/CartItemDTO';

export async function fetchCartList(xUserID: number) {
  return cartApi.list({ xUserID });
}

export async function addCartItem(xUserID: number, cartItem: CartItemDTO) {
  return cartApi.add({
    xUserID,
    cartItemDTO: cartItem,
  });
}

export async function deleteCartItems(xUserID: number, ids: number[]) {
  return cartApi._delete({
    xUserID,
    ids,
  });
}

export async function updateCartItemQuantity(xUserID: number, id: number, quantity: number) {
  return cartApi.updateQuantity({
    xUserID,
    id,
    quantity,
  });
}

export async function clearCart(xUserID: number) {
  return cartApi.clear({ xUserID });
}
