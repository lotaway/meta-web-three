import React, { useEffect, useState, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useCart } from '@/hooks/useCart';

interface ShoppingCartItem {
  id: number;
  productName: string;
  productPic: string;
  price: number;
  quantity: number;
  checked: boolean;
  spDataStr: string;
}

interface CheckoutItemParam {
  id: number;
  productName: string;
  productPic: string;
  price: number;
  quantity: number;
}

export default function CartScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { items: cartItems, loading, error, fetchCart, removeItems, updateQuantity } = useCart();

  useEffect(() => {
    fetchCart();
  }, [fetchCart]);

  const [checkedItems, setCheckedItems] = useState<Set<number>>(new Set());

  const toggleItemCheck = (itemId: number) => {
    const newChecked = new Set(checkedItems);
    if (newChecked.has(itemId)) {
      newChecked.delete(itemId);
    } else {
      newChecked.add(itemId);
    }
    setCheckedItems(newChecked);
  };

  const toggleAllSelection = () => {
    if (cartItems.length === 0) return;
    if (checkedItems.size === cartItems.length) {
      setCheckedItems(new Set());
    } else {
      setCheckedItems(new Set(cartItems.map((item) => item.id)));
    }
  };

  const handleUpdateQuantity = async (id: number, delta: number) => {
    const item = cartItems.find((i) => i.id === id);
    if (!item) return;
    const newQty = item.quantity + delta;
    if (newQty < 1) {
      await handleRemove([id]);
    } else {
      await updateQuantity(id, newQty);
    }
  };

  const handleRemove = async (ids: number[]) => {
    try {
      await removeItems(ids);
    } catch (e) {
      Alert.alert('错误', '删除失败');
    }
  };

  const shoppingCartItems = useMemo(() => {
    return cartItems.map((item) => ({
      id: item.id,
      productName: item.productName || '商品',
      productPic: item.imageUrl || '',
      price: Number(item.price || 0),
      quantity: item.quantity,
      checked: checkedItems.has(item.id),
      spDataStr: item.skuSpec || '',
    }));
  }, [cartItems, checkedItems]);

  const selectedTotalPrice = useMemo(() => {
    return shoppingCartItems
      .filter((item) => item.checked)
      .reduce((sum, item) => sum + item.price * item.quantity, 0)
      .toFixed(2);
  }, [shoppingCartItems]);

  const isAllItemSelected = shoppingCartItems.length > 0 && shoppingCartItems.every((item) => item.checked);
  const selectedItems = useMemo<CheckoutItemParam[]>(
    () =>
      shoppingCartItems
        .filter((item) => item.checked)
        .map((item) => ({
          id: item.id,
          productName: item.productName,
          productPic: item.productPic,
          price: item.price,
          quantity: item.quantity,
        })),
    [shoppingCartItems],
  );

  if (loading && cartItems.length === 0) {
    return (
      <View style={[styles.container, styles.empty, { backgroundColor: colors.background }]}>
        <Text style={[styles.emptyText, { color: colors.fontColorDisabled }]}>加载中...</Text>
      </View>
    );
  }

  if (shoppingCartItems.length === 0) {
    return (
      <View style={[styles.container, styles.empty, { backgroundColor: colors.background }]}>
        <IconSymbol name="cart.fill" size={100} color={colors.fontColorDisabled} />
        <Text style={[styles.emptyText, { color: colors.fontColorDisabled }]}>购物车竟然是空的</Text>
        <TouchableOpacity style={[styles.emptyBtn, { backgroundColor: colors.primary }]}>
          <Text style={styles.emptyBtnText}>去逛逛</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView style={styles.cartList} showsVerticalScrollIndicator={false}>
        {shoppingCartItems.map((item) => (
          <CartItemRow 
            key={item.id} 
            item={item} 
            colors={colors} 
            onToggleCheck={toggleItemCheck} 
            onUpdateQuantity={handleUpdateQuantity} 
            onRemove={handleRemove} 
          />
        ))}
        <View style={styles.footerSpacer} />
      </ScrollView>

      <CartActionBar 
        selectedItems={selectedItems}
        selectedTotal={selectedTotalPrice} 
        isAllSelected={isAllItemSelected} 
        colors={colors} 
        onToggleAll={toggleAllSelection} 
      />
    </SafeAreaView>
  );
}

function CartItemRow({ item, colors, onToggleCheck, onUpdateQuantity, onRemove }: any) {
  return (
    <View style={styles.cartItem}>
      <TouchableOpacity onPress={() => onToggleCheck(item.id)} style={styles.checkbox}>
        <IconSymbol
          name={item.checked ? 'checkmark.circle.fill' : 'circle'}
          size={24}
          color={item.checked ? colors.primary : colors.fontColorDisabled}
        />
      </TouchableOpacity>
      <Image source={{ uri: item.productPic }} style={styles.productImage} />
      <View style={styles.itemRight}>
        <Text numberOfLines={1} style={[styles.title, { color: colors.fontColorDark }]}>
          {item.productName}
        </Text>
        <Text style={[styles.attr, { color: colors.fontColorLight }]}>{item.spDataStr}</Text>
        <View style={styles.priceRow}>
          <Text style={[styles.price, { color: colors.primary }]}>¥{item.price}</Text>
          <View style={styles.stepBox}>
            <TouchableOpacity onPress={() => onUpdateQuantity(item.id, -1)} style={styles.stepBtn}>
              <Text style={styles.stepText}>-</Text>
            </TouchableOpacity>
            <Text style={styles.stepVal}>{item.quantity}</Text>
            <TouchableOpacity onPress={() => onUpdateQuantity(item.id, 1)} style={styles.stepBtn}>
              <Text style={styles.stepText}>+</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
      <TouchableOpacity onPress={() => onRemove([item.id])} style={styles.delBtn}>
        <IconSymbol name="xmark" size={16} color={colors.fontColorLight} />
      </TouchableOpacity>
    </View>
  );
}

function CartActionBar({ selectedItems, selectedTotal, isAllSelected, colors, onToggleAll }: any) {
  const goToCheckout = () => {
    if (selectedItems.length === 0) {
      return;
    }
    router.push({
      pathname: '/checkout' as any,
      params: {
        items: JSON.stringify(selectedItems),
      },
    });
  };

  return (
    <View style={styles.actionSection}>
      <TouchableOpacity onPress={onToggleAll} style={styles.allCheck}>
        <IconSymbol
          name={isAllSelected ? 'checkmark.circle.fill' : 'circle'}
          size={24}
          color={isAllSelected ? colors.primary : colors.fontColorDisabled}
        />
        <Text style={[styles.allCheckText, { color: colors.fontColorBase }]}>全选</Text>
      </TouchableOpacity>
      <View style={styles.totalBox}>
        <Text style={[styles.totalLabel, { color: colors.fontColorDark }]}>
          合计: <Text style={[styles.totalPrice, { color: colors.primary }]}>¥{selectedTotal}</Text>
        </Text>
      </View>
      <TouchableOpacity
        style={[styles.confirmBtn, { backgroundColor: colors.primary }]}
        onPress={goToCheckout}
      >
        <Text style={styles.confirmBtnText}>结算</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  empty: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    marginTop: 20,
    fontSize: 16,
  },
  emptyBtn: {
    marginTop: 20,
    paddingHorizontal: 40,
    paddingVertical: 10,
    borderRadius: 20,
  },
  emptyBtnText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  cartList: {
    paddingHorizontal: 15,
    paddingVertical: 10,
  },
  cartItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
    position: 'relative',
  },
  checkbox: {
    marginRight: 10,
  },
  productImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
  },
  itemRight: {
    flex: 1,
    marginLeft: 15,
  },
  title: {
    fontSize: 15,
    fontWeight: '500',
  },
  attr: {
    fontSize: 12,
    marginTop: 4,
  },
  priceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 10,
  },
  price: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  stepBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: 4,
  },
  stepBtn: {
    width: 30,
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepText: {
    fontSize: 18,
    color: '#333',
  },
  stepVal: {
    width: 40,
    textAlign: 'center',
    fontSize: 14,
  },
  delBtn: {
    position: 'absolute',
    top: 10,
    right: 10,
    padding: 5,
  },
  actionSection: {
    position: 'absolute',
    left: 15,
    right: 15,
    bottom: 20,
    height: 60,
    backgroundColor: '#fff',
    borderRadius: 30,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    elevation: 8,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
  },
  allCheck: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  allCheckText: {
    marginLeft: 8,
    fontSize: 14,
  },
  totalBox: {
    flex: 1,
    alignItems: 'flex-end',
    paddingRight: 15,
  },
  totalLabel: {
    fontSize: 14,
  },
  totalPrice: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  confirmBtn: {
    paddingHorizontal: 25,
    paddingVertical: 10,
    borderRadius: 20,
  },
  confirmBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  footerSpacer: {
    height: 120,
  },
});
