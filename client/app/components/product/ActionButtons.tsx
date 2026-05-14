import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useTranslation } from 'react-i18next';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useCart } from '@/hooks/useCart';
import { productCollectionApi, DEFAULT_USER_ID } from '@/api/generated';
import { router } from 'expo-router';

interface ActionButtonsProps {
  colors: any;
  productDetails: any;
  flashInfo?: any;
  onOpenSKU: (type: 'cart' | 'buy') => void;
}

export function ActionButtons({ colors, productDetails, flashInfo, onOpenSKU }: ActionButtonsProps) {
  const { t } = useTranslation();
  const { addItem } = useCart();
  const [isFavorite, setIsFavorite] = useState(false);
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];

  useEffect(() => {
    loadFavoriteStatus();
  }, [productDetails?.id]);

  const loadFavoriteStatus = async () => {
    try {
      const response = await productCollectionApi.detail({
        xUserId: DEFAULT_USER_ID,
        productId: productDetails?.id,
      });
      setIsFavorite(response.data != null);
    } catch (error) {
      console.error('Failed to load favorite status:', error);
    }
  };

  const handleToggleFavorite = async () => {
    try {
      if (isFavorite) {
        await productCollectionApi.delete({
          xUserId: DEFAULT_USER_ID,
          productId: productDetails.id,
        });
        setIsFavorite(false);
        Alert.alert(t('common.success'), '取消收藏成功');
      } else {
        await productCollectionApi.add({
          productId: productDetails.id,
          productName: productDetails.name,
          productPic: productDetails.pic,
          productPrice: productDetails.price,
          productSubTitle: productDetails.subTitle,
        });
        setIsFavorite(true);
        Alert.alert(t('common.success'), '收藏成功');
      }
    } catch (error) {
      Alert.alert(t('common.error'), '操作失败');
    }
  };

  const handleAddToCart = useCallback(async () => {
    try {
      await addItem({
        productId: productDetails.id,
        quantity: 1,
      });
      Alert.alert(t('common.success'), t('home.product.added_to_cart'));
    } catch (error) {
      Alert.alert(t('common.error'), t('home.product.add_to_cart_failed'));
    }
  }, [addItem, productDetails, t]);

  return (
    <View style={[styles.bottomBar, { backgroundColor: themeColors.background }]}>
      <View style={styles.bottomLeft}>
        <TouchableOpacity onPress={() => router.replace('/(tabs)')} style={styles.bottomIconBtn}>
          <IconSymbol name="house" size={24} color={themeColors.fontColorBase} />
          <Text style={[styles.bottomIconText, { color: themeColors.fontColorBase }]}>{t('common.tabs.home')}</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => router.replace('/(tabs)/cart')} style={styles.bottomIconBtn}>
          <IconSymbol name="cart" size={24} color={themeColors.fontColorBase} />
          <Text style={[styles.bottomIconText, { color: themeColors.fontColorBase }]}>{t('common.tabs.cart')}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.bottomIconBtn} onPress={handleToggleFavorite}>
          <IconSymbol 
            name={isFavorite ? 'heart.fill' : 'heart'} 
            size={24} 
            color={isFavorite ? themeColors.primary : themeColors.fontColorBase} 
          />
          <Text style={[styles.bottomIconText, { color: isFavorite ? themeColors.primary : themeColors.fontColorBase }]}>{t('profile.menu.favorite')}</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.bottomRight}>
        {flashInfo ? (
          <TouchableOpacity 
            style={[styles.actionBtn, styles.flashBuyBtn]}
            onPress={() => onOpenSKU('buy')}
          >
            <Text style={styles.actionBtnFlashPrice}>￥{flashInfo.flashPrice}</Text>
            <Text style={styles.actionBtnText}>闪购价购买</Text>
          </TouchableOpacity>
        ) : (
          <>
            <TouchableOpacity 
              style={[styles.actionBtn, styles.cartBtn]}
              onPress={() => onOpenSKU('cart')}
            >
              <Text style={styles.actionBtnText}>{t('home.product.add_to_cart')}</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.actionBtn, styles.buyBtn, { backgroundColor: themeColors.primary }]}
              onPress={() => onOpenSKU('buy')}
            >
              <Text style={styles.actionBtnText}>{t('home.product.buy_now')}</Text>
            </TouchableOpacity>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 80,
    flexDirection: 'row',
    paddingHorizontal: 15,
    paddingTop: 10,
    paddingBottom: 25,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  bottomLeft: {
    flexDirection: 'row',
    flex: 1,
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  bottomIconBtn: {
    alignItems: 'center',
  },
  bottomIconText: {
    fontSize: 10,
    marginTop: 4,
  },
  bottomRight: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
  actionBtn: {
    flex: 1,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cartBtn: {
    backgroundColor: '#ff9500',
    borderTopLeftRadius: 22,
    borderBottomLeftRadius: 22,
  },
  buyBtn: {
    borderTopRightRadius: 22,
    borderBottomRightRadius: 22,
  },
  flashBuyBtn: {
    flex: 1,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 22,
    backgroundColor: '#ff2d2d',
    flexDirection: 'row',
    gap: 4,
  },
  actionBtnFlashPrice: {
    color: '#fff',
    fontSize: 12,
    textDecorationLine: 'line-through',
    opacity: 0.8,
  },
  actionBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
});
