import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useTranslation } from 'react-i18next';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useCart } from '@/hooks/useCart';
import { productCollectionApi, DEFAULT_USER_ID, API_BASE_URL } from '@/api/generated';
import { router } from 'expo-router';

interface ActionButtonsProps {
  colors: any;
  productDetails: any;
  flashInfo?: any;
  onOpenSKU: (type: 'cart' | 'buy') => void;
}

type Translate = (key: string) => string;

async function fetchFavoriteState(productId: number | undefined, t: Translate) {
  try {
    const response = await productCollectionApi.listCollection1({
      xUserId: DEFAULT_USER_ID,
    });
    const isFav = response.data?.some((item: any) => item.id === productId);
    return { isFav: !!isFav, failed: false };
  } catch (error) {
    console.error('Failed to load favorite status:', error);
    Alert.alert(t('favorites.load_failed_title'), t('favorites.load_failed'));
    return { isFav: false, failed: true };
  }
}

async function removeFavoriteRequest(productDetails: any, t: Translate, setIsFavorite: (value: boolean) => void) {
  await productCollectionApi.deleteCollection2({
    xUserId: DEFAULT_USER_ID,
    productId: productDetails.id,
  });
  setIsFavorite(false);
  Alert.alert(t('common.success'), t('favorites.remove_success'));
}

async function addFavoriteRequest(productDetails: any, t: Translate, setIsFavorite: (value: boolean) => void) {
  await productCollectionApi.addCollection1({
    xUserId: DEFAULT_USER_ID,
    collectionParam: {
      productId: productDetails.id,
      productName: productDetails.name,
      productPic: productDetails.pic,
    },
  });
  setIsFavorite(true);
  Alert.alert(t('common.success'), t('favorites.add_success'));
}

async function toggleFavoriteRequest(
  isFavorite: boolean,
  productDetails: any,
  t: Translate,
  setIsFavorite: (value: boolean) => void,
) {
  try {
    if (isFavorite) {
      await removeFavoriteRequest(productDetails, t, setIsFavorite);
    } else {
      await addFavoriteRequest(productDetails, t, setIsFavorite);
    }
  } catch (error) {
    Alert.alert(t('common.error'), t('favorites.toggle_failed'));
  }
}

function useProductFavorite(productDetails: any, t: Translate) {
  const [isFavorite, setIsFavorite] = useState(false);

  useEffect(() => {
    fetchFavoriteState(productDetails?.id, t).then((result) => {
      if (!result.failed) setIsFavorite(result.isFav);
    });
  }, [productDetails?.id]);

  const handleToggleFavorite = () => toggleFavoriteRequest(isFavorite, productDetails, t, setIsFavorite);
  return { isFavorite, handleToggleFavorite };
}

async function addToCart(productDetails: any, addItem: (item: any) => Promise<unknown>, t: Translate) {
  try {
    await addItem({
      productId: productDetails.id,
      quantity: 1,
    });
    Alert.alert(t('common.success'), t('home.product.added_to_cart'));
  } catch (error) {
    Alert.alert(t('common.error'), t('home.product.add_to_cart_failed'));
  }
}

function createServiceHandler(productId: number | undefined, t: Translate) {
  return async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/cs/conversation/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Id': String(DEFAULT_USER_ID) },
        body: JSON.stringify({ channel: 'PRODUCT', productId, orderId: null }),
      })
      const json = await res.json()
      if (json.data?.sessionId) router.push(`/cs/${json.data.sessionId}`)
    } catch {
      Alert.alert(t('common.tip'), t('home.product.chat_failed'))
    }
  }
}

function BottomIconButton({ name, label, color, onPress }: {
  name: React.ComponentProps<typeof IconSymbol>['name'];
  label: string;
  color: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity accessibilityRole="button" accessibilityLabel={label} onPress={onPress} style={styles.bottomIconBtn}>
      <IconSymbol name={name} size={24} color={color} />
      <Text style={[styles.bottomIconText, { color }]}>{label}</Text>
    </TouchableOpacity>
  );
}

function BottomLeftBar({ themeColors, t, isFavorite, onToggleFavorite, onServicePress }: {
  themeColors: any;
  t: Translate;
  isFavorite: boolean;
  onToggleFavorite: () => void;
  onServicePress: () => void;
}) {
  const baseColor = themeColors.fontColorBase;
  const favoriteColor = isFavorite ? themeColors.primary : baseColor;
  return (
    <View style={styles.bottomLeft}>
      <BottomIconButton name="house" label={t('common.tabs.home')} color={baseColor} onPress={() => router.replace('/(tabs)')} />
      <BottomIconButton name="cart" label={t('common.tabs.cart')} color={baseColor} onPress={() => router.replace('/(tabs)/cart')} />
      <BottomIconButton name="message" label={t('home.product.service_title')} color={baseColor} onPress={onServicePress} />
      <BottomIconButton name={isFavorite ? 'heart.fill' : 'heart'} label={t('profile.menu.favorite')} color={favoriteColor} onPress={onToggleFavorite} />
    </View>
  );
}

function FlashBuyButton({ flashInfo, t, onPress }: { flashInfo: any; t: Translate; onPress: () => void }) {
  return (
    <TouchableOpacity
      accessibilityRole="button"
      style={[styles.actionBtn, styles.flashBuyBtn]}
      onPress={onPress}
    >
      <Text style={styles.actionBtnFlashPrice}>￥{flashInfo.flashPrice}</Text>
      <Text style={styles.actionBtnText}>{t('home.product.flash_buy')}</Text>
    </TouchableOpacity>
  );
}

function NormalActionButtons({ themeColors, t, onOpenSKU }: {
  themeColors: any;
  t: Translate;
  onOpenSKU: (type: 'cart' | 'buy') => void;
}) {
  return (
    <>
      <TouchableOpacity
        accessibilityRole="button"
        style={[styles.actionBtn, styles.cartBtn]}
        onPress={() => onOpenSKU('cart')}
      >
        <Text style={styles.actionBtnText}>{t('home.product.add_to_cart')}</Text>
      </TouchableOpacity>
      <TouchableOpacity
        accessibilityRole="button"
        style={[styles.actionBtn, styles.buyBtn, { backgroundColor: themeColors.primary }]}
        onPress={() => onOpenSKU('buy')}
      >
        <Text style={styles.actionBtnText}>{t('home.product.buy_now')}</Text>
      </TouchableOpacity>
    </>
  );
}

function BottomRightBar({ themeColors, t, flashInfo, onOpenSKU }: {
  themeColors: any;
  t: Translate;
  flashInfo?: any;
  onOpenSKU: (type: 'cart' | 'buy') => void;
}) {
  if (flashInfo) return <FlashBuyButton flashInfo={flashInfo} t={t} onPress={() => onOpenSKU('buy')} />;
  return <NormalActionButtons themeColors={themeColors} t={t} onOpenSKU={onOpenSKU} />;
}

function ActionButtonsView({ themeColors, t, isFavorite, flashInfo, onToggleFavorite, onServicePress, onOpenSKU }: {
  themeColors: any;
  t: Translate;
  isFavorite: boolean;
  flashInfo?: any;
  onToggleFavorite: () => void;
  onServicePress: () => void;
  onOpenSKU: (type: 'cart' | 'buy') => void;
}) {
  return (
    <View style={[styles.bottomBar, { backgroundColor: themeColors.background }]}>
      <BottomLeftBar themeColors={themeColors} t={t} isFavorite={isFavorite} onToggleFavorite={onToggleFavorite} onServicePress={onServicePress} />
      <BottomRightBar themeColors={themeColors} t={t} flashInfo={flashInfo} onOpenSKU={onOpenSKU} />
    </View>
  );
}

export function ActionButtons({ colors, productDetails, flashInfo, onOpenSKU }: ActionButtonsProps) {
  const { t } = useTranslation();
  const { addItem } = useCart();
  const { isFavorite, handleToggleFavorite } = useProductFavorite(productDetails, t);
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];
  const handleAddToCart = useCallback(() => addToCart(productDetails, addItem, t), [addItem, productDetails, t]);
  const handleServicePress = createServiceHandler(productDetails?.id, t);
  return (
    <ActionButtonsView
      themeColors={themeColors}
      t={t}
      isFavorite={isFavorite}
      flashInfo={flashInfo}
      onToggleFavorite={handleToggleFavorite}
      onServicePress={handleServicePress}
      onOpenSKU={onOpenSKU}
    />
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
