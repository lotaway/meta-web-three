import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { ProductInfoRow } from './ProductInfoRow';

interface ProductInfoProps {
  productDetails: any;
  colors: any;
  onOpenSKU: (type: 'cart' | 'buy') => void;
}

export function ProductInfo({ productDetails, colors, onOpenSKU }: ProductInfoProps) {
  const { t } = useTranslation();
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];

  return (
    <View>
      <View style={styles.introSection}>
        <Text style={[styles.title, { color: themeColors.fontColorDark }]}>{productDetails.name}</Text>
        <Text style={[styles.subtitle, { color: themeColors.fontColorLight }]}>{productDetails.subTitle}</Text>
        <View style={styles.priceRow}>
          <Text style={[styles.priceTag, { color: themeColors.primary }]}>¥</Text>
          <Text style={[styles.priceValue, { color: themeColors.primary }]}>{productDetails.price}</Text>
          <Text style={[styles.originalPrice, { color: themeColors.fontColorLight }]}>¥{productDetails.originalPrice}</Text>
        </View>
        <View style={styles.statsBar}>
          <Text style={[styles.statsText, { color: themeColors.fontColorLight }]}>{t('home.product.sales')}: {productDetails.sale}</Text>
          <Text style={[styles.statsText, { color: themeColors.fontColorLight }]}>{t('home.product.stock')}: {productDetails.stock}</Text>
          <Text style={[styles.statsText, { color: themeColors.fontColorLight }]}>{t('home.product.comments')}: 86</Text>
        </View>
      </View>

      <View style={styles.cList}>
        <ProductInfoRow title={t('home.product.buy_type')} content={t('home.product.select_specs')} showArrow colors={themeColors} onPress={() => onOpenSKU('cart')} />
        <ProductInfoRow title={t('home.product.params')} content={t('home.product.view')} showArrow colors={themeColors} />
        <ProductInfoRow 
          title={t('home.product.coupons')} 
          content={t('home.product.get_coupons')} 
          showArrow 
          colors={themeColors} 
          contentStyle={{ color: themeColors.primary }} 
        />
        <ProductInfoRow title={t('home.product.service')} content={t('home.product.service_content')} colors={themeColors} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  introSection: {
    backgroundColor: '#fff',
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    marginBottom: 15,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  priceTag: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  priceValue: {
    fontSize: 24,
    fontWeight: 'bold',
    marginHorizontal: 4,
  },
  originalPrice: {
    fontSize: 14,
    textDecorationLine: 'line-through',
    marginLeft: 10,
  },
  statsBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  statsText: {
    fontSize: 12,
  },
  cList: {
    backgroundColor: '#fff',
    marginTop: 15,
    paddingHorizontal: 20,
  },
});
