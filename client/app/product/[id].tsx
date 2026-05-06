import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Dimensions,
  useWindowDimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, router, Stack } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useCart } from '@/hooks/useCart';
import { ProductDetailContainer } from '@/containers/product/ProductDetailContainer';
import RenderHTML from 'react-native-render-html';
import { SkuSelector, SpecGroup, SKUInfo } from '@/components/product/SkuSelector';
import { ProductInfo } from '@/components/product/ProductInfo';
import { CommentPreview } from '@/components/product/CommentPreview';
import { RelatedProducts } from '@/components/product/RelatedProducts';
import { ActionButtons } from '@/components/product/ActionButtons';

const { width: PAGE_WIDTH } = Dimensions.get('window');

function getProductImages(productDetails: any) {
  if (!productDetails) return [];
  const albumPics = productDetails.albumPics ? productDetails.albumPics.split(',') : [];
  return [productDetails.pic, ...albumPics].filter(Boolean);
}

function getDefaultSpecs(): SpecGroup[] {
  return [
    { id: 'color', name: '颜色', options: [{ id: 'black', name: '黑色' }, { id: 'white', name: '白色' }, { id: 'blue', name: '蓝色' }] },
    { id: 'size', name: '尺寸', options: [{ id: 's', name: 'S' }, { id: 'm', name: 'M' }, { id: 'l', name: 'L' }] },
  ];
}

function ProductBanner({ images }: { images: string[] }) {
  return (
    <ScrollView horizontal pagingEnabled showsHorizontalScrollIndicator={false} style={styles.bannerContainer}>
      {images.map((uri, index) => (
        <Image key={index} source={{ uri }} style={styles.bannerImage} />
      ))}
    </ScrollView>
  );
}

function ProductDetailHTML({ contentWidth, html, colors, t }: any) {
  return (
    <View style={styles.detailDesc}>
      <View style={styles.detailHeader}>
        <View style={styles.headerLine} />
        <Text style={[styles.headerText, { color: colors.fontColorDark }]}>{t('home.product.graph_detail')}</Text>
        <View style={styles.headerLine} />
      </View>
      <View style={styles.htmlWrapper}>
        <RenderHTML
          contentWidth={contentWidth - 20}
          source={{ html: html || `<p>${t('home.product.no_detail')}</p>` }}
          tagsStyles={{ img: { width: '100%', height: 'auto' }, p: { color: colors.fontColorDark } }}
        />
      </View>
    </View>
  );
}

function LoadingView({ t }: any) {
  return (
    <View style={[styles.container, styles.center]}>
      <Text>{t('common.loading')}</Text>
    </View>
  );
}

function ProductDetailBody({ productDetails, colors, t, contentWidth, productImages }: any) {
  return (
    <ScrollView showsVerticalScrollIndicator={false}>
      <ProductBanner images={productImages} />
      <ProductInfo productDetails={productDetails} colors={colors} onOpenSKU={() => {}} />
      <CommentPreview productId={productDetails?.id} productName={productDetails?.name} colors={colors} />
      <ProductDetailHTML contentWidth={contentWidth} html={productDetails.detailMobileHtml} colors={colors} t={t} />
      <RelatedProducts productId={productDetails?.id} colors={colors} />
      <View style={styles.footerSpacer} />
    </ScrollView>
  );
}

function ProductDetailFooter({ colors, productDetails, onOpenSKU, skuSelectorVisible, skuActionType, productImages, mockSpecs, mockSKUs, onSKUConfirm }: any) {
  return (
    <>
      <ActionButtons colors={colors} productDetails={productDetails} onOpenSKU={onOpenSKU} />
      <SkuSelector
        visible={skuSelectorVisible}
        productImage={productImages[0]}
        productName={productDetails?.name}
        productPrice={productDetails?.price}
        productOriginalPrice={productDetails?.originalPrice}
        specs={mockSpecs}
        skus={mockSKUs}
        onClose={() => onSKUConfirm(null, null, 0, true)}
        onConfirm={(sel: any, sku: any, qty: number) => onSKUConfirm(sel, sku, qty, false)}
        actionType={skuActionType}
      />
    </>
  );
}

function useProductDetailLogic(t: any, id: any) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { width: contentWidth } = useWindowDimensions();
  const { addItem } = useCart();  
  const [skuSelectorVisible, setSkuSelectorVisible] = useState(false);
  const [skuActionType, setSkuActionType] = useState<'cart' | 'buy'>('cart');

  const handleOpenSKU = (actionType: 'cart' | 'buy') => {
    setSkuActionType(actionType);
    setSkuSelectorVisible(true);
  };

  const handleSKUConfirm = (selectedSpecs: any, sku: SKUInfo | null, quantity: number, closeOnly: boolean) => {
    setSkuSelectorVisible(false);
    if (closeOnly) return;
    if (skuActionType === 'cart') {
      addItem({ productId: (sku || {}).id, quantity, skuId: (sku || {}).skuId });
      Alert.alert(t('common.success'), t('home.product.added_to_cart'));
    } else {
      router.push({ pathname: '/checkout', params: { productId: id, quantity, skuId: sku?.skuId } });
    }
  };

  return { colors, contentWidth, skuSelectorVisible, skuActionType, handleOpenSKU, handleSKUConfirm };
}

export default function ProductDetailScreen() {
  const { t } = useTranslation();
  const { id } = useLocalSearchParams();
  const { colors, contentWidth, skuSelectorVisible, skuActionType, handleOpenSKU, handleSKUConfirm } = useProductDetailLogic(t, id);

  return (
    <ProductDetailContainer productId={id ? Number(id) : null}>
      {({ productDetails, isPageLoading }) => {
        if (isPageLoading || !productDetails) return <LoadingView t={t} />;
        const productImages = getProductImages(productDetails);
        const mockSpecs: SpecGroup[] = productDetails?.specs || getDefaultSpecs();
        const mockSKUs: SKUInfo[] = productDetails?.skus || [];
        return (
          <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
            <Stack.Screen options={{ title: t('home.product.detail_title'), headerTransparent: true, headerTitle: '' }} />
            <ProductDetailBody productDetails={productDetails} colors={colors} t={t} contentWidth={contentWidth} productImages={productImages} />
            <ProductDetailFooter colors={colors} productDetails={productDetails} onOpenSKU={handleOpenSKU} skuSelectorVisible={skuSelectorVisible} skuActionType={skuActionType} productImages={productImages} mockSpecs={mockSpecs} mockSKUs={mockSKUs} onSKUConfirm={handleSKUConfirm} />
          </SafeAreaView>
        );
      }}
    </ProductDetailContainer>
  );
}

const styles = StyleSheet.create({
  container: { flex:1 },
  center: { justifyContent: 'center', alignItems: 'center' },
  bannerContainer: { height: PAGE_WIDTH },
  bannerImage: { width: PAGE_WIDTH, height: PAGE_WIDTH, resizeMode: 'cover' },
  detailDesc: { backgroundColor: '#fff', marginTop: 15, paddingBottom: 20 },
  detailHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 20 },
  headerLine: { width: 50, height: 1, backgroundColor: '#ddd', marginHorizontal: 15 },
  headerText: { fontSize: 16, fontWeight: '600' },
  htmlWrapper: { paddingHorizontal: 10 },
  footerSpacer: { height: 100 },
});
