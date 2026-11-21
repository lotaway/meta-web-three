import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  useWindowDimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, router, Stack } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useCart } from '@/hooks/useCart';
import { API_BASE_URL } from '@/api/generated';
import { ProductDetailContainer } from '@/containers/product/ProductDetailContainer';
import { recommendationHooks } from '@/lib/api/graphql-hooks';
import { useAuth } from '@/contexts/AuthContext';
import RenderHTML from 'react-native-render-html';
import { SkuSelector, SpecGroup, SKUInfo } from '@/components/product/SkuSelector';
import { ProductInfo } from '@/components/product/ProductInfo';
import { CommentPreview } from '@/components/product/CommentPreview';
import { RelatedProducts } from '@/components/product/RelatedProducts';
import { ActionButtons } from '@/components/product/ActionButtons';

type RouteId = string | string[] | undefined;
type Translate = ReturnType<typeof useTranslation>['t'];
type AppColors = typeof Colors.light;
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>;
type SetFlashInfo = React.Dispatch<React.SetStateAction<any>>;
type SkuActionType = 'cart' | 'buy';

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

async function loadFlashInfo(id: RouteId, setFlashInfo: SetFlashInfo) {
  try {
    const res = await fetch(`${API_BASE_URL}/mall-trade-service/flash/product/${id}`);
    const data = await res.json();
    if (data?.data?.inFlash) setFlashInfo(data.data);
  } catch {}
}

function addSelectedSkuToCart(args: SkuConfirmArgs) {
  args.addItem({ productId: Number(args.id), quantity: args.quantity, productSkuId: Number(args.sku?.skuId ?? args.id) } as any);
  if (args.userId) recommendationHooks.recordBehavior(args.userId, String(args.id), 'CART').catch(() => {});
  Alert.alert(args.t('common.success'), args.t('home.product.added_to_cart'));
}

function openFlashCheckout(args: SkuConfirmArgs) {
  router.push({
    pathname: '/flash/checkout',
    params: {
      sessionId: args.flashInfo.sessionId, productId: args.id,
      skuId: args.sku?.skuId ?? args.id, productName: '', productPic: '',
      quantity: args.quantity, flashPrice: args.flashInfo.flashPrice,
    },
  });
}

function openNormalCheckout(args: SkuConfirmArgs) {
  router.push({ pathname: '/checkout', params: { productId: args.id, quantity: args.quantity, skuId: args.sku?.skuId } });
}

function confirmSkuSelection(args: SkuConfirmArgs) {
  args.setSkuSelectorVisible(false);
  if (args.closeOnly) return;
  if (args.skuActionType === 'cart') return addSelectedSkuToCart(args);
  if (args.skuActionType === 'buy' && args.flashInfo) return openFlashCheckout(args);
  openNormalCheckout(args);
}

function ProductBanner({ images }: { images: string[] }) {
  const { width } = useWindowDimensions();
  return (
    <ScrollView horizontal pagingEnabled showsHorizontalScrollIndicator={false} style={[styles.bannerContainer, { height: width }]}>
      {images.map((uri, index) => (
        <Image key={index} source={{ uri }} style={[styles.bannerImage, { width, height: width }]} />
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

function ProductDetailFooter({ colors, productDetails, flashInfo, onOpenSKU, skuSelectorVisible, skuActionType, productImages, mockSpecs, mockSKUs, onSKUConfirm }: any) {
  return (
    <>
      <ActionButtons colors={colors} productDetails={productDetails} flashInfo={flashInfo} onOpenSKU={onOpenSKU} />
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

function useProductDetailState() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { width: contentWidth } = useWindowDimensions();
  const { addItem } = useCart();
  const { userId } = useAuth();
  const [skuSelectorVisible, setSkuSelectorVisible] = useState(false);
  const [skuActionType, setSkuActionType] = useState<SkuActionType>('cart');
  const [flashInfo, setFlashInfo] = useState<any>(null);
  return { colors, contentWidth, addItem, userId, skuSelectorVisible, setSkuSelectorVisible, skuActionType, setSkuActionType, flashInfo, setFlashInfo };
}

type ProductDetailState = ReturnType<typeof useProductDetailState>;

function useProductDetailEffects(id: RouteId, userId: ProductDetailState['userId'], setFlashInfo: SetFlashInfo) {
  useEffect(() => {
    if (userId && id) recommendationHooks.recordBehavior(userId, String(id), 'VIEW').catch(() => {});
  }, [userId, id]);
  useEffect(() => {
    void loadFlashInfo(id, setFlashInfo);
  }, [id]);
}

interface SkuConfirmArgs {
  t: Translate;
  id: RouteId;
  sku: SKUInfo | null;
  quantity: number;
  closeOnly: boolean;
  skuActionType: SkuActionType;
  flashInfo: any;
  userId: ProductDetailState['userId'];
  addItem: ProductDetailState['addItem'];
  setSkuSelectorVisible: SetBoolean;
}

function createProductDetailHandlers(t: Translate, id: RouteId, state: ProductDetailState) {
  return {
    handleOpenSKU: (actionType: SkuActionType) => {
      state.setSkuActionType(actionType);
      state.setSkuSelectorVisible(true);
    },
    handleSKUConfirm: (_selectedSpecs: any, sku: SKUInfo | null, quantity: number, closeOnly: boolean) =>
      confirmSkuSelection({ t, id, sku, quantity, closeOnly, ...state }),
  };
}

function useProductDetailLogic(t: Translate, id: RouteId) {
  const state = useProductDetailState();
  useProductDetailEffects(id, state.userId, state.setFlashInfo);
  return { ...state, ...createProductDetailHandlers(t, id, state) };
}

type ProductDetailLogic = ReturnType<typeof useProductDetailLogic>;

interface ProductDetailContentProps extends ProductDetailLogic {
  t: Translate;
  productDetails: any;
  isPageLoading: boolean;
}

function ProductDetailReadyView(props: ProductDetailContentProps) {
  const { colors, t, contentWidth, productDetails, flashInfo, skuSelectorVisible, skuActionType, handleOpenSKU, handleSKUConfirm } = props;
  const productImages = getProductImages(productDetails);
  const detailWithSku = productDetails as any;
  const mockSpecs: SpecGroup[] = detailWithSku?.specs || getDefaultSpecs();
  const mockSKUs: SKUInfo[] = detailWithSku?.skus || [];
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <Stack.Screen options={{ title: t('home.product.detail_title'), headerTransparent: true, headerTitle: '' }} />
      <ProductDetailBody productDetails={productDetails} colors={colors} t={t} contentWidth={contentWidth} productImages={productImages} />
      <ProductDetailFooter colors={colors} productDetails={productDetails} flashInfo={flashInfo} onOpenSKU={handleOpenSKU} skuSelectorVisible={skuSelectorVisible} skuActionType={skuActionType} productImages={productImages} mockSpecs={mockSpecs} mockSKUs={mockSKUs} onSKUConfirm={handleSKUConfirm} />
    </SafeAreaView>
  );
}

function ProductDetailContent(props: ProductDetailContentProps) {
  const { t, productDetails, isPageLoading } = props;
  if (isPageLoading || !productDetails) return <LoadingView t={t} />;
  return <ProductDetailReadyView {...props} />;
}

export default function ProductDetailScreen() {
  const { t } = useTranslation();
  const { id } = useLocalSearchParams();
  const logic = useProductDetailLogic(t, id);
  return (
    <ProductDetailContainer productId={id ? Number(id) : null}>
      {(content) => <ProductDetailContent {...logic} t={t} {...content} />}
    </ProductDetailContainer>
  );
}

const styles = StyleSheet.create({
  container: { flex:1 },
  center: { justifyContent: 'center', alignItems: 'center' },
  bannerContainer: { height: 375 },
  bannerImage: { width: 375, height: 375, resizeMode: 'cover' },
  detailDesc: { backgroundColor: '#fff', marginTop: 15, paddingBottom: 20 },
  detailHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 20 },
  headerLine: { width: 50, height: 1, backgroundColor: '#ddd', marginHorizontal: 15 },
  headerText: { fontSize: 16, fontWeight: '600' },
  htmlWrapper: { paddingHorizontal: 10 },
  footerSpacer: { height: 100 },
});
