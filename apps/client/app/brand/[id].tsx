import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  useWindowDimensions,
  Alert,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { brandApi } from '@/api/generated';
import { Brand } from '@/src/generated/api/models/Brand';
import { ProductDTO } from '@/src/generated/api/models/ProductDTO';

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type Translate = ReturnType<typeof useTranslation>['t'];

async function loadBrandDetailData(brandId: number): Promise<{ brand: Brand | null; products: ProductDTO[] | null }> {
  const [detailRes, productsRes] = await Promise.all([
    brandApi.details({ id: brandId }),
    brandApi.listProducts1({ id: brandId }),
  ]);
  return {
    brand: detailRes.data ? detailRes.data as Brand : null,
    products: productsRes.data && Array.isArray(productsRes.data) ? productsRes.data as ProductDTO[] : null,
  };
}

async function loadBrandDetailFlow(brandId: number, state: BrandDetailState) {
  state.setLoading(true);
  try {
    const data = await loadBrandDetailData(brandId);
    if (data.brand) state.setBrand(data.brand);
    if (data.products) state.setProducts(data.products);
  } catch (error) {
    console.error('Failed to load brand detail:', error);
    Alert.alert('Error', 'Failed to load the brand detail.');
  } finally {
    state.setLoading(false);
  }
}

function useBrandDetailState() {
  const [brand, setBrand] = useState<Brand | null>(null);
  const [products, setProducts] = useState<ProductDTO[]>([]);
  const [loading, setLoading] = useState(true);
  return { brand, setBrand, products, setProducts, loading, setLoading };
}

type BrandDetailState = ReturnType<typeof useBrandDetailState>;

function useBrandDetailController() {
  const { t } = useTranslation();
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const state = useBrandDetailState();
  useEffect(() => {
    if (!id) return;
    loadBrandDetailFlow(Number(id), state);
  }, [id]);
  return {
    t, router, colors, ...state,
    handleBack: () => router.back(),
    navigateToProduct: (productId: number) => router.push({ pathname: '/product/[id]', params: { id: productId } }),
  };
}

interface BrandDetailViewProps extends BrandDetailState {
  t: Translate;
  colors: AppColors;
  brand: Brand;
  handleBack: () => void;
  navigateToProduct: (productId: number) => void;
}

function DetailHeader({ colors, handleBack, title }: { colors: AppColors; handleBack: () => void; title?: string }) {
  return (
    <View style={[styles.header, { borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{title}</Text>
      <View style={styles.headerRight} />
    </View>
  );
}

function LoadingView({ colors, handleBack }: { colors: AppColors; handleBack: () => void }) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>品牌详情</Text>
        <View style={styles.headerRight} />
      </View>
      <View style={styles.center}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function BrandNotFound({ colors, handleBack }: { colors: AppColors; handleBack: () => void }) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>品牌详情</Text>
        <View style={styles.headerRight} />
      </View>
      <View style={styles.center}>
        <Text style={{ color: colors.textSecondary }}>品牌不存在</Text>
      </View>
    </SafeAreaView>
  );
}

function BrandBanner({ brand, colors }: { brand: Brand; colors: AppColors }) {
  const { width } = useWindowDimensions();
  if (!brand.areaPicture) return null;
  return <Image source={{ uri: brand.areaPicture }} style={[styles.bannerImage, { width }]} resizeMode="cover" />;
}

function BrandInfoCard({ brand, colors, t }: { brand: Brand; colors: AppColors; t: Translate }) {
  return (
    <View style={[styles.brandInfo, { backgroundColor: colors.card }]}>
      <Image source={{ uri: brand.logo || 'https://via.placeholder.com/80' }} style={styles.brandLogo} resizeMode="contain" />
      <View style={styles.brandMeta}>
        <Text style={[styles.brandName, { color: colors.text }]}>{brand.name}</Text>
        {brand.firstLetter && (
          <Text style={[styles.brandLetter, { color: colors.textSecondary }]}>
            {t('brand.first_letter', { letter: brand.firstLetter })}
          </Text>
        )}
        <Text style={[styles.productCount, { color: colors.textSecondary }]}>
          {t('brand.product_count', { count: brand.productCount ?? 0 })}
        </Text>
      </View>
    </View>
  );
}

function BrandStorySection({ brand, colors }: { brand: Brand; colors: AppColors }) {
  if (!brand.story) return null;
  return (
    <>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>品牌故事</Text>
      <View style={[styles.storyCard, { backgroundColor: colors.card }]}>
        <Text style={[styles.storyText, { color: colors.textSecondary }]}>{brand.story}</Text>
      </View>
    </>
  );
}

function BrandProductCard({ product, colors, navigateToProduct }: { product: ProductDTO; colors: AppColors; navigateToProduct: (id: number) => void }) {
  const { width } = useWindowDimensions();
  const cardWidth = (width - 36) / 2;
  return (
    <TouchableOpacity style={[styles.productCard, { width: cardWidth, backgroundColor: colors.card }]} onPress={() => navigateToProduct(product.id!)} activeOpacity={0.7}>
      <Image source={{ uri: product.imageUrl || 'https://via.placeholder.com/200' }} style={[styles.productImage, { width: cardWidth, height: cardWidth }]} resizeMode="cover" />
      <Text style={[styles.productName, { color: colors.text }]} numberOfLines={2}>{product.name}</Text>
      <Text style={styles.productPrice}>¥{product.price}</Text>
    </TouchableOpacity>
  );
}

function RelatedProductsSection({ products, colors, navigateToProduct }: { products: ProductDTO[]; colors: AppColors; navigateToProduct: (id: number) => void }) {
  if (products.length === 0) return null;
  return (
    <>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>相关商品</Text>
      <View style={styles.productGrid}>
        {products.map((product) => <BrandProductCard key={product.id} product={product} colors={colors} navigateToProduct={navigateToProduct} />)}
      </View>
    </>
  );
}

function BrandDetailView(props: BrandDetailViewProps) {
  const { colors, brand, products, t, handleBack, navigateToProduct } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <DetailHeader colors={colors} handleBack={handleBack} title={brand.name} />
      <ScrollView showsVerticalScrollIndicator={false}>
        <BrandBanner brand={brand} colors={colors} />
        <BrandInfoCard brand={brand} colors={colors} t={t} />
        <BrandStorySection brand={brand} colors={colors} />
        <RelatedProductsSection products={products} colors={colors} navigateToProduct={navigateToProduct} />
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

export default function BrandDetailScreen() {
  const controller = useBrandDetailController();
  if (controller.loading) return <LoadingView colors={controller.colors} handleBack={controller.handleBack} />;
  if (!controller.brand) return <BrandNotFound colors={controller.colors} handleBack={controller.handleBack} />;
  return <BrandDetailView {...controller} brand={controller.brand} />;
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  backBtn: { padding: 8 },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginRight: 40,
  },
  headerRight: { width: 40 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  bannerImage: {
    width: 375,
    height: 200,
  },
  brandInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    margin: 12,
    borderRadius: 12,
  },
  brandLogo: {
    width: 72,
    height: 72,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
  },
  brandMeta: {
    flex: 1,
    marginLeft: 16,
  },
  brandName: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  brandLetter: {
    fontSize: 13,
    marginBottom: 2,
  },
  productCount: {
    fontSize: 13,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
  },
  storyCard: {
    marginHorizontal: 12,
    padding: 16,
    borderRadius: 12,
  },
  storyText: {
    fontSize: 14,
    lineHeight: 22,
  },
  productGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 8,
  },
  productCard: {
    width: 169.5,
    marginHorizontal: 6,
    marginBottom: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  productImage: {
    width: 169.5,
    height: 169.5,
  },
  productName: {
    fontSize: 13,
    padding: 8,
    lineHeight: 18,
  },
  productPrice: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FF3B30',
    paddingHorizontal: 8,
    paddingBottom: 8,
  },
  bottomSpacer: {
    height: 40,
  },
});
