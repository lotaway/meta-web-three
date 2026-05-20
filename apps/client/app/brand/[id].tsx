import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Dimensions,
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

const { width: WINDOW_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = (WINDOW_WIDTH - 36) / 2;

export default function BrandDetailScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  const [brand, setBrand] = useState<Brand | null>(null);
  const [products, setProducts] = useState<ProductDTO[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    loadBrandDetail(Number(id));
  }, [id]);

  const loadBrandDetail = async (brandId: number) => {
    setLoading(true);
    try {
      const [detailRes, productsRes] = await Promise.all([
        brandApi.details({ id: brandId }),
        brandApi.listProducts1({ id: brandId }),
      ]);
      if (detailRes.data) setBrand(detailRes.data as Brand);
      if (productsRes.data && Array.isArray(productsRes.data)) {
        setProducts(productsRes.data as ProductDTO[]);
      }
    } catch (error) {
      console.error('Failed to load brand detail:', error);
    } finally {
      setLoading(false);
    }
  };

  const navigateToProduct = (productId: number) => {
    router.push({ pathname: '/product/[id]', params: { id: productId } });
  };

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
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

  if (!brand) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
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

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>{brand.name}</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {brand.areaPicture && (
          <Image
            source={{ uri: brand.areaPicture }}
            style={styles.bannerImage}
            resizeMode="cover"
          />
        )}

        <View style={[styles.brandInfo, { backgroundColor: colors.card }]}>
          <Image
            source={{ uri: brand.logo || 'https://via.placeholder.com/80' }}
            style={styles.brandLogo}
            resizeMode="contain"
          />
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

        {brand.story && (
          <>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>品牌故事</Text>
            <View style={[styles.storyCard, { backgroundColor: colors.card }]}>
              <Text style={[styles.storyText, { color: colors.textSecondary }]}>
                {brand.story}
              </Text>
            </View>
          </>
        )}

        {products.length > 0 && (
          <>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>相关商品</Text>
            <View style={styles.productGrid}>
              {products.map((product) => (
                <TouchableOpacity
                  key={product.id}
                  style={[styles.productCard, { backgroundColor: colors.card }]}
                  onPress={() => navigateToProduct(product.id!)}
                  activeOpacity={0.7}
                >
                  <Image
                    source={{ uri: product.imageUrl || 'https://via.placeholder.com/200' }}
                    style={styles.productImage}
                    resizeMode="cover"
                  />
                  <Text style={[styles.productName, { color: colors.text }]} numberOfLines={2}>
                    {product.name}
                  </Text>
                  <Text style={styles.productPrice}>
                    ¥{product.price}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </>
        )}
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
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
    width: WINDOW_WIDTH,
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
    width: CARD_WIDTH,
    marginHorizontal: 6,
    marginBottom: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  productImage: {
    width: CARD_WIDTH,
    height: CARD_WIDTH,
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
