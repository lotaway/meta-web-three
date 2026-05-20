import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { useFocusEffect } from '@react-navigation/native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { brandApi } from '@/api/generated';
import { Brand } from '@/src/generated/api/models/Brand';

const PAGE_SIZE = 10;

export default function BrandListScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  const [brands, setBrands] = useState<Brand[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);

  const fetchBrands = async (): Promise<Brand[] | null> => {
    try {
      const response = await brandApi.recommendList({
        pageNum: pageNumRef.current,
        pageSize: PAGE_SIZE,
      });
      if (response.data && Array.isArray(response.data)) {
        return response.data as Brand[];
      }
      return null;
    } catch (error) {
      return null;
    }
  };

  const loadBrands = useCallback(async (isRefresh = false) => {
    if (loading || loadingMore) return;
    if (!isRefresh && !hasMore) return;

    if (isRefresh) {
      setLoading(true);
      pageNumRef.current = 1;
    } else {
      setLoadingMore(true);
    }

    try {
      const newBrands = await fetchBrands();
      if (newBrands) {
        if (isRefresh) {
          setBrands(newBrands);
        } else {
          setBrands(prev => [...prev, ...newBrands]);
        }
        setHasMore(newBrands.length >= PAGE_SIZE);
        pageNumRef.current += 1;
      } else {
        setHasMore(false);
      }
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [loading, loadingMore, hasMore]);

  useFocusEffect(
    useCallback(() => {
      loadBrands(true);
    }, [loadBrands])
  );

  const renderBrandItem = ({ item }: { item: Brand }) => (
    <TouchableOpacity
      style={[styles.brandItem, { backgroundColor: colors.card }]}
      onPress={() => router.push({ pathname: '/brand/[id]', params: { id: item.id! } })}
      activeOpacity={0.7}
    >
      <Image
        source={{ uri: item.logo || 'https://via.placeholder.com/80' }}
        style={styles.brandLogo}
        resizeMode="contain"
      />
      <View style={styles.brandInfo}>
        <Text style={[styles.brandName, { color: colors.text }]}>{item.name}</Text>
        <Text style={[styles.productCount, { color: colors.textSecondary }]}>
          {t('brand.product_count', { count: item.productCount ?? 0 })}
        </Text>
      </View>
      <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <IconSymbol name="chevron.left" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>品牌列表</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.tint} />
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
        <Text style={[styles.headerTitle, { color: colors.text }]}>品牌列表</Text>
        <View style={styles.headerRight} />
      </View>

      {brands.length === 0 ? (
        <View style={styles.center}>
          <IconSymbol name="bag" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无品牌数据</Text>
        </View>
      ) : (
        <FlatList
          data={brands}
          keyExtractor={(item) => String(item.id!)}
          renderItem={renderBrandItem}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadBrands(false)}
          onEndReachedThreshold={0.5}
          ListFooterComponent={
            loadingMore ? (
              <View style={styles.footer}>
                <ActivityIndicator size="small" color={colors.tint} />
              </View>
            ) : !hasMore && brands.length > 0 ? (
              <View style={styles.footer}>
                <Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text>
              </View>
            ) : null
          }
        />
      )}
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
  emptyText: { marginTop: 12, fontSize: 14 },
  listContent: { padding: 12 },
  brandItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    marginBottom: 12,
    borderRadius: 12,
  },
  brandLogo: {
    width: 64,
    height: 64,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
  },
  brandInfo: {
    flex: 1,
    marginLeft: 16,
  },
  brandName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  productCount: {
    fontSize: 13,
  },
  footer: {
    padding: 16,
    alignItems: 'center',
  },
});
