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

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type Translate = ReturnType<typeof useTranslation>['t'];
type SetBrands = React.Dispatch<React.SetStateAction<Brand[]>>;
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>;
type PageRef = React.MutableRefObject<number>;

async function fetchBrandsData(pageNum: number): Promise<Brand[] | null> {
  try {
    const response = await brandApi.recommendList({ pageNum, pageSize: PAGE_SIZE });
    if (response.data && Array.isArray(response.data)) return response.data as Brand[];
    return null;
  } catch (error) {
    return null;
  }
}

function applyBrandsPage(newBrands: Brand[] | null, isRefresh: boolean, state: BrandListState) {
  if (!newBrands) {
    state.setHasMore(false);
    return;
  }
  if (isRefresh) state.setBrands(newBrands);
  else state.setBrands(prev => [...prev, ...newBrands]);
  state.setHasMore(newBrands.length >= PAGE_SIZE);
  state.pageNumRef.current += 1;
}

async function loadBrandsData(isRefresh: boolean, state: BrandListState) {
  if (state.loading || state.loadingMore) return;
  if (!isRefresh && !state.hasMore) return;
  if (isRefresh) {
    state.setLoading(true);
    state.pageNumRef.current = 1;
  } else {
    state.setLoadingMore(true);
  }
  try {
    const newBrands = await fetchBrandsData(state.pageNumRef.current);
    applyBrandsPage(newBrands, isRefresh, state);
  } finally {
    state.setLoading(false);
    state.setLoadingMore(false);
  }
}

function useBrandListState() {
  const [brands, setBrands] = useState<Brand[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);
  return { brands, setBrands, loading, setLoading, loadingMore, setLoadingMore, hasMore, setHasMore, pageNumRef };
}

type BrandListState = ReturnType<typeof useBrandListState>;

function createBrandListActions(state: BrandListState, router: AppRouter) {
  return {
    handleBrandPress: (id: number) => router.push({ pathname: '/brand/[id]', params: { id } }),
    handleBack: () => router.back(),
  };
}

type BrandListActions = ReturnType<typeof createBrandListActions>;

function useBrandListController() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const state = useBrandListState();
  const loadBrands = useCallback(async (isRefresh = false) => {
    await loadBrandsData(isRefresh, state);
  }, [state.loading, state.loadingMore, state.hasMore]);
  useFocusEffect(
    useCallback(() => {
      loadBrands(true);
    }, [loadBrands])
  );
  return { t, router, colors, ...state, loadBrands, ...createBrandListActions(state, router) };
}

interface BrandListViewProps extends BrandListState, BrandListActions {
  t: Translate;
  router: AppRouter;
  colors: AppColors;
  loadBrands: (isRefresh?: boolean) => Promise<void>;
}

function BrandHeader({ colors, handleBack, withBorder }: Pick<BrandListViewProps, 'colors' | 'handleBack'> & { withBorder?: boolean }) {
  return (
    <View style={withBorder ? [styles.header, { borderBottomColor: colors.border }] : styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>品牌列表</Text>
      <View style={styles.headerRight} />
    </View>
  );
}

function LoadingView(props: BrandListViewProps) {
  const { colors } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <BrandHeader {...props} />
      <View style={styles.center}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function EmptyBrands({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.center}>
      <IconSymbol name="bag" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无品牌数据</Text>
    </View>
  );
}

function BrandItemRow({ item, colors, t, handleBrandPress }: Pick<BrandListViewProps, 'colors' | 't' | 'handleBrandPress'> & { item: Brand }) {
  return (
    <TouchableOpacity style={[styles.brandItem, { backgroundColor: colors.card }]} onPress={() => handleBrandPress(item.id!)} activeOpacity={0.7}>
      <Image source={{ uri: item.logo || 'https://via.placeholder.com/80' }} style={styles.brandLogo} resizeMode="contain" />
      <View style={styles.brandInfo}>
        <Text style={[styles.brandName, { color: colors.text }]}>{item.name}</Text>
        <Text style={[styles.productCount, { color: colors.textSecondary }]}>
          {t('brand.product_count', { count: item.productCount ?? 0 })}
        </Text>
      </View>
      <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
    </TouchableOpacity>
  );
}

function ListFooter({ loadingMore, hasMore, brands, colors, t }: Pick<BrandListViewProps, 'loadingMore' | 'hasMore' | 'brands' | 'colors' | 't'>) {
  if (loadingMore) return <View style={styles.footer}><ActivityIndicator size="small" color={colors.tint} /></View>;
  if (!hasMore && brands.length > 0) return <View style={styles.footer}><Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text></View>;
  return null;
}

function BrandListView(props: BrandListViewProps) {
  const { colors, brands, loadBrands, t, handleBrandPress } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <BrandHeader {...props} withBorder />
      {brands.length === 0 ? <EmptyBrands colors={colors} /> : (
        <FlatList
          data={brands} keyExtractor={(item) => String(item.id!)}
          renderItem={({ item }) => <BrandItemRow item={item} colors={colors} t={t} handleBrandPress={handleBrandPress} />}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadBrands(false)} onEndReachedThreshold={0.5}
          ListFooterComponent={<ListFooter {...props} />}
        />
      )}
    </SafeAreaView>
  );
}

export default function BrandListScreen() {
  const controller = useBrandListController();
  return controller.loading ? <LoadingView {...controller} /> : <BrandListView {...controller} />;
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
