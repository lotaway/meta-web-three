import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Dimensions,
  Alert,
} from 'react-native';
import { useLocalSearchParams, router } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { productApi, categoryApi } from '@/api/generated';
import type { ProductDTO } from '@/src/generated/api/models';

const { width: WINDOW_WIDTH } = Dimensions.get('window');
const PRODUCT_WIDTH = (WINDOW_WIDTH - 24 - 12) / 2;

type SortType = 'comprehensive' | 'sales' | 'price-asc' | 'price-desc';
type RouteId = string | string[] | undefined;
type AppColors = typeof Colors.light;
type AppRouter = typeof router;
type Translate = ReturnType<typeof useTranslation>['t'];
type SetProducts = React.Dispatch<React.SetStateAction<ProductDTO[]>>;
type SetCateList = React.Dispatch<React.SetStateAction<any[]>>;
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>;
type SetSortType = React.Dispatch<React.SetStateAction<SortType>>;
type SetNumberOrNull = React.Dispatch<React.SetStateAction<number | null>>;
type PageRef = React.MutableRefObject<number>;

const SORT_PARAM_BY_TYPE: Record<SortType, number> = {
  comprehensive: 0,
  sales: 2,
  'price-asc': 3,
  'price-desc': 4,
};

function getSortParam(sortType: SortType) {
  return SORT_PARAM_BY_TYPE[sortType];
}

async function loadCategoryListData(id: RouteId, setCateList: SetCateList) {
  try {
    const parentId = id ? Math.floor(Number(id) / 100) * 100 : 0;
    const response = await categoryApi.viewChildren({ parentId });
    if (response.data) setCateList(response.data as any[]);
  } catch (error) {
    console.error('Load category list failed:', error);
    Alert.alert('Error', 'Failed to load the category list.');
  }
}

async function fetchProductsData(selectedCateId: number | null) {
  const response = await productApi.listProducts({
    categoryId: selectedCateId || undefined,
  });
  return response.data as ProductDTO[] | null;
}

function updateProductListState(newProducts: ProductDTO[], isRefresh: boolean, setProducts: SetProducts, setHasMore: SetBoolean, pageNumRef: PageRef, pageSize: number) {
  if (isRefresh) setProducts(newProducts);
  else setProducts(prev => [...prev, ...newProducts]);
  setHasMore(newProducts.length >= pageSize);
  pageNumRef.current += 1;
}

interface LoadProductDeps {
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  setLoading: SetBoolean;
  setLoadingMore: SetBoolean;
  pageNumRef: PageRef;
  pageSize: number;
  fetchProducts: () => Promise<ProductDTO[] | null>;
  updateProductState: (products: ProductDTO[], refresh: boolean) => void;
}

async function loadProductList(isRefresh: boolean, deps: LoadProductDeps) {
  if (deps.loading || deps.loadingMore) return;
  if (isRefresh) {
    deps.setLoading(true);
    deps.pageNumRef.current = 1;
  } else {
    if (!deps.hasMore) return;
    deps.setLoadingMore(true);
  }
  try {
    const newProducts = await deps.fetchProducts();
    if (newProducts) deps.updateProductState(newProducts, isRefresh);
  } catch (error) {
    console.error('Load products failed:', error);
    Alert.alert('Error', 'Failed to load products.');
  } finally {
    deps.setLoading(false);
    deps.setLoadingMore(false);
  }
}

function resolveNextSortType(type: SortType, current: SortType) {
  if (type !== 'price-asc' && type !== 'price-desc') return type;
  if (current === 'price-asc') return 'price-desc';
  if (current === 'price-desc') return 'price-asc';
  return type;
}

function openProduct(productId: number, appRouter: AppRouter) {
  appRouter.push({ pathname: '/product/[id]', params: { id: productId } });
}

function useCategoryListState() {
  const { id } = useLocalSearchParams();
  const [products, setProducts] = useState<ProductDTO[]>([]);
  const [cateList, setCateList] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [sortType, setSortType] = useState<SortType>('comprehensive');
  const [priceOrder, setPriceOrder] = useState<'asc' | 'desc'>('asc');
  const [cateMaskVisible, setCateMaskVisible] = useState(false);
  const [selectedCateId, setSelectedCateId] = useState<number | null>(id ? Number(id) : null);
  const pageNumRef = useRef(1);
  const pageSize = 10;
  return { id, products, setProducts, cateList, setCateList, loading, setLoading, loadingMore, setLoadingMore, hasMore, setHasMore, sortType, setSortType, priceOrder, setPriceOrder, cateMaskVisible, setCateMaskVisible, selectedCateId, setSelectedCateId, pageNumRef, pageSize };
}

type CategoryState = ReturnType<typeof useCategoryListState>;

function useCategoryListActions(state: CategoryState) {
  const loadCateList = useCallback(async () => {
    await loadCategoryListData(state.id, state.setCateList);
  }, [state.id]);
  const fetchProducts = useCallback(async () => fetchProductsData(state.selectedCateId), [state.selectedCateId, state.sortType]);
  const updateProductState = useCallback((products: ProductDTO[], refresh: boolean) => {
    updateProductListState(products, refresh, state.setProducts, state.setHasMore, state.pageNumRef, state.pageSize);
  }, []);
  const loadProducts = useCallback(async (isRefresh = false) => {
    await loadProductList(isRefresh, {
      loading: state.loading, loadingMore: state.loadingMore, hasMore: state.hasMore,
      setLoading: state.setLoading, setLoadingMore: state.setLoadingMore,
      pageNumRef: state.pageNumRef, pageSize: state.pageSize, fetchProducts, updateProductState,
    });
  }, [state.selectedCateId, state.sortType, state.loading, state.loadingMore, state.hasMore, fetchProducts, updateProductState]);
  React.useEffect(() => {
    loadProducts(true);
  }, []);
  return { loadCateList, loadProducts };
}

type CategoryActions = ReturnType<typeof useCategoryListActions>;

function createCategoryHandlers(state: CategoryState, actions: CategoryActions) {
  return {
    handleCateChange: (cateId: number) => {
      state.setSelectedCateId(cateId);
      state.setCateMaskVisible(false);
      state.setProducts([]);
      state.pageNumRef.current = 1;
      actions.loadProducts(true);
    },
    handleSortChange: (type: SortType) => {
      state.setSortType(resolveNextSortType(type, state.sortType));
      state.setProducts([]);
      state.pageNumRef.current = 1;
      actions.loadProducts(true);
    },
    handleProductPress: (productId: number) => openProduct(productId, router),
  };
}

type CategoryHandlers = ReturnType<typeof createCategoryHandlers>;

function useCategoryListController() {
  const { t } = useTranslation();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const state = useCategoryListState();
  const actions = useCategoryListActions(state);
  return { t, colors, ...state, ...actions, ...createCategoryHandlers(state, actions) };
}

interface CategoryListViewProps extends CategoryState, CategoryActions, CategoryHandlers {
  t: Translate;
  colors: AppColors;
}

function ProductItem({ item, colors, handleProductPress }: Pick<CategoryListViewProps, 'colors' | 'handleProductPress'> & { item: ProductDTO }) {
  return (
    <TouchableOpacity
      style={[styles.productItem, { backgroundColor: colors.card }]}
      onPress={() => handleProductPress(item.id!)}
    >
      <Image source={{ uri: item.imageUrl || 'https://via.placeholder.com/150' }} style={styles.productImage} resizeMode="cover" />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>{item.name}</Text>
        <View style={styles.priceRow}>
          <Text style={[styles.price, { color: colors.tint }]}>¥{item.price}</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

function SortOption({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.sortItem, active && styles.sortItemActive]}
      onPress={onPress}
    >
      <Text style={[styles.sortText, active && styles.sortTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

function PriceSortOption({ colors, sortType, handleSortChange, t }: Pick<CategoryListViewProps, 'colors' | 'sortType' | 'handleSortChange' | 't'>) {
  const priceActive = sortType === 'price-asc' || sortType === 'price-desc';
  return (
    <TouchableOpacity
      style={[styles.sortItem, priceActive && styles.sortItemActive]}
      onPress={() => handleSortChange('price-asc')}
    >
      <Text style={[styles.sortText, priceActive && styles.sortTextActive]}>
        {t('search.sort.price')}
      </Text>
      <View style={styles.priceOrderIcons}>
        <IconSymbol name="chevron.up" size={12} color={sortType === 'price-asc' ? colors.tint : colors.textSecondary} />
        <IconSymbol name="chevron.down" size={12} color={sortType === 'price-desc' ? colors.tint : colors.textSecondary} />
      </View>
    </TouchableOpacity>
  );
}

function SortBar({ colors, sortType, t, handleSortChange, setCateMaskVisible }: Pick<CategoryListViewProps, 'colors' | 'sortType' | 't' | 'handleSortChange' | 'setCateMaskVisible'>) {
  return (
    <View style={[styles.sortBar, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <SortOption label={t('search.sort.comprehensive')} active={sortType === 'comprehensive'} onPress={() => handleSortChange('comprehensive')} />
      <SortOption label={t('search.sort.sales')} active={sortType === 'sales'} onPress={() => handleSortChange('sales')} />
      <PriceSortOption colors={colors} sortType={sortType} handleSortChange={handleSortChange} t={t} />
      <TouchableOpacity style={styles.cateButton} onPress={() => setCateMaskVisible(true)}>
        <IconSymbol name="line.3.horizontal.decrease.circle" size={20} color={colors.text} />
      </TouchableOpacity>
    </View>
  );
}

function CateItem({ item, selectedCateId, colors, handleCateChange }: Pick<CategoryListViewProps, 'selectedCateId' | 'colors' | 'handleCateChange'> & { item: any }) {
  return (
    <TouchableOpacity
      style={[styles.cateItem, selectedCateId === item.id && styles.cateItemSelected]}
      onPress={() => handleCateChange(item.id)}
    >
      <Text style={[styles.cateItemText, { color: selectedCateId === item.id ? colors.tint : colors.text }]}>
        {item.name}
      </Text>
    </TouchableOpacity>
  );
}

function CateMask({ cateMaskVisible, cateList, colors, selectedCateId, handleCateChange, setCateMaskVisible }: Pick<CategoryListViewProps, 'cateMaskVisible' | 'cateList' | 'colors' | 'selectedCateId' | 'handleCateChange' | 'setCateMaskVisible'>) {
  return (
    <View
      pointerEvents={cateMaskVisible ? 'auto' : 'none'}
      style={[styles.cateMask, { backgroundColor: 'rgba(0,0,0,0.4)' }, cateMaskVisible ? styles.cateMaskVisible : styles.cateMaskHidden]}
    >
      <TouchableOpacity style={styles.cateMaskOverlay} onPress={() => setCateMaskVisible(false)} />
      <View style={[styles.catePanel, { backgroundColor: colors.background }]}>
        <FlatList
          data={cateList}
          keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => <CateItem item={item} selectedCateId={selectedCateId} colors={colors} handleCateChange={handleCateChange} />}
        />
      </View>
    </View>
  );
}

function ListFooter({ loadingMore, hasMore, products, colors, t }: Pick<CategoryListViewProps, 'loadingMore' | 'hasMore' | 'products' | 'colors' | 't'>) {
  if (loadingMore) return <View style={styles.footer}><ActivityIndicator size="small" color={colors.tint} /></View>;
  if (!hasMore && products.length > 0) return <View style={styles.footer}><Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text></View>;
  return null;
}

function EmptyView({ loading, products, colors, t }: Pick<CategoryListViewProps, 'loading' | 'products' | 'colors' | 't'>) {
  if (loading || products.length > 0) return null;
  return <View style={styles.empty}><Text style={{ color: colors.textSecondary }}>{t('common.no_data')}</Text></View>;
}

function LoadingView({ colors }: Pick<CategoryListViewProps, 'colors'>) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ActivityIndicator size="large" color={colors.tint} />
    </SafeAreaView>
  );
}

function CategoryListView(props: CategoryListViewProps) {
  const { colors, products, loadProducts, handleProductPress } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <SortBar {...props} />
      <FlatList
        data={products} keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => <ProductItem item={item} colors={colors} handleProductPress={handleProductPress} />}
        numColumns={2} columnWrapperStyle={styles.productRow} contentContainerStyle={styles.productList}
        onEndReached={() => loadProducts(false)} onEndReachedThreshold={0.5}
        ListFooterComponent={<ListFooter {...props} />}
        ListEmptyComponent={<EmptyView {...props} />}
      />
      <CateMask {...props} />
    </SafeAreaView>
  );
}

export default function CategoryListScreen() {
  const controller = useCategoryListController();
  return controller.loading ? <LoadingView {...controller} /> : <CategoryListView {...controller} />;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  sortBar: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 44,
    borderBottomWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: 12,
  },
  sortItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
  },
  sortItemActive: {
    // 激活状态样式
  },
  sortText: {
    fontSize: 14,
    color: '#666',
  },
  sortTextActive: {
    color: '#007AFF',
    fontWeight: '600',
  },
  priceOrderIcons: {
    marginLeft: 4,
  },
  cateButton: {
    padding: 8,
  },
  productList: {
    padding: 12,
  },
  productRow: {
    justifyContent: 'space-between',
  },
  productItem: {
    width: PRODUCT_WIDTH,
    marginBottom: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  productImage: {
    width: '100%',
    height: PRODUCT_WIDTH,
  },
  productInfo: {
    padding: 8,
  },
  productName: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 4,
  },
  productSubtitle: {
    fontSize: 12,
    marginBottom: 8,
  },
  priceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  price: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  sales: {
    fontSize: 12,
  },
  cateMask: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 100,
  },
  cateMaskVisible: {
    display: 'flex',
  },
  cateMaskHidden: {
    display: 'none',
  },
  cateMaskOverlay: {
    flex: 1,
  },
  catePanel: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    width: 200,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: -2, height: 0 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 5,
  },
  cateItem: {
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#eee',
  },
  cateItemSelected: {
    backgroundColor: '#f0f0f0',
  },
  cateItemText: {
    fontSize: 14,
  },
  footer: {
    padding: 16,
    alignItems: 'center',
  },
  empty: {
    padding: 40,
    alignItems: 'center',
  },
});
