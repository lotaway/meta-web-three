import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter, useFocusEffect } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { readHistoryApi, DEFAULT_USER_ID } from '@/api/generated';
import type { ProductDTO } from '@/src/generated/api/models';

const PAGE_SIZE = 10;

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type Translate = ReturnType<typeof useTranslation>['t'];
type SetProducts = React.Dispatch<React.SetStateAction<ProductDTO[]>>;
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>;
type PageRef = React.MutableRefObject<number>;

async function fetchHistoryData(): Promise<ProductDTO[] | null> {
  try {
    const response = await readHistoryApi.listHistory1({ xUserId: DEFAULT_USER_ID });
    return response.data ? response.data as ProductDTO[] : null;
  } catch (error) {
    return null;
  }
}

function updateHistoryState(newItems: ProductDTO[], isRefresh: boolean, state: BrowseHistoryState) {
  if (isRefresh) state.setProducts(newItems);
  else state.setProducts(prev => [...prev, ...newItems]);
  state.setHasMore(newItems.length >= PAGE_SIZE);
  state.pageNumRef.current += 1;
}

async function loadHistoryData(isRefresh: boolean, state: BrowseHistoryState) {
  if (state.loading || state.loadingMore) return;
  if (!isRefresh && !state.hasMore) return;
  if (isRefresh) {
    state.setLoading(true);
    state.pageNumRef.current = 1;
  } else {
    state.setLoadingMore(true);
  }
  try {
    const newItems = await fetchHistoryData();
    if (newItems) updateHistoryState(newItems, isRefresh, state);
  } catch (error) {
    console.error('Load history failed:', error);
    Alert.alert('Error', 'Failed to load the history.');
  } finally {
    state.setLoading(false);
    state.setLoadingMore(false);
  }
}

async function deleteHistoryItem(productId: number, setProducts: SetProducts) {
  try {
    await readHistoryApi.deleteCollection2({ xUserId: DEFAULT_USER_ID, productId });
    setProducts(prev => prev.filter(item => item.id !== productId));
  } catch (error) {
    console.error('Delete history failed:', error);
    Alert.alert('Error', 'Failed to delete the history item.');
  }
}

function handleClearHistory() {
  // TODO: 需要后端 API 支持清除浏览历史
  console.log('Clear history not implemented');
  /* try {
    await readHistoryApi.clearHistory({ xUserId: DEFAULT_USER_ID });
    setProducts([]);
  } catch (error) {
    console.error('Clear history failed:', error);
    Alert.alert('Error', 'Failed to clear the history.');
  } */
}

function useBrowseHistoryState() {
  const [products, setProducts] = useState<ProductDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);
  return { products, setProducts, loading, setLoading, loadingMore, setLoadingMore, hasMore, setHasMore, pageNumRef };
}

type BrowseHistoryState = ReturnType<typeof useBrowseHistoryState>;

function createHistoryActions(state: BrowseHistoryState, router: AppRouter) {
  return {
    handleDelete: (productId: number) => deleteHistoryItem(productId, state.setProducts),
    handleClear: handleClearHistory,
    handleProductPress: (productId: number) => router.push({ pathname: '/product/[id]', params: { id: productId } }),
    handleBack: () => router.back(),
  };
}

type HistoryActions = ReturnType<typeof createHistoryActions>;

function useBrowseHistoryController() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const state = useBrowseHistoryState();
  const loadHistory = useCallback(async (isRefresh = false) => {
    await loadHistoryData(isRefresh, state);
  }, [state.loading, state.loadingMore, state.hasMore]);
  useFocusEffect(
    useCallback(() => {
      loadHistory(true);
    }, [loadHistory])
  );
  return { t, router, colors, ...state, loadHistory, ...createHistoryActions(state, router) };
}

interface HistoryViewProps extends BrowseHistoryState, HistoryActions {
  t: Translate;
  router: AppRouter;
  colors: AppColors;
  loadHistory: (isRefresh?: boolean) => Promise<void>;
}

function HistoryHeader({ colors, handleBack, handleClear, showClear }: Pick<HistoryViewProps, 'colors' | 'handleBack' | 'handleClear'> & { showClear: boolean }) {
  return (
    <View style={[styles.header, { borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>浏览历史</Text>
      {showClear ? (
        <TouchableOpacity style={styles.clearBtn} onPress={handleClear}>
          <Text style={{ color: colors.tint }}>清空</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.headerRight} />
      )}
    </View>
  );
}

function LoadingView(props: HistoryViewProps) {
  const { colors } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <HistoryHeader {...props} showClear={false} />
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function EmptyHistory({ colors }: Pick<HistoryViewProps, 'colors'>) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="clock" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无浏览历史</Text>
    </View>
  );
}

function ProductItem({ item, colors, handleProductPress, handleDelete }: Pick<HistoryViewProps, 'colors' | 'handleProductPress' | 'handleDelete'> & { item: ProductDTO }) {
  return (
    <TouchableOpacity style={[styles.productItem, { backgroundColor: colors.card }]} onPress={() => handleProductPress(item.id!)}>
      <Image source={{ uri: item.imageUrl || 'https://via.placeholder.com/90' }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>{item.name}</Text>
        <View style={styles.priceRow}><Text style={[styles.price, { color: colors.tint }]}>¥{item.price}</Text></View>
      </View>
      <TouchableOpacity style={styles.deleteBtn} onPress={() => handleDelete(item.id!)}>
        <IconSymbol name="xmark.circle.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </TouchableOpacity>
  );
}

function ListFooter({ loadingMore, hasMore, products, colors, t }: Pick<HistoryViewProps, 'loadingMore' | 'hasMore' | 'products' | 'colors' | 't'>) {
  if (loadingMore) return <View style={styles.footer}><ActivityIndicator size="small" color={colors.tint} /></View>;
  if (!hasMore && products.length > 0) return <View style={styles.footer}><Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text></View>;
  return null;
}

function HistoryView(props: HistoryViewProps) {
  const { colors, products, loadHistory } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <HistoryHeader {...props} showClear />
      {products.length === 0 ? <EmptyHistory colors={colors} /> : (
        <FlatList
          data={products} keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => <ProductItem item={item} colors={colors} handleProductPress={props.handleProductPress} handleDelete={props.handleDelete} />}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadHistory(false)} onEndReachedThreshold={0.5}
          ListFooterComponent={<ListFooter {...props} />}
        />
      )}
    </SafeAreaView>
  );
}

export default function BrowseHistoryScreen() {
  const controller = useBrowseHistoryController();
  return controller.loading ? <LoadingView {...controller} /> : <HistoryView {...controller} />;
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
  clearBtn: { padding: 8 },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: { marginTop: 12, fontSize: 14 },
  listContent: { padding: 12 },
  productItem: {
    flexDirection: 'row',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    alignItems: 'center',
  },
  productImage: {
    width: 90,
    height: 90,
    borderRadius: 8,
  },
  productInfo: {
    flex: 1,
    marginLeft: 12,
  },
  productName: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 4,
  },
  subTitle: {
    fontSize: 12,
    marginBottom: 8,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
  },
  price: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  deleteBtn: { padding: 8 },
  footer: {
    padding: 16,
    alignItems: 'center',
  },
});
