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
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { useFocusEffect } from '@react-navigation/native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { productCollectionApi, DEFAULT_USER_ID } from '@/api/generated';
import type { ProductDTO } from '@/src/generated/api/models';

const PAGE_SIZE = 10;

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type Translate = ReturnType<typeof useTranslation>['t'];
type SetFavorites = React.Dispatch<React.SetStateAction<ProductDTO[]>>;
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>;
type PageRef = React.MutableRefObject<number>;

function applyFavoritesPage(newItems: ProductDTO[], isRefresh: boolean, state: FavoritesState) {
  if (isRefresh) state.setFavorites(newItems);
  else state.setFavorites(prev => [...prev, ...newItems]);
  state.setHasMore(newItems.length >= PAGE_SIZE);
  state.pageNumRef.current += 1;
}

async function loadFavoritesData(isRefresh: boolean, state: FavoritesState) {
  if (state.loading || state.loadingMore) return;
  if (isRefresh) {
    state.setLoading(true);
    state.pageNumRef.current = 1;
  } else {
    if (!state.hasMore) return;
    state.setLoadingMore(true);
  }
  try {
    const response = await productCollectionApi.listCollection1({ xUserId: DEFAULT_USER_ID });
    if (response.data) applyFavoritesPage(response.data as ProductDTO[], isRefresh, state);
  } catch (error) {
    console.error('Load favorites failed:', error);
    Alert.alert('Error', 'Failed to load favorites.');
  } finally {
    state.setLoading(false);
    state.setLoadingMore(false);
  }
}

async function removeFavorite(productId: number, setFavorites: SetFavorites) {
  try {
    await productCollectionApi.deleteCollection2({ xUserId: DEFAULT_USER_ID, productId });
    setFavorites(prev => prev.filter(item => item.id !== productId));
  } catch (error) {
    console.error('Remove favorite failed:', error);
    Alert.alert('Error', 'Failed to remove the favorite.');
  }
}

function useFavoritesState() {
  const [favorites, setFavorites] = useState<ProductDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);
  return { favorites, setFavorites, loading, setLoading, loadingMore, setLoadingMore, hasMore, setHasMore, pageNumRef };
}

type FavoritesState = ReturnType<typeof useFavoritesState>;

function createFavoritesActions(state: FavoritesState, router: AppRouter) {
  return {
    handleRemove: (productId: number) => removeFavorite(productId, state.setFavorites),
    handleProductPress: (id: number) => router.push({ pathname: '/product/[id]', params: { id } }),
    handleBack: () => router.back(),
    handleGoShopping: () => router.replace('/(tabs)'),
  };
}

type FavoritesActions = ReturnType<typeof createFavoritesActions>;

function useFavoritesController() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const state = useFavoritesState();
  const loadFavorites = useCallback(async (isRefresh = false) => {
    await loadFavoritesData(isRefresh, state);
  }, [state.loading, state.loadingMore, state.hasMore]);
  useFocusEffect(
    useCallback(() => {
      loadFavorites(true);
    }, [loadFavorites])
  );
  return { t, router, colors, ...state, loadFavorites, ...createFavoritesActions(state, router) };
}

interface FavoritesViewProps extends FavoritesState, FavoritesActions {
  t: Translate;
  router: AppRouter;
  colors: AppColors;
  loadFavorites: (isRefresh?: boolean) => Promise<void>;
}

function FavoritesHeader({ colors, t, handleBack, withBorder }: Pick<FavoritesViewProps, 'colors' | 't' | 'handleBack'> & { withBorder?: boolean }) {
  return (
    <View style={withBorder ? [styles.header, { borderBottomColor: colors.border }] : styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{t('favorites.title')}</Text>
      <View style={styles.headerRight} />
    </View>
  );
}

function LoadingView(props: FavoritesViewProps) {
  const { colors } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <FavoritesHeader {...props} />
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function EmptyFavorites({ colors, t, handleGoShopping }: Pick<FavoritesViewProps, 'colors' | 't' | 'handleGoShopping'>) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="heart" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('favorites.empty')}</Text>
      <TouchableOpacity style={[styles.goShoppingBtn, { backgroundColor: colors.tint }]} onPress={handleGoShopping}>
        <Text style={styles.goShoppingBtnText}>{t('favorites.go_shopping')}</Text>
      </TouchableOpacity>
    </View>
  );
}

function ProductItem({ item, colors, handleProductPress, handleRemove }: Pick<FavoritesViewProps, 'colors' | 'handleProductPress' | 'handleRemove'> & { item: ProductDTO }) {
  return (
    <TouchableOpacity style={[styles.productCard, { backgroundColor: colors.card }]} onPress={() => handleProductPress(item.id!)}>
      <Image source={{ uri: item.imageUrl || 'https://via.placeholder.com/90' }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>{item.name}</Text>
        <View style={styles.priceRow}><Text style={[styles.price, { color: colors.tint }]}>¥{item.price}</Text></View>
      </View>
      <TouchableOpacity style={styles.removeBtn} onPress={() => handleRemove(item.id!)}>
        <IconSymbol name="heart.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </TouchableOpacity>
  );
}

function ListFooter({ loadingMore, hasMore, favorites, colors, t }: Pick<FavoritesViewProps, 'loadingMore' | 'hasMore' | 'favorites' | 'colors' | 't'>) {
  if (loadingMore) return <View style={styles.footer}><ActivityIndicator size="small" color={colors.tint} /></View>;
  if (!hasMore && favorites.length > 0) return <View style={styles.footer}><Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text></View>;
  return null;
}

function FavoritesView(props: FavoritesViewProps) {
  const { colors, favorites, loadFavorites } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <FavoritesHeader {...props} withBorder />
      {favorites.length === 0 ? <EmptyFavorites {...props} /> : (
        <FlatList
          data={favorites} keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => <ProductItem item={item} colors={props.colors} handleProductPress={props.handleProductPress} handleRemove={props.handleRemove} />}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadFavorites(false)} onEndReachedThreshold={0.5}
          ListFooterComponent={<ListFooter {...props} />}
        />
      )}
    </SafeAreaView>
  );
}

export default function FavoritesScreen() {
  const controller = useFavoritesController();
  return controller.loading ? <LoadingView {...controller} /> : <FavoritesView {...controller} />;
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
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
    marginBottom: 24,
  },
  goShoppingBtn: {
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 24,
  },
  goShoppingBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  listContent: { padding: 12 },
  productCard: {
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
  removeBtn: {
    padding: 8,
  },
  footer: {
    padding: 16,
    alignItems: 'center',
  },
});
