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
import { memberAttentionApi, DEFAULT_USER_ID } from '@/api/generated';

interface BrandItem {
  id: number;
  name: string;
  logo: string;
  firstLetter: string;
}

const PAGE_SIZE = 10;

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type Translate = ReturnType<typeof useTranslation>['t'];
type SetBrands = React.Dispatch<React.SetStateAction<BrandItem[]>>;
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>;
type PageRef = React.MutableRefObject<number>;

async function fetchAttentionBrands(): Promise<BrandItem[] | null> {
  try {
    const response = await memberAttentionApi.listAttention1({ xUserId: DEFAULT_USER_ID });
    return response.data ? response.data as BrandItem[] : null;
  } catch (error) {
    return null;
  }
}

function updateBrandsState(newBrands: BrandItem[], isRefresh: boolean, state: AttentionState) {
  if (isRefresh) state.setBrands(newBrands);
  else state.setBrands(prev => [...prev, ...newBrands]);
  state.setHasMore(newBrands.length >= PAGE_SIZE);
  state.pageNumRef.current += 1;
}

async function loadBrandsData(isRefresh: boolean, state: AttentionState) {
  if (state.loading || state.loadingMore) return;
  if (!isRefresh && !state.hasMore) return;
  if (isRefresh) {
    state.setLoading(true);
    state.pageNumRef.current = 1;
  } else {
    state.setLoadingMore(true);
  }
  try {
    const newBrands = await fetchAttentionBrands();
    if (newBrands) updateBrandsState(newBrands, isRefresh, state);
  } catch (error) {
    console.error('Load brands failed:', error);
    Alert.alert('Error', 'Failed to load brands.');
  } finally {
    state.setLoading(false);
    state.setLoadingMore(false);
  }
}

function handleDeleteBrand(brandId: number) {
  // TODO: 需要后端 API 支持删除品牌关注
  console.log('Delete brand not implemented, brandId:', brandId);
  /* try {
    await memberAttentionApi.deleteCollection({
      xUserId: DEFAULT_USER_ID,
      brandId,
    });
    setBrands(prev => prev.filter(item => item.id !== brandId));
  } catch (error) {
    console.error('Delete brand failed:', error);
    Alert.alert('Error', 'Failed to delete the brand.');
  } */
}

function useAttentionState() {
  const [brands, setBrands] = useState<BrandItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);
  return { brands, setBrands, loading, setLoading, loadingMore, setLoadingMore, hasMore, setHasMore, pageNumRef };
}

type AttentionState = ReturnType<typeof useAttentionState>;

function createAttentionActions(state: AttentionState, router: AppRouter) {
  return {
    handleDelete: handleDeleteBrand,
    handleBack: () => router.back(),
  };
}

type AttentionActions = ReturnType<typeof createAttentionActions>;

function useBrandAttentionController() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const state = useAttentionState();
  const loadBrands = useCallback(async (isRefresh = false) => {
    await loadBrandsData(isRefresh, state);
  }, [state.loading, state.loadingMore, state.hasMore]);
  useFocusEffect(
    useCallback(() => {
      loadBrands(true);
    }, [loadBrands])
  );
  return { t, router, colors, ...state, loadBrands, ...createAttentionActions(state, router) };
}

interface AttentionViewProps extends AttentionState, AttentionActions {
  t: Translate;
  router: AppRouter;
  colors: AppColors;
  loadBrands: (isRefresh?: boolean) => Promise<void>;
}

function AttentionHeader({ colors, handleBack, withBorder }: Pick<AttentionViewProps, 'colors' | 'handleBack'> & { withBorder?: boolean }) {
  return (
    <View style={withBorder ? [styles.header, { borderBottomColor: colors.border }] : styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>品牌关注</Text>
      <View style={styles.headerRight} />
    </View>
  );
}

function LoadingView(props: AttentionViewProps) {
  const { colors } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <AttentionHeader {...props} />
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function EmptyBrands({ colors }: Pick<AttentionViewProps, 'colors'>) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="star" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无关注品牌</Text>
    </View>
  );
}

function BrandItemRow({ item, colors, handleDelete }: Pick<AttentionViewProps, 'colors' | 'handleDelete'> & { item: BrandItem }) {
  return (
    <View style={[styles.brandItem, { backgroundColor: colors.card }]}>
      <Image source={{ uri: item.logo || 'https://via.placeholder.com/60' }} style={styles.brandLogo} />
      <View style={styles.brandInfo}>
        <Text style={[styles.brandName, { color: colors.text }]}>{item.name}</Text>
        <Text style={[styles.brandLetter, { color: colors.textSecondary }]}>{item.firstLetter}</Text>
      </View>
      <TouchableOpacity style={styles.deleteBtn} onPress={() => handleDelete(item.id)}>
        <IconSymbol name="heart.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </View>
  );
}

function ListFooter({ loadingMore, hasMore, brands, colors, t }: Pick<AttentionViewProps, 'loadingMore' | 'hasMore' | 'brands' | 'colors' | 't'>) {
  if (loadingMore) return <View style={styles.footer}><ActivityIndicator size="small" color={colors.tint} /></View>;
  if (!hasMore && brands.length > 0) return <View style={styles.footer}><Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text></View>;
  return null;
}

function AttentionView(props: AttentionViewProps) {
  const { colors, brands, loadBrands } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <AttentionHeader {...props} withBorder />
      {brands.length === 0 ? <EmptyBrands colors={colors} /> : (
        <FlatList
          data={brands} keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => <BrandItemRow item={item} colors={colors} handleDelete={props.handleDelete} />}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadBrands(false)} onEndReachedThreshold={0.5}
          ListFooterComponent={<ListFooter {...props} />}
        />
      )}
    </SafeAreaView>
  );
}

export default function BrandAttentionScreen() {
  const controller = useBrandAttentionController();
  return controller.loading ? <LoadingView {...controller} /> : <AttentionView {...controller} />;
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
  emptyText: { marginTop: 12, fontSize: 14 },
  listContent: { padding: 12 },
  brandItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    marginBottom: 12,
    borderRadius: 12,
  },
  brandLogo: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  brandInfo: {
    flex: 1,
    marginLeft: 12,
  },
  brandName: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 4,
  },
  brandLetter: {
    fontSize: 12,
  },
  deleteBtn: { padding: 8 },
  footer: {
    padding: 16,
    alignItems: 'center',
  },
});
