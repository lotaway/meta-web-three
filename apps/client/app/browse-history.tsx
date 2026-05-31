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
import { useRouter, useFocusEffect } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { readHistoryApi, DEFAULT_USER_ID } from '@/api/generated';
import type { ProductDTO } from '@/src/generated/api/models';

const PAGE_SIZE = 10;

export default function BrowseHistoryScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  const [products, setProducts] = useState<ProductDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);

  const fetchHistory = async (): Promise<ProductDTO[] | null> => {
    try {
      const response = await readHistoryApi.listHistory({
        xUserId: DEFAULT_USER_ID,
      });
      return response.data ? response.data as ProductDTO[] : null;
    } catch (error) {
      return null;
    }
  };

  const updateHistoryState = (newItems: ProductDTO[], isRefresh: boolean) => {
    if (isRefresh) {
      setProducts(newItems);
    } else {
      setProducts(prev => [...prev, ...newItems]);
    }
    setHasMore(newItems.length >= PAGE_SIZE);
    pageNumRef.current += 1;
  };

  const loadHistory = useCallback(async (isRefresh = false) => {
    if (loading || loadingMore) return;
    if (!isRefresh && !hasMore) return;
    
    if (isRefresh) {
      setLoading(true);
      pageNumRef.current = 1;
    } else {
      setLoadingMore(true);
    }
    
    try {
      const newItems = await fetchHistory();
      if (newItems) updateHistoryState(newItems, isRefresh);
    } catch (error) {
      console.error('Load history failed:', error);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [loading, loadingMore, hasMore]);

  useFocusEffect(
    useCallback(() => {
      loadHistory(true);
    }, [loadHistory])
  );

  const handleDelete = async (productId: number) => {
    try {
      await readHistoryApi.deleteCollection({
        xUserId: DEFAULT_USER_ID,
        productId,
      });
      setProducts(prev => prev.filter(item => item.id !== productId));
    } catch (error) {
      console.error('Delete history failed:', error);
    }
  };

  const handleClear = async () => {
    // TODO: 需要后端 API 支持清除浏览历史
    console.log('Clear history not implemented');
    /* try {
      await readHistoryApi.clearHistory({ xUserId: DEFAULT_USER_ID });
      setProducts([]);
    } catch (error) {
      console.error('Clear history failed:', error);
    } */
  };

  const handleProductPress = (productId: number) => {
    router.push({ pathname: '/product/[id]', params: { id: productId } });
  };

  const renderProductItem = ({ item }: { item: ProductDTO }) => (
    <TouchableOpacity
      style={[styles.productItem, { backgroundColor: colors.card }]}
      onPress={() => handleProductPress(item.id!)}
    >
      <Image
        source={{ uri: item.pic || 'https://via.placeholder.com/90' }}
        style={styles.productImage}
      />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
          {item.name}
        </Text>
        {item.subTitle && (
          <Text numberOfLines={1} style={[styles.subTitle, { color: colors.textSecondary }]}>
            {item.subTitle}
          </Text>
        )}
        <View style={styles.priceRow}>
          <Text style={[styles.price, { color: colors.tint }]}>¥{item.price}</Text>
        </View>
      </View>
      <TouchableOpacity
        style={styles.deleteBtn}
        onPress={() => handleDelete(item.id!)}
      >
        <IconSymbol name="xmark.circle.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={[styles.header, { borderBottomColor: colors.border }]}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <IconSymbol name="chevron.left" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>浏览历史</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.loadingContainer}>
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
        <Text style={[styles.headerTitle, { color: colors.text }]}>浏览历史</Text>
        <TouchableOpacity style={styles.clearBtn} onPress={handleClear}>
          <Text style={{ color: colors.tint }}>清空</Text>
        </TouchableOpacity>
      </View>

      {products.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="clock" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无浏览历史</Text>
        </View>
      ) : (
        <FlatList
          data={products}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderProductItem}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadHistory(false)}
          onEndReachedThreshold={0.5}
          ListFooterComponent={
            loadingMore ? (
              <View style={styles.footer}>
                <ActivityIndicator size="small" color={colors.tint} />
              </View>
            ) : !hasMore && products.length > 0 ? (
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
