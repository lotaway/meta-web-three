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
import { productCollectionApi, DEFAULT_USER_ID } from '@/api/generated';
import type { ProductDTO } from '@/src/generated/api/models';

const PAGE_SIZE = 10;

export default function FavoritesScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  const [favorites, setFavorites] = useState<ProductDTO[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);

  const loadFavorites = useCallback(async (isRefresh = false) => {
    if (loading || loadingMore) return;

    if (isRefresh) {
      setLoading(true);
      pageNumRef.current = 1;
    } else {
      if (!hasMore) return;
      setLoadingMore(true);
    }

    try {
      const response = await productCollectionApi.list({
        xUserId: DEFAULT_USER_ID,
        pageNum: pageNumRef.current,
        pageSize: PAGE_SIZE,
      });

      if (response.data) {
        const newItems = response.data as ProductDTO[];
        if (isRefresh) {
          setFavorites(newItems);
        } else {
          setFavorites(prev => [...prev, ...newItems]);
        }
        setHasMore(newItems.length >= PAGE_SIZE);
        pageNumRef.current += 1;
      }
    } catch (error) {
      console.error('Load favorites failed:', error);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [loading, loadingMore, hasMore]);

  useFocusEffect(
    useCallback(() => {
      loadFavorites(true);
    }, [loadFavorites])
  );

  const handleRemove = async (productId: number) => {
    try {
      await productCollectionApi.delete({
        xUserId: DEFAULT_USER_ID,
        productId,
      });
      setFavorites(prev => prev.filter(item => item.id !== productId));
    } catch (error) {
      console.error('Remove favorite failed:', error);
    }
  };

  const handleProductPress = (id: number) => {
    router.push({ pathname: '/product/[id]', params: { id } });
  };

  const renderProductItem = ({ item }: { item: ProductDTO }) => (
    <TouchableOpacity
      style={[styles.productCard, { backgroundColor: colors.card }]}
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
        style={styles.removeBtn}
        onPress={() => handleRemove(item.id!)}
      >
        <IconSymbol name="heart.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <IconSymbol name="chevron.left" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>{t('favorites.title')}</Text>
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
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('favorites.title')}</Text>
        <View style={styles.headerRight} />
      </View>

      {favorites.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="heart" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('favorites.empty')}</Text>
          <TouchableOpacity
            style={[styles.goShoppingBtn, { backgroundColor: colors.tint }]}
            onPress={() => router.replace('/(tabs)')}
          >
            <Text style={styles.goShoppingBtnText}>{t('favorites.go_shopping')}</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={favorites}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderProductItem}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadFavorites(false)}
          onEndReachedThreshold={0.5}
          ListFooterComponent={
            loadingMore ? (
              <View style={styles.footer}>
                <ActivityIndicator size="small" color={colors.tint} />
              </View>
            ) : !hasMore && favorites.length > 0 ? (
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
