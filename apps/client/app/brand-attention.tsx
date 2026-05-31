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
import { memberAttentionApi, DEFAULT_USER_ID } from '@/api/generated';

interface BrandItem {
  id: number;
  name: string;
  logo: string;
  firstLetter: string;
}

const PAGE_SIZE = 10;

export default function BrandAttentionScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  const [brands, setBrands] = useState<BrandItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageNumRef = useRef(1);

  const fetchBrands = async (): Promise<BrandItem[] | null> => {
    try {
      const response = await memberAttentionApi.listAttention({
        xUserId: DEFAULT_USER_ID,
      });
      return response.data ? response.data as BrandItem[] : null;
    } catch (error) {
      return null;
    }
  };

  const updateBrandsState = (newBrands: BrandItem[], isRefresh: boolean) => {
    if (isRefresh) {
      setBrands(newBrands);
    } else {
      setBrands(prev => [...prev, ...newBrands]);
    }
    setHasMore(newBrands.length >= PAGE_SIZE);
    pageNumRef.current += 1;
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
      if (newBrands) updateBrandsState(newBrands, isRefresh);
    } catch (error) {
      console.error('Load brands failed:', error);
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

  const handleDelete = async (brandId: number) => {
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
    } */
  };

  const renderBrandItem = ({ item }: { item: BrandItem }) => (
    <View style={[styles.brandItem, { backgroundColor: colors.card }]}>
      <Image
        source={{ uri: item.logo || 'https://via.placeholder.com/60' }}
        style={styles.brandLogo}
      />
      <View style={styles.brandInfo}>
        <Text style={[styles.brandName, { color: colors.text }]}>{item.name}</Text>
        <Text style={[styles.brandLetter, { color: colors.textSecondary }]}>
          {item.firstLetter}
        </Text>
      </View>
      <TouchableOpacity
        style={styles.deleteBtn}
        onPress={() => handleDelete(item.id)}
      >
        <IconSymbol name="heart.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </View>
  );

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <IconSymbol name="chevron.left" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>品牌关注</Text>
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
        <Text style={[styles.headerTitle, { color: colors.text }]}>品牌关注</Text>
        <View style={styles.headerRight} />
      </View>

      {brands.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="star" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无关注品牌</Text>
        </View>
      ) : (
        <FlatList
          data={brands}
          keyExtractor={(item) => String(item.id)}
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
