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

export default function CategoryListScreen() {
  const { t } = useTranslation();
  const { id } = useLocalSearchParams();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  
  const [products, setProducts] = useState<ProductDTO[]>([]);
  const [cateList, setCateList] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [sortType, setSortType] = useState<SortType>('comprehensive');
  const [priceOrder, setPriceOrder] = useState<'asc' | 'desc'>('asc');
  const [cateMaskVisible, setCateMaskVisible] = useState(false);
  const [selectedCateId, setSelectedCateId] = useState<number | null>(
    id ? Number(id) : null
  );
  
  const pageNumRef = useRef(1);
  const pageSize = 10;

  const getSortParam = (): number => {
    switch (sortType) {
      case 'comprehensive': return 0;
      case 'sales': return 2;
      case 'price-asc': return 3;
      case 'price-desc': return 4;
      default: return 0;
    }
  };

  const loadCateList = useCallback(async () => {
    try {
      const parentId = id ? Math.floor(Number(id) / 100) * 100 : 0;
      const response = await categoryApi.viewChildren({ parentId });
      if (response.data) {
        setCateList(response.data as any[]);
      }
    } catch (error) {
      console.error('Load category list failed:', error);
    }
  }, [id]);

  const fetchProducts = useCallback(async () => {
    const response = await productApi.listProducts({
      productCategoryId: selectedCateId || undefined,
      pageNum: pageNumRef.current,
      pageSize,
      sort: getSortParam(),
    });
    return response.data as ProductDTO[] | null;
  }, [selectedCateId, sortType]);

  const updateProductState = useCallback((newProducts: ProductDTO[], isRefresh: boolean) => {
    if (isRefresh) {
      setProducts(newProducts);
    } else {
      setProducts(prev => [...prev, ...newProducts]);
    }
    setHasMore(newProducts.length >= pageSize);
    pageNumRef.current += 1;
  }, []);

  const loadProducts = useCallback(async (isRefresh = false) => {
    if (loading || loadingMore) return;
    if (isRefresh) { setLoading(true); pageNumRef.current = 1; }
    else { if (!hasMore) return; setLoadingMore(true); }
    try {
      const newProducts = await fetchProducts();
      if (newProducts) updateProductState(newProducts, isRefresh);
    } catch (error) {
      console.error('Load products failed:', error);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [selectedCateId, sortType, loading, loadingMore, hasMore, fetchProducts, updateProductState]);

  React.useEffect(() => {
    loadProducts(true);
  }, []);

  // 切换分类
  const handleCateChange = (cateId: number) => {
    setSelectedCateId(cateId);
    setCateMaskVisible(false);
    setProducts([]);
    pageNumRef.current = 1;
    loadProducts(true);
  };

  // 切换排序
  const handleSortChange = (type: SortType) => {
    if (type === 'price-asc' || type === 'price-desc') {
      if (sortType === 'price-asc') {
        setSortType('price-desc');
      } else if (sortType === 'price-desc') {
        setSortType('price-asc');
      } else {
        setSortType(type);
      }
    } else {
      setSortType(type);
    }
    setProducts([]);
    pageNumRef.current = 1;
    loadProducts(true);
  };

  // 查看商品详情
  const handleProductPress = (productId: number) => {
    router.push({ pathname: '/product/[id]', params: { id: productId } });
  };

  // 渲染商品项
  const renderProductItem = ({ item }: { item: ProductDTO }) => (
    <TouchableOpacity
      style={[styles.productItem, { backgroundColor: colors.card }]}
      onPress={() => handleProductPress(item.id!)}
    >
      <Image
        source={{ uri: item.pic || 'https://via.placeholder.com/150' }}
        style={styles.productImage}
        resizeMode="cover"
      />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
          {item.name}
        </Text>
        {item.subTitle && (
          <Text numberOfLines={1} style={[styles.productSubtitle, { color: colors.textSecondary }]}>
            {item.subTitle}
          </Text>
        )}
        <View style={styles.priceRow}>
          <Text style={[styles.price, { color: colors.tint }]}>¥{item.price}</Text>
        </View>
      </View>
    </TouchableOpacity>
  )

  const renderSortBar = () => (
    <View style={[styles.sortBar, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity
        style={[styles.sortItem, sortType === 'comprehensive' && styles.sortItemActive]}
        onPress={() => handleSortChange('comprehensive')}
      >
        <Text style={[styles.sortText, sortType === 'comprehensive' && styles.sortTextActive]}>
          {t('search.sort.comprehensive')}
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={[styles.sortItem, sortType === 'sales' && styles.sortItemActive]}
        onPress={() => handleSortChange('sales')}
      >
        <Text style={[styles.sortText, sortType === 'sales' && styles.sortTextActive]}>
          {t('search.sort.sales')}
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={[styles.sortItem, (sortType === 'price-asc' || sortType === 'price-desc') && styles.sortItemActive]}
        onPress={() => handleSortChange('price-asc')}
      >
        <Text style={[styles.sortText, (sortType === 'price-asc' || sortType === 'price-desc') && styles.sortTextActive]}>
          {t('search.sort.price')}
        </Text>
        <View style={styles.priceOrderIcons}>
          <IconSymbol
            name="chevron.up"
            size={12}
            color={sortType === 'price-asc' ? colors.tint : colors.textSecondary}
          />
          <IconSymbol
            name="chevron.down"
            size={12}
            color={sortType === 'price-desc' ? colors.tint : colors.textSecondary}
          />
        </View>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={styles.cateButton}
        onPress={() => setCateMaskVisible(true)}
      >
        <IconSymbol name="line.3.horizontal.decrease.circle" size={20} color={colors.text} />
      </TouchableOpacity>
    </View>
  );

  // 渲染分类筛选面板
  const renderCateMask = () => (
    <View
      pointerEvents={cateMaskVisible ? 'auto' : 'none'}
      style={[
        styles.cateMask,
        { backgroundColor: 'rgba(0,0,0,0.4)' },
        cateMaskVisible ? styles.cateMaskVisible : styles.cateMaskHidden,
      ]}
    >
      <TouchableOpacity
        style={styles.cateMaskOverlay}
        onPress={() => setCateMaskVisible(false)}
      />
      <View style={[styles.catePanel, { backgroundColor: colors.background }]}>
        <FlatList
          data={cateList}
          keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={[
                styles.cateItem,
                selectedCateId === item.id && styles.cateItemSelected
              ]}
              onPress={() => handleCateChange(item.id)}
            >
              <Text style={[
                styles.cateItemText,
                { color: selectedCateId === item.id ? colors.tint : colors.text }
              ]}>
                {item.name}
              </Text>
            </TouchableOpacity>
          )}
        />
      </View>
    </View>
  );

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <ActivityIndicator size="large" color={colors.tint} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      {renderSortBar()}
      
      <FlatList
        data={products}
        keyExtractor={(item) => String(item.id)}
        renderItem={renderProductItem}
        numColumns={2}
        columnWrapperStyle={styles.productRow}
        contentContainerStyle={styles.productList}
        onEndReached={() => loadProducts(false)}
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
        ListEmptyComponent={
          !loading && products.length === 0 ? (
            <View style={styles.empty}>
              <Text style={{ color: colors.textSecondary }}>{t('common.no_data')}</Text>
            </View>
          ) : null
        }
      />
      
      {renderCateMask()}
    </SafeAreaView>
  );
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
