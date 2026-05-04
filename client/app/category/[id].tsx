import React, { useState, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Image,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { productApi, categoryApi } from '@/api/generated'
import type { ProductDTO } from '@/src/generated/api/models'

type SortType = 'default' | 'price_asc' | 'price_desc' | 'sales'

export default function CategoryProductListScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  
  const categoryId = params.id ? parseInt(params.id as string, 10) : 0
  const categoryName = params.name as string || ''

  const [products, setProducts] = useState<ProductDTO[]>([])
  const [loading, setLoading] = useState(true)
  const [sortType, setSortType] = useState<SortType>('default')
  const [viewType, setViewType] = useState<'grid' | 'list'>('grid')

  const loadProducts = useCallback(async () => {
    setLoading(true)
    try {
      const response = await productApi.listProducts({ categoryId })
      let result = response.data || []
      
      switch (sortType) {
        case 'price_asc':
          result = [...result].sort((a, b) => (a.price || 0) - (b.price || 0))
          break
        case 'price_desc':
          result = [...result].sort((a, b) => (b.price || 0) - (a.price || 0))
          break
        case 'sales':
          result = [...result].sort((a, b) => (b.sale || 0) - (a.sale || 0))
          break
      }
      
      setProducts(result)
    } catch (error) {
      console.error('Failed to load products:', error)
    } finally {
      setLoading(false)
    }
  }, [categoryId, sortType])

  React.useEffect(() => {
    loadProducts()
  }, [loadProducts])

  const handleSort = (type: SortType) => {
    setSortType(type)
  }

  const handleProductPress = (productId: number) => {
    router.push({ pathname: '/product/[id]', params: { id: productId } })
  }

  const renderHeader = () => (
    <View style={[styles.header, { backgroundColor: colors.background }]}>
      <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>
        {categoryName || t('category.title')}
      </Text>
      <View style={styles.headerRight}>
        <TouchableOpacity onPress={() => setViewType(viewType === 'grid' ? 'list' : 'grid')}>
          <IconSymbol
            name={viewType === 'grid' ? 'square.grid.2x2' : 'list.bullet'}
            size={24}
            color={colors.text}
          />
        </TouchableOpacity>
      </View>
    </View>
  )

  const renderSortBar = () => (
    <View style={[styles.sortBar, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity
        style={[styles.sortItem, sortType === 'default' && styles.sortItemActive]}
        onPress={() => handleSort('default')}
      >
        <Text style={[styles.sortText, { color: sortType === 'default' ? colors.primary : colors.text }]}>
          {t('category.sort_default')}
        </Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.sortItem, sortType === 'price_asc' && styles.sortItemActive]}
        onPress={() => handleSort('price_asc')}
      >
        <Text style={[styles.sortText, { color: sortType === 'price_asc' ? colors.primary : colors.text }]}>
          {t('category.sort_price')}
        </Text>
        <IconSymbol
          name={sortType === 'price_asc' ? 'arrow.up' : 'arrow.up.arrow.down'}
          size={12}
          color={sortType === 'price_asc' ? colors.primary : colors.textSecondary}
        />
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.sortItem, sortType === 'sales' && styles.sortItemActive]}
        onPress={() => handleSort('sales')}
      >
        <Text style={[styles.sortText, { color: sortType === 'sales' ? colors.primary : colors.text }]}>
          {t('category.sort_sales')}
        </Text>
      </TouchableOpacity>
    </View>
  )

  const renderProductCard = ({ item }: { item: ProductDTO }) => {
    if (viewType === 'grid') {
      return (
        <TouchableOpacity style={styles.gridProductCard} onPress={() => handleProductPress(item.id!)}>
          <Image
            source={{ uri: item.pic || item.album?.[0] || '' }}
            style={styles.gridProductImage}
          />
          <View style={styles.gridProductInfo}>
            <Text numberOfLines={2} style={[styles.gridProductName, { color: colors.text }]}>
              {item.name}
            </Text>
            <View style={styles.priceRow}>
              <Text style={[styles.gridProductPrice, { color: colors.primary }]}>
                ¥{item.price}
              </Text>
              {item.originalPrice && item.originalPrice > (item.price || 0) && (
                <Text style={[styles.originalPrice, { color: colors.textSecondary }]}>
                  ¥{item.originalPrice}
                </Text>
              )}
            </View>
          </View>
        </TouchableOpacity>
      )
    }

    return (
      <TouchableOpacity style={styles.listProductCard} onPress={() => handleProductPress(item.id!)}>
        <Image
          source={{ uri: item.pic || item.album?.[0] || '' }}
          style={styles.listProductImage}
        />
        <View style={styles.listProductInfo}>
          <Text numberOfLines={2} style={[styles.listProductName, { color: colors.text }]}>
            {item.name}
          </Text>
          <Text numberOfLines={1} style={[styles.listProductSubtitle, { color: colors.textSecondary }]}>
            {item.subTitle || ''}
          </Text>
          <View style={styles.listPriceRow}>
            <Text style={[styles.listProductPrice, { color: colors.primary }]}>
              ¥{item.price}
            </Text>
            {item.sale && item.sale > 0 && (
              <Text style={[styles.salesText, { color: colors.textSecondary }]}>
                {t('category.sales')}: {item.sale}
              </Text>
            )}
          </View>
        </View>
      </TouchableOpacity>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      {renderHeader()}
      {renderSortBar()}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : products.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="shippingbox" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
            {t('category.no_products')}
          </Text>
        </View>
      ) : viewType === 'grid' ? (
        <FlatList
          data={products}
          keyExtractor={(item) => String(item.id)}
          numColumns={2}
          columnWrapperStyle={styles.gridRow}
          contentContainerStyle={styles.gridContent}
          renderItem={renderProductCard}
        />
      ) : (
        <FlatList
          data={products}
          keyExtractor={(item) => String(item.id)}
          contentContainerStyle={styles.listContent}
          renderItem={renderProductCard}
          ItemSeparatorComponent={() => <View style={styles.listSeparator} />}
        />
      )}
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  backBtn: { padding: 8 },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginRight: 40,
  },
  headerRight: {
    position: 'absolute',
    right: 12,
  },
  sortBar: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    paddingHorizontal: 8,
  },
  sortItem: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 4,
  },
  sortItemActive: {
    borderBottomWidth: 2,
    borderBottomColor: '#333',
  },
  sortText: {
    fontSize: 14,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
  },
  gridContent: {
    padding: 8,
  },
  gridRow: {
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  gridProductCard: {
    width: '48%',
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  gridProductImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
  },
  gridProductInfo: {
    padding: 8,
  },
  gridProductName: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 4,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  gridProductPrice: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  originalPrice: {
    fontSize: 12,
    textDecorationLine: 'line-through',
  },
  listContent: {
    padding: 8,
  },
  listSeparator: {
    height: 8,
  },
  listProductCard: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 8,
    overflow: 'hidden',
    padding: 12,
  },
  listProductImage: {
    width: 120,
    height: 120,
    borderRadius: 8,
  },
  listProductInfo: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'space-between',
  },
  listProductName: {
    fontSize: 16,
    fontWeight: '500',
    lineHeight: 22,
  },
  listProductSubtitle: {
    fontSize: 13,
    marginTop: 4,
  },
  listPriceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  listProductPrice: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  salesText: {
    fontSize: 12,
  },
})
