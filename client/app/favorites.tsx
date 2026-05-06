import React, { useState, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native'
import { useRouter } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { useFocusEffect } from '@react-navigation/native'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

const FAVORITES_KEY = '@meta_web_three:favorites'

interface FavoriteProduct {
  id: number
  name: string
  pic: string
  price: number
  originalPrice?: number
  addedAt: string
}

export default function FavoritesScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  const [favorites, setFavorites] = useState<FavoriteProduct[]>([])
  const [loading, setLoading] = useState(true)

  const loadFavorites = useCallback(async () => {
    setLoading(true)
    try {
      const data = await AsyncStorage.getItem(FAVORITES_KEY)
      if (data) {
        setFavorites(JSON.parse(data))
      } else {
        setFavorites([])
      }
    } catch {
      setFavorites([])
    } finally {
      setLoading(false)
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      loadFavorites()
    }, [loadFavorites])
  )

  const handleRemove = async (id: number) => {
    const updated = favorites.filter(f => f.id !== id)
    setFavorites(updated)
    try {
      await AsyncStorage.setItem(FAVORITES_KEY, JSON.stringify(updated))
    } catch {
      Alert.alert(t('common.error'), t('favorites.remove_failed'))
    }
  }

  const handleProductPress = (id: number) => {
    router.push({ pathname: '/product/[id]', params: { id } })
  }

  const renderProductItem = ({ item }: { item: FavoriteProduct }) => (
    <TouchableOpacity style={styles.productCard} onPress={() => handleProductPress(item.id)}>
      <Image source={{ uri: item.pic }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
          {item.name}
        </Text>
        <View style={styles.priceRow}>
          <Text style={[styles.price, { color: colors.primary }]}>¥{item.price}</Text>
          {item.originalPrice && item.originalPrice > item.price && (
            <Text style={[styles.originalPrice, { color: colors.textSecondary }]}>
              ¥{item.originalPrice}
            </Text>
          )}
        </View>
        <Text style={[styles.addedAt, { color: colors.textSecondary }]}>
          {t('favorites.added_at')}: {new Date(item.addedAt).toLocaleDateString()}
        </Text>
      </View>
      <TouchableOpacity
        style={styles.removeBtn}
        onPress={() => handleRemove(item.id)}
      >
        <IconSymbol name="heart.fill" size={20} color="#FF3B30" />
      </TouchableOpacity>
    </TouchableOpacity>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('favorites.title')}</Text>
        <View style={styles.headerRight} />
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : favorites.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="heart" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('favorites.empty')}</Text>
          <TouchableOpacity style={[styles.goShoppingBtn, { backgroundColor: colors.primary }]} onPress={() => router.replace('/(tabs)')}>
            <Text style={styles.goShoppingBtnText}>{t('favorites.go_shopping')}</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={favorites}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderProductItem}
          contentContainerStyle={styles.listContent}
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
    backgroundColor: '#fff',
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
    marginBottom: 8,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
    marginBottom: 4,
  },
  price: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  originalPrice: {
    fontSize: 13,
    textDecorationLine: 'line-through',
  },
  addedAt: {
    fontSize: 12,
  },
  removeBtn: {
    padding: 8,
  },
})
