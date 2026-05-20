import React, { useState, useCallback, useRef } from 'react'
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  Image,
  ActivityIndicator,
  Keyboard,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { useFocusEffect } from '@react-navigation/native'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { productApi } from '@/api/generated'
import type { ProductDTO } from '@/src/generated/api/models'

const SEARCH_HISTORY_KEY = '@meta_web_three:search_history'
const MAX_HISTORY = 10

const HOT_SEARCHES = [
  { id: 1, keyword: '手机' },
  { id: 2, keyword: '耳机' },
  { id: 3, keyword: '电脑' },
  { id: 4, keyword: '手表' },
  { id: 5, keyword: '平板' },
  { id: 6, keyword: '相机' },
]

export default function SearchScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  
  const [keyword, setKeyword] = useState((params.keyword as string) || '')
  const [searchHistory, setSearchHistory] = useState<string[]>([])
  const [searchResults, setSearchResults] = useState<ProductDTO[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const inputRef = useRef<TextInput>(null)

  useFocusEffect(
    useCallback(() => {
      loadSearchHistory()
      inputRef.current?.focus()
    }, [])
  )

  const loadSearchHistory = async () => {
    try {
      const history = await AsyncStorage.getItem(SEARCH_HISTORY_KEY)
      if (history) {
        setSearchHistory(JSON.parse(history))
      }
    } catch (e) {
      setSearchHistory([])
    }
  }

  const saveSearchHistory = async (newKeyword: string) => {
    if (!newKeyword.trim()) return
    
    const updated = [newKeyword, ...searchHistory.filter(k => k !== newKeyword)].slice(0, MAX_HISTORY)
    setSearchHistory(updated)
    try {
      await AsyncStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(updated))
    } catch (e) {
      console.error('Save search history failed:', e)
    }
  }

  const clearHistory = async () => {
    setSearchHistory([])
    try {
      await AsyncStorage.removeItem(SEARCH_HISTORY_KEY)
    } catch (e) {
      console.error('Clear search history failed:', e)
    }
  }

  const handleSearch = async (searchKeyword?: string) => {
    const kw = (searchKeyword || keyword).trim()
    if (!kw) return

    Keyboard.dismiss()
    setKeyword(kw)
    setIsSearching(true)
    setHasSearched(true)
    await saveSearchHistory(kw)

    try {
      const response = await productApi.listProducts({ keyword: kw })
      if (response.data) {
        setSearchResults(response.data)
      }
    } catch (error) {
      console.error('Search failed:', error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleProductPress = (productId: number) => {
    router.push({ pathname: '/product/[id]', params: { id: productId } })
  }

  const renderHeader = () => (
    <View style={[styles.header, { backgroundColor: colors.background }]}>
      <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <View style={[styles.searchBar, { backgroundColor: colors.card, borderColor: colors.border }]}>
        <IconSymbol name="magnifyingglass" size={20} color={colors.textSecondary} />
        <TextInput
          ref={inputRef}
          style={[styles.searchInput, { color: colors.text }]}
          placeholder={t('common.search_placeholder')}
          placeholderTextColor={colors.textSecondary}
          value={keyword}
          onChangeText={setKeyword}
          onSubmitEditing={() => handleSearch()}
          returnKeyType="search"
          autoCapitalize="none"
        />
        {keyword ? (
          <TouchableOpacity onPress={() => setKeyword('')}>
            <IconSymbol name="xmark.circle.fill" size={20} color={colors.textSecondary} />
          </TouchableOpacity>
        ) : null}
      </View>
      <TouchableOpacity style={styles.searchBtn} onPress={() => handleSearch()}>
        <Text style={styles.searchBtnText}>{t('search.btn')}</Text>
      </TouchableOpacity>
    </View>
  )

  if (!hasSearched) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        {renderHeader()}
        <View style={styles.content}>
          {searchHistory.length > 0 && (
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('search.history')}</Text>
                <TouchableOpacity onPress={clearHistory}>
                  <IconSymbol name="trash" size={18} color={colors.textSecondary} />
                </TouchableOpacity>
              </View>
              <View style={styles.historyList}>
                {searchHistory.map((item, index) => (
                  <TouchableOpacity
                    key={index}
                    style={[styles.historyTag, { backgroundColor: colors.card, borderColor: colors.border }]}
                    onPress={() => handleSearch(item)}
                  >
                    <Text style={[styles.historyTagText, { color: colors.text }]}>{item}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          )}

          <View style={styles.section}>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('search.hot')}</Text>
            <View style={styles.historyList}>
              {HOT_SEARCHES.map((item) => (
                <TouchableOpacity
                  key={item.id}
                  style={[styles.historyTag, { backgroundColor: colors.card, borderColor: colors.border }]}
                  onPress={() => handleSearch(item.keyword)}
                >
                  <IconSymbol name="flame.fill" size={14} color="#FF6B35" />
                  <Text style={[styles.historyTagText, { color: colors.text }]}>{item.keyword}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      {renderHeader()}
      <View style={styles.content}>
        {isSearching ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={colors.primary} />
            <Text style={[styles.loadingText, { color: colors.textSecondary }]}>{t('search.loading')}</Text>
          </View>
        ) : searchResults.length === 0 ? (
          <View style={styles.emptyContainer}>
            <IconSymbol name="magnifyingglass" size={60} color={colors.textSecondary} />
            <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('search.no_results')}</Text>
          </View>
        ) : (
          <FlatList
            data={searchResults}
            keyExtractor={(item) => String(item.id)}
            numColumns={2}
            columnWrapperStyle={styles.row}
            contentContainerStyle={styles.listContent}
            renderItem={({ item }) => (
              <TouchableOpacity style={styles.productCard} onPress={() => handleProductPress(item.id!)}>
                <Image source={{ uri: item.pic || item.album?.[0] || '' }} style={styles.productImage} />
                <View style={styles.productInfo}>
                  <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
                    {item.name}
                  </Text>
                  <Text style={[styles.productPrice, { color: colors.primary }]}>
                    ¥{item.price}
                  </Text>
                  {item.originalPrice && item.originalPrice > (item.price || 0) && (
                    <Text style={[styles.productOriginalPrice, { color: colors.textSecondary }]}>
                      ¥{item.originalPrice}
                    </Text>
                  )}
                </View>
              </TouchableOpacity>
            )}
          />
        )}
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  backBtn: { padding: 8 },
  searchBar: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginHorizontal: 8,
    borderWidth: 1,
  },
  searchInput: {
    flex: 1,
    marginLeft: 8,
    fontSize: 14,
  },
  searchBtn: {
    paddingHorizontal: 12,
  },
  searchBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  content: { flex: 1 },
  section: {
    padding: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  historyList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  historyTag: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    borderWidth: 1,
    gap: 6,
  },
  historyTagText: {
    fontSize: 14,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
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
  listContent: {
    padding: 8,
  },
  row: {
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  productCard: {
    width: '48%',
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  productImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
  },
  productInfo: {
    padding: 8,
  },
  productName: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 4,
  },
  productPrice: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  productOriginalPrice: {
    fontSize: 12,
    textDecorationLine: 'line-through',
  },
})
