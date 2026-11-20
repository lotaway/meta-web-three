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
  Alert,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { useFocusEffect } from '@react-navigation/native'
import * as ImagePicker from 'expo-image-picker'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { productApi } from '@/api/generated'
import type { ProductDTO } from '@/src/generated/api/models'
import {
  aiShoppingApi,
  type AiProductMatch,
  type TextCorrection,
} from '@/lib/api/aiShopping'

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

type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type Translate = ReturnType<typeof useTranslation>['t']
type SetStringArray = React.Dispatch<React.SetStateAction<string[]>>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetSearchResults = React.Dispatch<React.SetStateAction<ProductDTO[]>>
type SetAiMatches = React.Dispatch<React.SetStateAction<AiProductMatch[]>>
type SetCorrection = React.Dispatch<React.SetStateAction<TextCorrection | null>>
type ProductSearchResult = PromiseSettledResult<Awaited<ReturnType<typeof productApi.listProducts>>>
type AiSearchResult = PromiseSettledResult<Awaited<ReturnType<typeof aiShoppingApi.search>>>

function buildSearchHistory(newKeyword: string, history: string[]) {
  return [newKeyword, ...history.filter(k => k !== newKeyword)].slice(0, MAX_HISTORY)
}

async function writeSearchHistory(history: string[]) {
  try {
    await AsyncStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(history))
  } catch (e) {
    console.error('Save search history failed:', e)
    Alert.alert('Error', 'Failed to save the search history.')
  }
}

async function loadSearchHistoryData(setHistory: SetStringArray) {
  try {
    const history = await AsyncStorage.getItem(SEARCH_HISTORY_KEY)
    if (history) setHistory(JSON.parse(history))
  } catch (e) {
    setHistory([])
  }
}

async function clearSearchHistoryData(setHistory: SetStringArray) {
  setHistory([])
  try {
    await AsyncStorage.removeItem(SEARCH_HISTORY_KEY)
  } catch (e) {
    console.error('Clear search history failed:', e)
    Alert.alert('Error', 'Failed to clear the search history.')
  }
}

function applyProductResult(result: ProductSearchResult, setResults: SetSearchResults) {
  if (result.status === 'fulfilled' && result.value.data) {
    setResults(result.value.data as ProductDTO[])
    return
  }
  console.error('Search failed:', result)
  setResults([])
}

function applyAiResult(result: AiSearchResult, matches: SetAiMatches, correction: SetCorrection, enabled: SetBoolean) {
  if (result.status !== 'fulfilled') {
    console.error('AI search failed:', result)
    enabled(false)
    return
  }
  matches(result.value.matches || [])
  correction(result.value.correction || null)
  enabled(true)
}

async function runKeywordSearch(keyword: string, state: SearchState) {
  const [productResult, aiResult] = await Promise.allSettled([
    productApi.listProducts({ keyword }),
    aiShoppingApi.search(keyword, 10),
  ])
  applyProductResult(productResult, state.setSearchResults)
  applyAiResult(aiResult, state.setAiMatches, state.setCorrection, state.setAiEnabled)
}

async function runImageSearch(uri: string, state: SearchState) {
  try {
    const blob = await (await fetch(uri)).blob()
    const matches = await aiShoppingApi.imageSearch(blob, 10)
    state.setAiMatches(matches || [])
    state.setAiEnabled(true)
  } catch (error) {
    console.error('Image search failed:', error)
    state.setAiEnabled(false)
  } finally {
    state.setIsImageSearching(false)
  }
}

async function saveSearchKeyword(keyword: string, state: SearchState) {
  if (!keyword.trim()) return
  const updated = buildSearchHistory(keyword, state.searchHistory)
  state.setSearchHistory(updated)
  await writeSearchHistory(updated)
}

async function runKeywordSearchFlow(keywordInput: string | undefined, state: SearchState) {
  const keyword = (keywordInput || state.keyword).trim()
  if (!keyword) return
  Keyboard.dismiss()
  state.setKeyword(keyword)
  state.setIsSearching(true)
  state.setHasSearched(true)
  state.setAiMatches([])
  state.setCorrection(null)
  await saveSearchKeyword(keyword, state)
  await runKeywordSearch(keyword, state)
  state.setIsSearching(false)
}

function prepareImageSearch(state: SearchState) {
  Keyboard.dismiss()
  state.setHasSearched(true)
  state.setIsImageSearching(true)
  state.setAiMatches([])
  state.setCorrection(null)
  state.setSearchResults([])
}

async function runImageSearchFlow(state: SearchState) {
  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync()
  if (!permission.granted) {
    console.warn('Media library permission denied')
    return
  }
  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    allowsEditing: true,
    aspect: [1, 1],
    quality: 0.7,
  })
  if (result.canceled || !result.assets?.[0]) return
  prepareImageSearch(state)
  await runImageSearch(result.assets[0].uri, state)
}

function useSearchState() {
  const params = useLocalSearchParams()
  const [keyword, setKeyword] = useState((params.keyword as string) || '')
  const [searchHistory, setSearchHistory] = useState<string[]>([])
  const [searchResults, setSearchResults] = useState<ProductDTO[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [aiMatches, setAiMatches] = useState<AiProductMatch[]>([])
  const [correction, setCorrection] = useState<TextCorrection | null>(null)
  const [aiEnabled, setAiEnabled] = useState(true)
  const [isImageSearching, setIsImageSearching] = useState(false)
  const inputRef = useRef<TextInput>(null)
  return { keyword, setKeyword, searchHistory, setSearchHistory, searchResults, setSearchResults, isSearching, setIsSearching, hasSearched, setHasSearched, aiMatches, setAiMatches, correction, setCorrection, aiEnabled, setAiEnabled, isImageSearching, setIsImageSearching, inputRef }
}

type SearchState = ReturnType<typeof useSearchState>

function createSearchActions(state: SearchState, router: AppRouter) {
  return {
    handleSearch: (keyword?: string) => runKeywordSearchFlow(keyword, state),
    handleImageSearch: () => runImageSearchFlow(state),
    handleProductPress: (productId: number) => router.push({ pathname: '/product/[id]', params: { id: productId } }),
    clearHistory: () => clearSearchHistoryData(state.setSearchHistory),
  }
}

type SearchActions = ReturnType<typeof createSearchActions>

function useSearchController() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const state = useSearchState()
  useFocusEffect(
    useCallback(() => {
      loadSearchHistoryData(state.setSearchHistory)
      state.inputRef.current?.focus()
    }, [])
  )
  return { t, router, colors, ...state, ...createSearchActions(state, router) }
}

interface SearchViewProps extends SearchState, SearchActions {
  t: Translate
  colors: AppColors
  router: AppRouter
}

function ClearKeywordButton({ keyword, colors, setKeyword }: Pick<SearchViewProps, 'keyword' | 'colors' | 'setKeyword'>) {
  if (!keyword) return null
  return (
    <TouchableOpacity onPress={() => setKeyword('')}>
      <IconSymbol name="xmark.circle.fill" size={20} color={colors.textSecondary} />
    </TouchableOpacity>
  )
}

function SearchInputBar({ colors, keyword, inputRef, setKeyword, handleSearch, t }: SearchViewProps) {
  return (
    <View style={[styles.searchBar, { backgroundColor: colors.card, borderColor: colors.border }]}>
      <IconSymbol name="magnifyingglass" size={20} color={colors.textSecondary} />
      <TextInput
        ref={inputRef} style={[styles.searchInput, { color: colors.text }]}
        placeholder={t('common.search_placeholder')} placeholderTextColor={colors.textSecondary}
        value={keyword} onChangeText={setKeyword} onSubmitEditing={() => handleSearch()}
        returnKeyType="search" autoCapitalize="none"
      />
      <ClearKeywordButton keyword={keyword} colors={colors} setKeyword={setKeyword} />
    </View>
  )
}

function SearchHeader(props: SearchViewProps) {
  const { colors, router, isImageSearching, handleImageSearch, handleSearch, t } = props
  return (
    <View style={[styles.header, { backgroundColor: colors.background }]}>
      <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <SearchInputBar {...props} />
      <TouchableOpacity style={styles.imageBtn} onPress={handleImageSearch} disabled={isImageSearching}>
        <IconSymbol name="photo" size={22} color={colors.text} />
      </TouchableOpacity>
      <TouchableOpacity style={styles.searchBtn} onPress={() => handleSearch()}>
        <Text style={styles.searchBtnText}>{t('search.btn')}</Text>
      </TouchableOpacity>
    </View>
  )
}

function CorrectionBanner({ correction, keyword, colors, handleSearch, t }: SearchViewProps) {
  if (!correction || !correction.changed || !correction.corrected) return null
  const next = correction.corrected === keyword ? keyword : correction.corrected
  return (
    <View style={[styles.correctionBanner, { backgroundColor: colors.tint }]}>
      <IconSymbol name="wand.and.stars" size={16} color="#fff" />
      <Text style={styles.correctionText}>
        {t('search.correction_prefix')}
        <Text style={styles.correctionStrong}>{correction.corrected}</Text>
      </Text>
      <TouchableOpacity onPress={() => handleSearch(next)}>
        <Text style={styles.correctionAction}>{t('search.correction_apply')}</Text>
      </TouchableOpacity>
    </View>
  )
}

function AiMatchInfo({ item, colors, t }: Pick<SearchViewProps, 'colors' | 't'> & { item: AiProductMatch }) {
  return (
    <View style={styles.aiInfo}>
      <Text numberOfLines={2} style={[styles.aiName, { color: colors.text }]}>{item.name}</Text>
      <Text style={[styles.aiPrice, { color: colors.primary }]}>{item.price ? `¥${item.price}` : ''}</Text>
      {item.score ? (
        <Text style={[styles.aiReason, { color: colors.textSecondary }]}>
          {`${t('search.similarity')} ${(item.score * 100).toFixed(0)}%`}
        </Text>
      ) : null}
    </View>
  )
}

function AiMatchCard({ item, colors, handleProductPress, t }: SearchViewProps & { item: AiProductMatch }) {
  return (
    <TouchableOpacity
      style={[styles.aiCard, { backgroundColor: colors.card, borderColor: colors.border }]}
      onPress={() => handleProductPress(item.productId)}
    >
      <Image source={{ uri: item.pic || '' }} style={styles.aiImage} />
      <AiMatchInfo item={item} colors={colors} t={t} />
    </TouchableOpacity>
  )
}

function AiSectionHeader({ colors, t }: SearchViewProps) {
  return (
    <View style={styles.sectionHeader}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>
        <IconSymbol name="sparkles" size={16} color={colors.primary} />
        {'  '}{t('search.ai_smart_match')}
      </Text>
    </View>
  )
}

function AiMatchesSection(props: SearchViewProps) {
  const { aiEnabled, aiMatches } = props
  if (!aiEnabled || aiMatches.length === 0) return null
  return (
    <View style={styles.aiSection}>
      <AiSectionHeader {...props} />
      <FlatList
        horizontal data={aiMatches} keyExtractor={(item) => `ai-${item.productId}`}
        showsHorizontalScrollIndicator={false} contentContainerStyle={styles.aiListContent}
        renderItem={({ item }) => <AiMatchCard {...props} item={item} />}
      />
    </View>
  )
}

function HistorySectionHeader({ colors, clearHistory, t }: SearchViewProps) {
  return (
    <View style={styles.sectionHeader}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('search.history')}</Text>
      <TouchableOpacity onPress={clearHistory}>
        <IconSymbol name="trash" size={18} color={colors.textSecondary} />
      </TouchableOpacity>
    </View>
  )
}

function SearchHistorySection(props: SearchViewProps) {
  const { searchHistory, colors, handleSearch } = props
  if (searchHistory.length === 0) return null
  return (
    <View style={styles.section}>
      <HistorySectionHeader {...props} />
      <View style={styles.historyList}>
        {searchHistory.map((item, index) => (
          <TouchableOpacity
            key={index} style={[styles.historyTag, { backgroundColor: colors.card, borderColor: colors.border }]}
            onPress={() => handleSearch(item)}
          >
            <Text style={[styles.historyTagText, { color: colors.text }]}>{item}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  )
}

function HotSearchSection({ colors, handleSearch, t }: SearchViewProps) {
  return (
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
  )
}

function SearchStartContent(props: SearchViewProps) {
  return (
    <>
      <SearchHistorySection {...props} />
      <HotSearchSection {...props} />
    </>
  )
}

function ProductCard({ item, colors, handleProductPress }: SearchViewProps & { item: ProductDTO }) {
  return (
    <TouchableOpacity style={styles.productCard} onPress={() => handleProductPress(item.id!)}>
      <Image source={{ uri: item.imageUrl || '' }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
          {item.name}
        </Text>
        <Text style={[styles.productPrice, { color: colors.primary }]}>
          ¥{item.price}
        </Text>
        {item.marketPrice && Number(item.marketPrice) > Number(item.price || 0) && (
          <Text style={[styles.productOriginalPrice, { color: colors.textSecondary }]}>
            ¥{item.marketPrice}
          </Text>
        )}
      </View>
    </TouchableOpacity>
  )
}

function LoadingState({ isImageSearching, colors, t }: SearchViewProps) {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={colors.primary} />
      <Text style={[styles.loadingText, { color: colors.textSecondary }]}>
        {isImageSearching ? t('search.ai_image_searching') : t('search.loading')}
      </Text>
    </View>
  )
}

function EmptyState({ colors, t }: SearchViewProps) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="magnifyingglass" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('search.no_results')}</Text>
    </View>
  )
}

function SearchResultsView(props: SearchViewProps) {
  const { colors, isSearching, isImageSearching, searchResults, aiMatches } = props
  if (isSearching || isImageSearching) return <LoadingState {...props} />
  return (
    <FlatList
      data={searchResults} keyExtractor={(item) => String(item.id)}
      numColumns={2} columnWrapperStyle={styles.row} contentContainerStyle={styles.listContent}
      ListHeaderComponent={
        <>
          <CorrectionBanner {...props} />
          <AiMatchesSection {...props} />
        </>
      }
      ListEmptyComponent={aiMatches.length > 0 ? null : <EmptyState {...props} />}
      renderItem={({ item }) => <ProductCard {...props} item={item} />}
    />
  )
}

function SearchView(props: SearchViewProps) {
  const { colors, hasSearched } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <SearchHeader {...props} />
      <View style={styles.content}>
        {hasSearched ? <SearchResultsView {...props} /> : <SearchStartContent {...props} />}
      </View>
    </SafeAreaView>
  )
}

export default function SearchScreen() {
  const controller = useSearchController()
  return <SearchView {...controller} />
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
  imageBtn: {
    padding: 8,
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
    paddingTop: 80,
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
  correctionBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    marginHorizontal: 8,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 6,
  },
  correctionText: {
    flex: 1,
    color: '#fff',
    fontSize: 13,
  },
  correctionStrong: {
    fontWeight: '700',
  },
  correctionAction: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
    textDecorationLine: 'underline',
  },
  aiSection: {
    marginBottom: 8,
  },
  aiListContent: {
    paddingHorizontal: 8,
    gap: 10,
  },
  aiCard: {
    width: 140,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
  },
  aiImage: {
    width: '100%',
    aspectRatio: 1,
  },
  aiInfo: {
    padding: 8,
  },
  aiName: {
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 4,
  },
  aiPrice: {
    fontSize: 15,
    fontWeight: 'bold',
  },
  aiReason: {
    fontSize: 11,
    marginTop: 4,
  },
})
