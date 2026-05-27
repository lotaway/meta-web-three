import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  FlatList,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { useRouter } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { couponApi, DEFAULT_USER_ID } from '@/api/generated'

interface CouponItem {
  code: string
  couponTypeId: number
  couponTypeName: string
  useStatus: number
  minimumOrderAmount: number
  discountAmount: number
  startTime: number
  endTime: number
}

const USE_STATUS_TABS = [
  { value: null, label: '全部' },
  { value: 0, label: '未使用' },
  { value: 1, label: '已使用' },
  { value: 2, label: '已过期' },
]

export default function CouponListScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  const [coupons, setCoupons] = useState<CouponItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedTab, setSelectedTab] = useState<number | null>(null)
  const [claiming, setClaiming] = useState<string | null>(null)

  useEffect(() => {
    loadCoupons()
  }, [selectedTab])

  const loadCoupons = async () => {
    setLoading(true)
    try {
      const response = await couponApi.list({
        xUserId: DEFAULT_USER_ID,
        useStatus: selectedTab ?? undefined,
      })
      setCoupons(response.data ?? [])
    } catch (error) {
      Alert.alert(t('common.error'), '加载优惠券失败')
    } finally {
      setLoading(false)
    }
  }

  const handleClaimCoupon = async (couponTypeId: number) => {
    setClaiming(couponTypeId.toString())
    try {
      await couponApi.claimCoupon({
        xUserId: DEFAULT_USER_ID,
        couponTypeId,
      })
      Alert.alert(t('common.success'), '领取成功')
      loadCoupons()
    } catch (error) {
      Alert.alert(t('common.error'), '领取失败')
    } finally {
      setClaiming(null)
    }
  }

  const renderCouponItem = ({ item }: { item: CouponItem }) => (
    <TouchableOpacity
      style={[styles.couponItem, { borderColor: colors.border }]}
      onPress={() => router.push({
        pathname: '/favorite-detail',
        params: { id: item.couponTypeId }
      })}
    >
      <View style={styles.couponLeft}>
        <Text style={[styles.couponAmount, { color: colors.primary }]}>
          ¥{item.discountAmount}
        </Text>
        <Text style={[styles.couponCondition, { color: colors.textSecondary }]}>
          满¥{item.minimumOrderAmount}可用
        </Text>
      </View>
      <View style={styles.couponDivider} />
      <View style={styles.couponRight}>
        <Text style={[styles.couponName, { color: colors.text }]} numberOfLines={1}>
          {item.couponTypeName}
        </Text>
        <Text style={[styles.couponTime, { color: colors.textSecondary }]}>
          有效期至 {new Date(item.endTime).toLocaleDateString()}
        </Text>
        <View style={styles.couponFooter}>
          <Text style={[
            styles.couponStatus,
            { color: item.useStatus === 0 ? colors.primary : colors.textSecondary }
          ]}>
            {item.useStatus === 0 ? '未使用' : item.useStatus === 1 ? '已使用' : '已过期'}
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>优惠券</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Tab 切换 */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabContainer}>
        {USE_STATUS_TABS.map(tab => (
          <TouchableOpacity
            key={String(tab.value)}
            style={[
              styles.tab,
              { borderColor: selectedTab === tab.value ? colors.primary : colors.border },
              selectedTab === tab.value && { backgroundColor: colors.primary + '10' },
            ]}
            onPress={() => setSelectedTab(tab.value)}
          >
            <Text
              style={[
                styles.tabText,
                { color: selectedTab === tab.value ? colors.primary : colors.textSecondary },
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* 优惠券列表 */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : coupons.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="ticket" size={48} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
            暂无优惠券
          </Text>
        </View>
      ) : (
        <FlatList
          data={coupons}
          keyExtractor={(item) => item.code}
          renderItem={renderCouponItem}
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
  tabContainer: {
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  tab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    marginRight: 10,
  },
  tabText: {
    fontSize: 14,
  },
  listContent: {
    padding: 16,
  },
  couponItem: {
    flexDirection: 'row',
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  couponLeft: {
    width: 100,
    padding: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  couponAmount: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  couponCondition: {
    fontSize: 12,
    marginTop: 4,
  },
  couponDivider: {
    width: 1,
    backgroundColor: '#f0f0f0',
    marginVertical: 8,
  },
  couponRight: {
    flex: 1,
    padding: 12,
    justifyContent: 'space-between',
  },
  couponName: {
    fontSize: 14,
    fontWeight: '500',
  },
  couponTime: {
    fontSize: 12,
    marginTop: 4,
  },
  couponFooter: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 8,
  },
  couponStatus: {
    fontSize: 12,
    fontWeight: '500',
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
    fontSize: 14,
    marginTop: 12,
  },
})
