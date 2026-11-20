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

const COUPON_STATUS_LABELS: Record<number, string> = {
  0: '未使用',
  1: '已使用',
}

type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type Translate = ReturnType<typeof useTranslation>['t']
type SetCoupons = React.Dispatch<React.SetStateAction<CouponItem[]>>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetSelectedTab = React.Dispatch<React.SetStateAction<number | null>>
type SetClaiming = React.Dispatch<React.SetStateAction<string | null>>

function getCouponStatusLabel(useStatus: number) {
  return COUPON_STATUS_LABELS[useStatus] ?? '已过期'
}

async function fetchCoupons(selectedTab: number | null): Promise<CouponItem[]> {
  const response = await couponApi.listCoupons({ useStatus: selectedTab ?? undefined })
  return (response.data ?? []) as any
}

async function loadCouponsData(selectedTab: number | null, setLoading: SetBoolean, setCoupons: SetCoupons, t: Translate) {
  setLoading(true)
  try {
    setCoupons(await fetchCoupons(selectedTab))
  } catch (error) {
    Alert.alert(t('common.error'), '加载优惠券失败')
  } finally {
    setLoading(false)
  }
}

async function claimCoupon(couponTypeId: number, t: Translate, reload: () => Promise<void>, state: CouponState) {
  state.setClaiming(couponTypeId.toString())
  try {
    await couponApi.claimCoupon({ couponClaimRequest: { couponTypeId } })
    Alert.alert(t('common.success'), '领取成功')
    reload()
  } catch (error) {
    Alert.alert(t('common.error'), '领取失败')
  } finally {
    state.setClaiming(null)
  }
}

function useCouponState() {
  const [coupons, setCoupons] = useState<CouponItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedTab, setSelectedTab] = useState<number | null>(null)
  const [claiming, setClaiming] = useState<string | null>(null)
  return { coupons, setCoupons, loading, setLoading, selectedTab, setSelectedTab, claiming, setClaiming }
}

type CouponState = ReturnType<typeof useCouponState>

function useCouponController() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const state = useCouponState()
  const loadCoupons = () => loadCouponsData(state.selectedTab, state.setLoading, state.setCoupons, t)
  useEffect(() => {
    loadCoupons()
  }, [state.selectedTab])
  return {
    t, router, colors, ...state, loadCoupons,
    handleBack: () => router.back(),
    handleClaimCoupon: (id: number) => claimCoupon(id, t, loadCoupons, state),
  }
}

interface CouponViewProps extends ReturnType<typeof useCouponController> {}

function CouponHeader({ colors, handleBack }: Pick<CouponViewProps, 'colors' | 'handleBack'>) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>优惠券</Text>
      <View style={styles.headerRight} />
    </View>
  )
}

function StatusTabs({ selectedTab, setSelectedTab, colors }: Pick<CouponViewProps, 'selectedTab' | 'setSelectedTab' | 'colors'>) {
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabContainer}>
      {USE_STATUS_TABS.map(tab => (
        <TouchableOpacity
          key={String(tab.value)}
          style={[styles.tab, { borderColor: selectedTab === tab.value ? colors.primary : colors.border }, selectedTab === tab.value && { backgroundColor: colors.primary + '10' }]}
          onPress={() => setSelectedTab(tab.value)}
        >
          <Text style={[styles.tabText, { color: selectedTab === tab.value ? colors.primary : colors.textSecondary }]}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  )
}

function CouponItemCard({ item, colors, router }: Pick<CouponViewProps, 'colors' | 'router'> & { item: CouponItem }) {
  return (
    <TouchableOpacity style={[styles.couponItem, { borderColor: colors.border }]} onPress={() => router.push({ pathname: '/favorite-detail', params: { id: String(item.couponTypeId) } } as any)}>
      <View style={styles.couponLeft}>
        <Text style={[styles.couponAmount, { color: colors.primary }]}>¥{item.discountAmount}</Text>
        <Text style={[styles.couponCondition, { color: colors.textSecondary }]}>满¥{item.minimumOrderAmount}可用</Text>
      </View>
      <View style={styles.couponDivider} />
      <View style={styles.couponRight}>
        <Text style={[styles.couponName, { color: colors.text }]} numberOfLines={1}>{item.couponTypeName}</Text>
        <Text style={[styles.couponTime, { color: colors.textSecondary }]}>有效期至 {new Date(item.endTime).toLocaleDateString()}</Text>
        <View style={styles.couponFooter}>
          <Text style={[styles.couponStatus, { color: item.useStatus === 0 ? colors.primary : colors.textSecondary }]}>
            {getCouponStatusLabel(item.useStatus)}
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  )
}

function LoadingView({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={colors.primary} />
    </View>
  );
}

function EmptyCoupons({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="ticket" size={48} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无优惠券</Text>
    </View>
  );
}

function CouponView(props: CouponViewProps) {
  const { colors, router, loading, coupons, handleBack } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <CouponHeader colors={colors} handleBack={handleBack} />
      <StatusTabs {...props} />
      {loading ? <LoadingView colors={colors} /> : coupons.length === 0 ? <EmptyCoupons colors={colors} /> : (
        <FlatList data={coupons} keyExtractor={(item) => item.code} renderItem={({ item }) => <CouponItemCard item={item} colors={colors} router={router} />} contentContainerStyle={styles.listContent} />
      )}
    </SafeAreaView>
  );
}

export default function CouponListScreen() {
  const controller = useCouponController()
  return <CouponView {...controller} />
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
