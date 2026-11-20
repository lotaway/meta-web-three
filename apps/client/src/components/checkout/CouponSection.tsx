import React from 'react'
import { View, Text, TouchableOpacity, Modal, FlatList } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'

interface AvailableCoupon {
  id: number
  name: string
  amount: number
  minPoint: number
  endTime: string
  useType: number
}

type CouponColors = typeof Colors.light

interface CouponSectionProps {
  selectedCoupon: AvailableCoupon | null
  colors: CouponColors
  couponModalVisible: boolean
  availableCoupons: AvailableCoupon[]
  loadingCoupons: boolean
  onOpenCouponSelector: () => void
  onSelectCoupon: (coupon: AvailableCoupon) => void
  onClearCoupon: () => void
  onCloseModal: () => void
}

interface CouponAmountProps {
  coupon: AvailableCoupon
  colors: CouponColors
}

interface CouponDetailsProps {
  coupon: AvailableCoupon
  colors: CouponColors
}

interface CouponBadgeProps {
  coupon: AvailableCoupon
  colors: CouponColors
  onClearCoupon: () => void
}

interface CouponItemProps extends CouponAmountProps {
  onSelectCoupon: (coupon: AvailableCoupon) => void
}

interface CouponModalBodyProps {
  colors: CouponColors
  loadingCoupons: boolean
  availableCoupons: AvailableCoupon[]
  onSelectCoupon: (coupon: AvailableCoupon) => void
}

interface CouponSelectorModalProps extends CouponModalBodyProps {
  couponModalVisible: boolean
  onCloseModal: () => void
}

function SelectedCouponBadge({
  coupon,
  colors,
  onClearCoupon,
}: CouponBadgeProps) {
  return (
    <View style={styles.couponSelected}>
      <Text style={[styles.couponSelectedText, { color: colors.primary }]}>
        -¥{coupon.amount}
      </Text>
      <TouchableOpacity onPress={onClearCoupon} style={styles.clearCouponBtn}>
        <IconSymbol name="xmark.circle.fill" size={18} color={colors.fontColorDisabled} />
      </TouchableOpacity>
    </View>
  )
}

function CouponPlaceholder({ colors }: { colors: CouponColors }) {
  return (
    <View style={styles.couponPlaceholder}>
      <Text style={[styles.couponPlaceholderText, { color: colors.fontColorDisabled }]}>
        选择优惠券
      </Text>
      <IconSymbol name="chevron.right" size={16} color={colors.fontColorDisabled} />
    </View>
  )
}

type CouponRowProps = Pick<CouponSectionProps, 'selectedCoupon' | 'colors' | 'onClearCoupon'>

function CouponRowContent({ selectedCoupon, colors, onClearCoupon }: CouponRowProps) {
  return (
    <View style={styles.couponRow}>
      <IconSymbol name="ticket" size={20} color={colors.primary} />
      <Text style={[styles.sectionTitle, { color: colors.fontColorDark, flex: 1 }]}>
        优惠券
      </Text>
      {selectedCoupon ? (
        <SelectedCouponBadge
          coupon={selectedCoupon}
          colors={colors}
          onClearCoupon={onClearCoupon}
        />
      ) : (
        <CouponPlaceholder colors={colors} />
      )}
    </View>
  )
}

function CouponSummaryRow({
  colors,
  onOpenCouponSelector,
  ...rowProps
}: Pick<CouponSectionProps, 'selectedCoupon' | 'colors' | 'onOpenCouponSelector' | 'onClearCoupon'>) {
  return (
    <TouchableOpacity
      style={[styles.section, { backgroundColor: colors.background }]}
      onPress={onOpenCouponSelector}
    >
      <CouponRowContent {...rowProps} colors={colors} />
    </TouchableOpacity>
  )
}

function CouponAmount({ coupon, colors }: CouponAmountProps) {
  return (
    <View style={styles.couponItemLeft}>
      <Text style={[styles.couponItemAmount, { color: colors.primary }]}>
        ¥{coupon.amount}
      </Text>
      <Text style={[styles.couponItemCondition, { color: colors.fontColorDisabled }]}>
        满¥{coupon.minPoint}可用
      </Text>
    </View>
  )
}

function CouponDetails({ coupon, colors }: CouponDetailsProps) {
  return (
    <View style={styles.couponItemRight}>
      <Text style={[styles.couponItemName, { color: colors.fontColorDark }]} numberOfLines={1}>
        {coupon.name}
      </Text>
      <Text style={[styles.couponItemTime, { color: colors.fontColorLight }]}>
        有效期至 {coupon.endTime}
      </Text>
      <Text style={[styles.couponItemUseType, { color: colors.fontColorLight }]}>
        {coupon.useType === 0 ? '全场通用' : coupon.useType === 1 ? '指定分类可用' : '指定商品可用'}
      </Text>
    </View>
  )
}

function CouponItem({ coupon, colors, onSelectCoupon }: CouponItemProps) {
  return (
    <TouchableOpacity
      style={[styles.couponItem, { borderColor: colors.border }]}
      onPress={() => onSelectCoupon(coupon)}
    >
      <CouponAmount coupon={coupon} colors={colors} />
      <View style={[styles.couponItemDivider, { backgroundColor: colors.border }]} />
      <CouponDetails coupon={coupon} colors={colors} />
    </TouchableOpacity>
  )
}

function CouponLoadingMessage({ colors }: { colors: CouponColors }) {
  return (
    <View style={styles.modalLoading}>
      <Text style={{ color: colors.fontColorBase }}>加载中...</Text>
    </View>
  )
}

function CouponEmptyMessage({ colors }: { colors: CouponColors }) {
  return (
    <View style={styles.modalLoading}>
      <Text style={{ color: colors.fontColorDisabled }}>暂无可用优惠券</Text>
    </View>
  )
}

function CouponModalBody({ colors, loadingCoupons, availableCoupons, onSelectCoupon }: CouponModalBodyProps) {
  if (loadingCoupons) {
    return <CouponLoadingMessage colors={colors} />
  }
  if (availableCoupons.length === 0) {
    return <CouponEmptyMessage colors={colors} />
  }
  return (
    <FlatList
      data={availableCoupons}
      keyExtractor={(item) => String(item.id)}
      renderItem={({ item }) => (
        <CouponItem coupon={item} colors={colors} onSelectCoupon={onSelectCoupon} />
      )}
      contentContainerStyle={styles.couponListContent}
    />
  )
}

function CouponModalHeader({
  colors,
  onCloseModal,
}: {
  colors: CouponColors
  onCloseModal: () => void
}) {
  return (
    <View style={[styles.modalHeader, { borderBottomColor: colors.border }]}>
      <Text style={[styles.modalTitle, { color: colors.fontColorDark }]}>
        选择优惠券
      </Text>
      <TouchableOpacity onPress={onCloseModal}>
        <IconSymbol name="xmark" size={24} color={colors.fontColorBase} />
      </TouchableOpacity>
    </View>
  )
}

function CouponModalContent({
  colors,
  loadingCoupons,
  availableCoupons,
  onSelectCoupon,
  onCloseModal,
}: Omit<CouponSelectorModalProps, 'couponModalVisible'>) {
  return (
    <View style={[styles.modalContent, { backgroundColor: colors.background }]}>
      <CouponModalHeader colors={colors} onCloseModal={onCloseModal} />
      <CouponModalBody
        colors={colors}
        loadingCoupons={loadingCoupons}
        availableCoupons={availableCoupons}
        onSelectCoupon={onSelectCoupon}
      />
    </View>
  )
}

function CouponSelectorModal({
  colors,
  couponModalVisible,
  onCloseModal,
  ...contentProps
}: CouponSelectorModalProps) {
  return (
    <Modal
      visible={couponModalVisible}
      transparent
      animationType="slide"
      onRequestClose={onCloseModal}
    >
      <View style={styles.modalOverlay}>
        <CouponModalContent {...contentProps} colors={colors} onCloseModal={onCloseModal} />
      </View>
    </Modal>
  )
}

export default function CouponSection(props: CouponSectionProps) {
  return (
    <>
      <CouponSummaryRow {...props} />
      <CouponSelectorModal {...props} />
    </>
  )
}

const styles = {
  section: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    marginBottom: 12,
  },
  couponRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
  },
  couponSelected: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
  },
  couponSelectedText: {
    fontSize: 16,
    fontWeight: '600' as const,
    marginRight: 8,
  },
  clearCouponBtn: {
    padding: 2,
  },
  couponPlaceholder: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
  },
  couponPlaceholderText: {
    fontSize: 14,
    marginRight: 4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end' as const,
  },
  modalContent: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: 560,
  },
  modalHeader: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'space-between' as const,
    padding: 16,
    borderBottomWidth: 1,
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
  },
  modalLoading: {
    padding: 32,
    alignItems: 'center' as const,
  },
  couponListContent: {
    paddingBottom: 16,
  },
  couponItem: {
    flexDirection: 'row' as const,
    marginHorizontal: 16,
    marginTop: 12,
    borderRadius: 12,
    borderWidth: 1,
    overflow: 'hidden' as const,
    backgroundColor: '#fff',
  },
  couponItemLeft: {
    width: 90,
    padding: 12,
    justifyContent: 'center' as const,
    alignItems: 'center' as const,
  },
  couponItemAmount: {
    fontSize: 22,
    fontWeight: 'bold' as const,
  },
  couponItemCondition: {
    fontSize: 11,
    marginTop: 4,
  },
  couponItemDivider: {
    width: 1,
  },
  couponItemRight: {
    flex: 1,
    padding: 12,
    justifyContent: 'space-between' as const,
  },
  couponItemName: {
    fontSize: 14,
    fontWeight: '500' as const,
  },
  couponItemTime: {
    fontSize: 11,
    marginTop: 4,
  },
  couponItemUseType: {
    fontSize: 11,
    marginTop: 2,
  },
}
