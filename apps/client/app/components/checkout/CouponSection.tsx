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

interface CouponSectionProps {
  selectedCoupon: AvailableCoupon | null
  colors: typeof Colors.light
  couponModalVisible: boolean
  availableCoupons: AvailableCoupon[]
  loadingCoupons: boolean
  onOpenCouponSelector: () => void
  onSelectCoupon: (coupon: AvailableCoupon) => void
  onClearCoupon: () => void
  onCloseModal: () => void
}

export default function CouponSection({
  selectedCoupon,
  colors,
  couponModalVisible,
  availableCoupons,
  loadingCoupons,
  onOpenCouponSelector,
  onSelectCoupon,
  onClearCoupon,
  onCloseModal,
}: CouponSectionProps) {
  return (
    <>
      <TouchableOpacity
        style={[styles.section, { backgroundColor: colors.background }]}
        onPress={onOpenCouponSelector}
      >
        <View style={styles.couponRow}>
          <IconSymbol name="ticket" size={20} color={colors.primary} />
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark, flex: 1 }]}>
            优惠券
          </Text>
          {selectedCoupon ? (
            <View style={styles.couponSelected}>
              <Text style={[styles.couponSelectedText, { color: colors.primary }]}>
                -¥{selectedCoupon.amount}
              </Text>
              <TouchableOpacity onPress={onClearCoupon} style={styles.clearCouponBtn}>
                <IconSymbol name="xmark.circle.fill" size={18} color={colors.fontColorDisabled} />
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.couponPlaceholder}>
              <Text style={[styles.couponPlaceholderText, { color: colors.fontColorDisabled }]}>
                选择优惠券
              </Text>
              <IconSymbol name="chevron.right" size={16} color={colors.fontColorDisabled} />
            </View>
          )}
        </View>
      </TouchableOpacity>

      <Modal
        visible={couponModalVisible}
        transparent
        animationType="slide"
        onRequestClose={onCloseModal}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.background }]}>
            <View style={[styles.modalHeader, { borderBottomColor: colors.border }]}>
              <Text style={[styles.modalTitle, { color: colors.fontColorDark }]}>
                选择优惠券
              </Text>
              <TouchableOpacity onPress={onCloseModal}>
                <IconSymbol name="xmark" size={24} color={colors.fontColorBase} />
              </TouchableOpacity>
            </View>
            {loadingCoupons ? (
              <View style={styles.modalLoading}>
                <Text style={{ color: colors.fontColorBase }}>加载中...</Text>
              </View>
            ) : availableCoupons.length === 0 ? (
              <View style={styles.modalLoading}>
                <Text style={{ color: colors.fontColorDisabled }}>暂无可用优惠券</Text>
              </View>
            ) : (
              <FlatList
                data={availableCoupons}
                keyExtractor={(item) => String(item.id)}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[styles.couponItem, { borderColor: colors.border }]}
                    onPress={() => onSelectCoupon(item)}
                  >
                    <View style={styles.couponItemLeft}>
                      <Text style={[styles.couponItemAmount, { color: colors.primary }]}>
                        ¥{item.amount}
                      </Text>
                      <Text style={[styles.couponItemCondition, { color: colors.fontColorDisabled }]}>
                        满¥{item.minPoint}可用
                      </Text>
                    </View>
                    <View style={[styles.couponItemDivider, { backgroundColor: colors.border }]} />
                    <View style={styles.couponItemRight}>
                      <Text style={[styles.couponItemName, { color: colors.fontColorDark }]} numberOfLines={1}>
                        {item.name}
                      </Text>
                      <Text style={[styles.couponItemTime, { color: colors.fontColorLight }]}>
                        有效期至 {item.endTime}
                      </Text>
                      <Text style={[styles.couponItemUseType, { color: colors.fontColorLight }]}>
                        {item.useType === 0 ? '全场通用' : item.useType === 1 ? '指定分类可用' : '指定商品可用'}
                      </Text>
                    </View>
                  </TouchableOpacity>
                )}
                contentContainerStyle={styles.couponListContent}
              />
            )}
          </View>
        </View>
      </Modal>
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
    maxHeight: '70%',
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
