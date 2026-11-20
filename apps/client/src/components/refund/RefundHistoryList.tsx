import React from 'react'
import { View, Text, Modal, FlatList, TouchableOpacity, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light

const REFUND_STATUS_MAP: Record<number, { text: string; color: string }> = {
  0: { text: '待处理', color: '#FF6B35' },
  1: { text: '商家已同意', color: '#5fcda2' },
  2: { text: '商家已拒绝', color: '#FF3B30' },
  3: { text: '退款成功', color: '#4A90E2' },
  4: { text: '已取消', color: '#8E8E93' },
}

interface RefundHistoryListProps {
  visible: boolean
  onClose: () => void
  refundHistory: any[]
}

function getRefundStatus(status: number, colors: AppColors) {
  return REFUND_STATUS_MAP[status] || { text: '-', color: colors.textSecondary }
}

function HistoryItem({ item, colors }: { item: any; colors: AppColors }) {
  const statusInfo = getRefundStatus(item.status, colors)
  return (
    <View style={[styles.historyItem, { borderBottomColor: colors.border }]}>
      <View style={styles.historyItemHeader}>
        <Text style={[styles.historyItemTime, { color: colors.textSecondary }]}>
          {new Date(item.createTime).toLocaleString()}
        </Text>
        <Text style={[styles.historyItemStatus, { color: statusInfo.color }]}>
          {statusInfo.text}
        </Text>
      </View>
      <Text style={[styles.historyItemAmount, { color: colors.primary }]}>
        ¥{item.refundAmount}
      </Text>
    </View>
  )
}

function EmptyHistory({ colors, t }: { colors: AppColors; t: Translate }) {
  return (
    <View style={styles.emptyHistory}>
      <Text style={[styles.emptyHistoryText, { color: colors.textSecondary }]}>
        {t('refund.no_history')}
      </Text>
    </View>
  )
}

function HistoryContent({ colors, t, refundHistory }: { colors: AppColors; t: Translate; refundHistory: any[] }) {
  if (refundHistory.length === 0) return <EmptyHistory colors={colors} t={t} />
  return (
    <FlatList
      data={refundHistory}
      keyExtractor={(item) => String(item.id)}
      renderItem={({ item }) => <HistoryItem item={item} colors={colors} />}
    />
  )
}

function ModalHeader({ colors, t, onClose }: { colors: AppColors; t: Translate; onClose: () => void }) {
  return (
    <View style={styles.modalHeader}>
      <Text style={[styles.modalTitle, { color: colors.text }]}>{t('refund.history_title')}</Text>
      <TouchableOpacity onPress={onClose}>
        <IconSymbol name="xmark" size={24} color={colors.textSecondary} />
      </TouchableOpacity>
    </View>
  )
}

export function RefundHistoryList({ visible, onClose, refundHistory }: RefundHistoryListProps) {
  const { t } = useTranslation()
  const colors = Colors[useColorScheme() ?? 'light']
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.modalOverlay}>
        <TouchableOpacity style={styles.modalBg} onPress={onClose} />
        <View style={[styles.modalContent, { backgroundColor: colors.background }]}>
          <ModalHeader colors={colors} t={t} onClose={onClose} />
          <HistoryContent colors={colors} t={t} refundHistory={refundHistory} />
        </View>
      </View>
    </Modal>
  )
}

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  modalBg: {
    ...StyleSheet.absoluteFillObject,
  },
  modalContent: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '60%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  emptyHistory: {
    padding: 40,
    alignItems: 'center',
  },
  emptyHistoryText: {
    fontSize: 14,
  },
  historyItem: {
    padding: 16,
    borderBottomWidth: 1,
  },
  historyItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  historyItemTime: {
    fontSize: 12,
  },
  historyItemStatus: {
    fontSize: 14,
    fontWeight: '500',
  },
  historyItemAmount: {
    fontSize: 16,
    fontWeight: 'bold',
  },
})
