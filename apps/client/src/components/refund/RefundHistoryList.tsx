import React from 'react'
import { View, Text, Modal, FlatList, TouchableOpacity, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

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

export function RefundHistoryList({ visible, onClose, refundHistory }: RefundHistoryListProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.modalOverlay}>
        <TouchableOpacity style={styles.modalBg} onPress={onClose} />
        <View style={[styles.modalContent, { backgroundColor: colors.background }]}>
          <View style={styles.modalHeader}>
            <Text style={[styles.modalTitle, { color: colors.text }]}>{t('refund.history_title')}</Text>
            <TouchableOpacity onPress={onClose}>
              <IconSymbol name="xmark" size={24} color={colors.textSecondary} />
            </TouchableOpacity>
          </View>
          {refundHistory.length === 0 ? (
            <View style={styles.emptyHistory}>
              <Text style={[styles.emptyHistoryText, { color: colors.textSecondary }]}>
                {t('refund.no_history')}
              </Text>
            </View>
          ) : (
            <FlatList
              data={refundHistory}
              keyExtractor={(item) => String(item.id)}
              renderItem={({ item }) => {
                const statusInfo = REFUND_STATUS_MAP[item.status] || { text: '-', color: colors.textSecondary }
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
              }}
            />
          )}
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
