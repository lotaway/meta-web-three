import React, { useCallback } from 'react'
import { View, Text, FlatList, TouchableOpacity, StyleSheet } from 'react-native'
import { Stack, useRouter } from 'expo-router'
import { useTranslation } from 'react-i18next'
import { useConversation } from '@/hooks/useConversation'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

export default function MessagesScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme()
  const colors = Colors[colorScheme ?? 'light']
  const { conversations, loading } = useConversation()

  const statusLabel = (status: string) => {
    if (status === 'ACTIVE') return t('cs.active') || '进行中'
    if (status === 'QUEUING') return t('cs.queuing') || '排队中'
    return t('cs.closed') || '已结束'
  }

  const renderItem = useCallback(({ item }: any) => (
    <TouchableOpacity
      style={[styles.item, { borderBottomColor: colors.border || '#eee' }]}
      onPress={() => router.push(`/cs/${item.sessionId}`)}
    >
      <View style={styles.itemHeader}>
        <Text style={[styles.channel, { color: colors.tint }]}>
          {item.channel === 'PRODUCT' ? t('cs.from_product') || '商品咨询' :
           item.channel === 'ORDER' ? t('cs.from_order') || '订单咨询' : t('cs.from_shop') || '店铺咨询'}
        </Text>
        <Text style={[styles.status, {
          color: item.status === 'ACTIVE' ? '#52c41a' : '#faad14',
        }]}>{statusLabel(item.status)}</Text>
      </View>
      <Text style={[styles.time, { color: colors.icon || '#999' }]}>
        {new Date(item.createTime).toLocaleString()}
      </Text>
    </TouchableOpacity>
  ), [colors, router, t])

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <Stack.Screen options={{ title: t('cs.my_messages') || '我的消息' }} />
      <FlatList
        data={conversations}
        renderItem={renderItem}
        keyExtractor={(item) => item.sessionId}
        refreshing={loading}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={{ color: colors.icon || '#999' }}>{t('cs.no_messages') || '暂无消息'}</Text>
          </View>
        }
      />
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  item: { padding: 16, borderBottomWidth: StyleSheet.hairlineWidth },
  itemHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  channel: { fontSize: 15, fontWeight: '500' },
  status: { fontSize: 12 },
  time: { fontSize: 12, marginTop: 6 },
  empty: { alignItems: 'center', paddingTop: 60 },
})
