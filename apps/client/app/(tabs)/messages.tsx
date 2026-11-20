import React from 'react'
import { View, Text, FlatList, TouchableOpacity, StyleSheet } from 'react-native'
import { Stack, useRouter } from 'expo-router'
import { useTranslation } from 'react-i18next'
import { useConversation } from '@/hooks/useConversation'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type Translate = ReturnType<typeof useTranslation>['t']
type Conversation = ReturnType<typeof useConversation>['conversations'][number]

function getStatusLabel(status: string, t: Translate) {
  if (status === 'ACTIVE') return t('cs.active') || '进行中'
  if (status === 'QUEUING') return t('cs.queuing') || '排队中'
  return t('cs.closed') || '已结束'
}

function getChannelLabel(item: Conversation, t: Translate) {
  if (item.channel === 'PRODUCT') return t('cs.from_product') || '商品咨询'
  if (item.channel === 'ORDER') return t('cs.from_order') || '订单咨询'
  return t('cs.from_shop') || '店铺咨询'
}

interface ConversationItemProps {
  item: Conversation
  colors: AppColors
  router: AppRouter
  t: Translate
}

function ConversationItem({ item, colors, router, t }: ConversationItemProps) {
  return (
    <TouchableOpacity
      style={[styles.item, { borderBottomColor: colors.border || '#eee' }]}
      onPress={() => router.push(`/cs/${item.sessionId}`)}
    >
      <View style={styles.itemHeader}>
        <Text style={[styles.channel, { color: colors.tint }]}>{getChannelLabel(item, t)}</Text>
        <Text style={[styles.status, { color: item.status === 'ACTIVE' ? '#52c41a' : '#faad14' }]}>
          {getStatusLabel(item.status, t)}
        </Text>
      </View>
      <Text style={[styles.time, { color: colors.icon || '#999' }]}>
        {new Date(item.createTime).toLocaleString()}
      </Text>
    </TouchableOpacity>
  )
}

function EmptyMessages({ colors, t }: { colors: AppColors; t: Translate }) {
  return (
    <View style={styles.empty}>
      <Text style={{ color: colors.icon || '#999' }}>{t('cs.no_messages') || '暂无消息'}</Text>
    </View>
  )
}

interface MessagesViewProps {
  t: Translate
  router: AppRouter
  colors: AppColors
  conversations: Conversation[]
  loading: boolean
}

function MessagesView(props: MessagesViewProps) {
  const { t, colors, router, conversations, loading } = props;
  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <Stack.Screen options={{ title: t('cs.my_messages') || '我的消息' }} />
      <FlatList
        data={conversations}
        renderItem={({ item }) => <ConversationItem item={item} colors={colors} router={router} t={t} />}
        keyExtractor={(item) => item.sessionId}
        refreshing={loading}
        ListEmptyComponent={<EmptyMessages colors={colors} t={t} />}
      />
    </View>
  );
}

export default function MessagesScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme()
  const colors = Colors[colorScheme ?? 'light']
  const { conversations, loading } = useConversation()
  return <MessagesView t={t} router={router} colors={colors} conversations={conversations} loading={loading} />
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
