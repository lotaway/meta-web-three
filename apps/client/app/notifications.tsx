import React, { useState, useEffect, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  FlatList,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useRouter, useFocusEffect } from 'expo-router'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { notificationApi, DEFAULT_USER_ID } from '@/api/generated'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetNotifications = React.Dispatch<React.SetStateAction<NotificationItem[]>>
type SetUnreadCount = React.Dispatch<React.SetStateAction<number>>

interface NotificationItem {
  id: number
  userId: number
  title: string
  content: string
  icon: string
  imageUrl: string
  type: string
  relatedId: string
  readStatus: number
  createTime: string
}

interface TypeTabDef {
  value: string | null
  label: string
  icon: string
}

const TYPE_TABS: TypeTabDef[] = [
  { value: null, label: '全部', icon: 'bell.fill' },
  { value: 'SYSTEM', label: '系统', icon: 'gear' },
  { value: 'ORDER', label: '订单', icon: 'shippingbox' },
  { value: 'PROMOTION', label: '活动', icon: 'tag.fill' },
]

const TYPE_ICONS: Record<string, string> = {
  SYSTEM: 'gear',
  ORDER: 'shippingbox',
  PROMOTION: 'tag.fill',
}

const TYPE_COLORS: Record<string, string> = {
  SYSTEM: '#4A90E2',
  ORDER: '#5fcda2',
  PROMOTION: '#FF6B35',
}

function getTypeIcon(type: string) {
  return TYPE_ICONS[type] || 'bell'
}

function getTypeColor(type: string, colors: AppColors) {
  return TYPE_COLORS[type] || colors.fontColorDisabled
}

function formatNotificationTime(timeStr: string) {
  const date = new Date(timeStr)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes}分钟前`
  if (hours < 24) return `${hours}小时前`
  if (days < 7) return `${days}天前`
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

async function loadNotificationsData(selectedType: string | null, setNotifications: SetNotifications, setLoading: SetBoolean): Promise<void> {
  setLoading(true)
  try {
    const response = await notificationApi.listNotifications({
      xUserId: DEFAULT_USER_ID,
      type: selectedType ?? undefined,
    })
    if (response.data && Array.isArray(response.data)) {
      setNotifications(response.data as any)
    }
  } catch (error) {
    console.error('Failed to load notifications:', error)
    Alert.alert('Error', 'Failed to load notifications.')
  } finally {
    setLoading(false)
  }
}

async function loadUnreadCountData(setUnreadCount: SetUnreadCount): Promise<void> {
  try {
    const response = await notificationApi.getUnreadCount({ xUserId: DEFAULT_USER_ID })
    if (response.data != null) {
      setUnreadCount(response.data as number)
    }
  } catch (error) {
    console.error('Failed to load unread count:', error)
    Alert.alert('Error', 'Failed to load the unread count.')
  }
}

async function markNotificationRead(
  notificationId: number,
  setNotifications: SetNotifications,
  setUnreadCount: SetUnreadCount
): Promise<void> {
  try {
    await notificationApi.markAsRead({
      xUserId: DEFAULT_USER_ID,
      notificationId,
    })
    setNotifications((prev) =>
      prev.map((n) => (n.id === notificationId ? { ...n, readStatus: 1 } : n))
    )
    setUnreadCount((prev) => Math.max(0, prev - 1))
  } catch (error) {
    console.error('Failed to mark as read:', error)
    Alert.alert('Error', 'Failed to mark the notification as read.')
  }
}

async function markAllNotificationsRead(
  setNotifications: SetNotifications,
  setUnreadCount: SetUnreadCount
): Promise<void> {
  try {
    await notificationApi.markAllAsRead({ xUserId: DEFAULT_USER_ID })
    setNotifications((prev) => prev.map((n) => ({ ...n, readStatus: 1 })))
    setUnreadCount(0)
    Alert.alert('提示', '已全部标记为已读')
  } catch (error) {
    Alert.alert('错误', '标记失败')
  }
}

async function deleteNotification(
  notificationId: number,
  notifications: NotificationItem[],
  setNotifications: SetNotifications,
  setUnreadCount: SetUnreadCount
): Promise<void> {
  try {
    await notificationApi.deleteNotification({
      xUserId: DEFAULT_USER_ID,
      notificationId,
    })
    setNotifications((prev) => prev.filter((n) => n.id !== notificationId))
    if (notifications.find((n) => n.id === notificationId)?.readStatus === 0) {
      setUnreadCount((prev) => Math.max(0, prev - 1))
    }
  } catch (error) {
    Alert.alert('错误', '删除失败')
  }
}

function confirmDelete(
  notificationId: number,
  notifications: NotificationItem[],
  setNotifications: SetNotifications,
  setUnreadCount: SetUnreadCount
) {
  Alert.alert('确认删除', '确定要删除这条通知吗？', [
    { text: '取消', style: 'cancel' },
    {
      text: '删除',
      style: 'destructive',
      onPress: () => deleteNotification(notificationId, notifications, setNotifications, setUnreadCount),
    },
  ])
}

function openNotification(
  item: NotificationItem,
  markRead: (id: number) => void,
  router: AppRouter
) {
  if (item.readStatus === 0) {
    markRead(item.id)
  }

  if (item.type === 'ORDER' && item.relatedId) {
    router.push({ pathname: '/orders/[id]', params: { id: item.relatedId } })
  } else if (item.type === 'PROMOTION' && item.relatedId) {
    router.push({ pathname: '/product/[id]', params: { id: item.relatedId } })
  }
}

function useNotificationsState() {
  const [notifications, setNotifications] = useState<NotificationItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedType, setSelectedType] = useState<string | null>(null)
  const [unreadCount, setUnreadCount] = useState(0)
  return { notifications, setNotifications, loading, setLoading, selectedType, setSelectedType, unreadCount, setUnreadCount }
}

type NotificationState = ReturnType<typeof useNotificationsState>

function createMarkRead(state: NotificationState) {
  return (id: number) => markNotificationRead(id, state.setNotifications, state.setUnreadCount)
}

function createNotificationsActions(router: AppRouter, state: NotificationState) {
  const markRead = createMarkRead(state)
  return {
    onBack: () => router.back(),
    onMarkAllRead: () => markAllNotificationsRead(state.setNotifications, state.setUnreadCount),
    onSelectType: state.setSelectedType,
    onPress: (item: NotificationItem) => openNotification(item, markRead, router),
    onLongPress: (id: number) => confirmDelete(id, state.notifications, state.setNotifications, state.setUnreadCount),
  }
}

type NotificationActions = ReturnType<typeof createNotificationsActions>

function useNotificationsController() {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const router = useRouter()
  const state = useNotificationsState()
  const loadNotifications = () => loadNotificationsData(state.selectedType, state.setNotifications, state.setLoading)
  const loadUnreadCount = () => loadUnreadCountData(state.setUnreadCount)
  useFocusEffect(
    useCallback(() => {
      loadNotifications()
      loadUnreadCount()
    }, [state.selectedType])
  )
  return { colors, router, ...state, ...createNotificationsActions(router, state) }
}

interface NotificationsViewProps extends NotificationActions {
  colors: AppColors
  router: AppRouter
  notifications: NotificationItem[]
  loading: boolean
  selectedType: string | null
  unreadCount: number
}

function NotificationsHeader({ colors, unreadCount, onMarkAllRead, onBack }: Pick<NotificationsViewProps, 'colors' | 'unreadCount' | 'onMarkAllRead' | 'onBack'>) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>消息通知</Text>
      {unreadCount > 0 ? (
        <TouchableOpacity style={styles.markAllBtn} onPress={onMarkAllRead}>
          <Text style={[styles.markAllText, { color: colors.primary }]}>全部已读</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.headerRight} />
      )}
    </View>
  )
}

function TypeTab({ tab, selected, colors, onPress }: { tab: TypeTabDef; selected: boolean; colors: AppColors; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.tab, { borderColor: selected ? colors.primary : colors.border }, selected && { backgroundColor: colors.primary + '10' }]}
      onPress={onPress}
    >
      <IconSymbol name={tab.icon as any} size={14} color={selected ? colors.primary : colors.textSecondary} />
      <Text style={[styles.tabText, { color: selected ? colors.primary : colors.textSecondary }]}>
        {tab.label}
      </Text>
    </TouchableOpacity>
  )
}

function TypeTabs({ selectedType, colors, onSelectType }: Pick<NotificationsViewProps, 'selectedType' | 'colors' | 'onSelectType'>) {
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabContainer}>
      {TYPE_TABS.map(tab => (
        <TypeTab
          key={tab.value || 'all'}
          tab={tab}
          selected={selectedType === tab.value}
          colors={colors}
          onPress={() => onSelectType(tab.value)}
        />
      ))}
    </ScrollView>
  )
}

function LoadingState({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={colors.primary} />
    </View>
  )
}

function EmptyNotifications({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="bell.slash" size={48} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无通知</Text>
    </View>
  )
}

function NotificationContent({ item, colors }: { item: NotificationItem; colors: AppColors }) {
  return (
    <View style={styles.notificationContent}>
      <View style={styles.notificationHeader}>
        <Text style={[styles.notificationTitle, { color: colors.text }]} numberOfLines={1}>
          {item.title}
        </Text>
        {item.readStatus === 0 && <View style={styles.unreadDot} />}
      </View>
      <Text style={[styles.notificationText, { color: colors.textSecondary }]} numberOfLines={2}>
        {item.content}
      </Text>
      <Text style={[styles.notificationTime, { color: colors.textSecondary }]}>
        {formatNotificationTime(item.createTime)}
      </Text>
    </View>
  )
}

function NotificationItemView({ item, colors, onPress, onLongPress }: { item: NotificationItem; colors: AppColors; onPress: () => void; onLongPress: () => void }) {
  const typeColor = getTypeColor(item.type, colors)
  return (
    <TouchableOpacity
      style={[styles.notificationItem, { backgroundColor: item.readStatus === 0 ? colors.card : colors.background, borderColor: colors.border }]}
      onPress={onPress}
      onLongPress={onLongPress}
    >
      <View style={[styles.notificationIcon, { backgroundColor: typeColor + '15' }]}>
        <IconSymbol name={item.icon as any || getTypeIcon(item.type) as any} size={24} color={typeColor} />
      </View>
      <NotificationContent item={item} colors={colors} />
      {!!item.imageUrl && <Image source={{ uri: item.imageUrl }} style={styles.notificationImage} />}
    </TouchableOpacity>
  )
}

function NotificationList({ notifications, colors, onPress, onLongPress }: {
  notifications: NotificationItem[]
  colors: AppColors
  onPress: (item: NotificationItem) => void
  onLongPress: (id: number) => void
}) {
  return (
    <FlatList
      data={notifications}
      keyExtractor={(item) => String(item.id)}
      renderItem={({ item }) => <NotificationItemView item={item} colors={colors} onPress={() => onPress(item)} onLongPress={() => onLongPress(item.id)} />}
      contentContainerStyle={styles.listContent}
    />
  )
}

function NotificationsView(props: NotificationsViewProps) {
  const { colors, notifications, loading } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <NotificationsHeader {...props} />
      <TypeTabs {...props} />
      {loading ? (
        <LoadingState colors={colors} />
      ) : notifications.length === 0 ? (
        <EmptyNotifications colors={colors} />
      ) : (
        <NotificationList {...props} />
      )}
    </SafeAreaView>
  )
}

export default function NotificationsScreen() {
  const controller = useNotificationsController()
  return <NotificationsView {...controller} />
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
  markAllBtn: { padding: 8 },
  markAllText: { fontSize: 14 },
  tabContainer: {
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  tab: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    marginRight: 10,
    gap: 6,
  },
  tabText: {
    fontSize: 13,
  },
  listContent: {
    padding: 16,
    paddingBottom: 32,
  },
  notificationItem: {
    flexDirection: 'row',
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
  },
  notificationIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    flexShrink: 0,
  },
  notificationContent: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'space-between',
  },
  notificationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  notificationTitle: {
    fontSize: 15,
    fontWeight: '600',
    flex: 1,
  },
  unreadDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FF3B30',
    marginLeft: 8,
  },
  notificationText: {
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 4,
  },
  notificationTime: {
    fontSize: 11,
  },
  notificationImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginLeft: 8,
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
