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

const TYPE_TABS = [
  { value: null, label: '全部', icon: 'bell.fill' },
  { value: 'SYSTEM', label: '系统', icon: 'gear' },
  { value: 'ORDER', label: '订单', icon: 'shippingbox' },
  { value: 'PROMOTION', label: '活动', icon: 'tag.fill' },
]

export default function NotificationsScreen() {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const router = useRouter()

  const [notifications, setNotifications] = useState<NotificationItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedType, setSelectedType] = useState<string | null>(null)
  const [unreadCount, setUnreadCount] = useState(0)

  useFocusEffect(
    useCallback(() => {
      loadNotifications()
      loadUnreadCount()
    }, [selectedType])
  )

  const loadNotifications = async () => {
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
    } finally {
      setLoading(false)
    }
  }

  const loadUnreadCount = async () => {
    try {
      const response = await notificationApi.getUnreadCount({ xUserId: DEFAULT_USER_ID })
      if (response.data != null) {
        setUnreadCount(response.data as number)
      }
    } catch (error) {
      console.error('Failed to load unread count:', error)
    }
  }

  const handleMarkRead = async (notificationId: number) => {
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
    }
  }

  const handleMarkAllRead = async () => {
    try {
      await notificationApi.markAllAsRead({ xUserId: DEFAULT_USER_ID })
      setNotifications((prev) => prev.map((n) => ({ ...n, readStatus: 1 })))
      setUnreadCount(0)
      Alert.alert('提示', '已全部标记为已读')
    } catch (error) {
      Alert.alert('错误', '标记失败')
    }
  }

  const handleDelete = (notificationId: number) => {
    Alert.alert('确认删除', '确定要删除这条通知吗？', [
      { text: '取消', style: 'cancel' },
      {
        text: '删除',
        style: 'destructive',
        onPress: async () => {
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
        },
      },
    ])
  }

  const handleNotificationPress = (item: NotificationItem) => {
    if (item.readStatus === 0) {
      handleMarkRead(item.id)
    }

    if (item.type === 'ORDER' && item.relatedId) {
      router.push({ pathname: '/orders/[id]', params: { id: item.relatedId } })
    } else if (item.type === 'PROMOTION' && item.relatedId) {
      router.push({ pathname: '/product/[id]', params: { id: item.relatedId } })
    }
  }

  const formatTime = (timeStr: string) => {
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

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'SYSTEM': return 'gear'
      case 'ORDER': return 'shippingbox'
      case 'PROMOTION': return 'tag.fill'
      default: return 'bell'
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'SYSTEM': return '#4A90E2'
      case 'ORDER': return '#5fcda2'
      case 'PROMOTION': return '#FF6B35'
      default: return colors.fontColorDisabled
    }
  }

  const renderNotification = ({ item }: { item: NotificationItem }) => (
    <TouchableOpacity
      style={[styles.notificationItem, {
        backgroundColor: item.readStatus === 0 ? colors.card : colors.background,
        borderColor: colors.border,
      }]}
      onPress={() => handleNotificationPress(item)}
      onLongPress={() => handleDelete(item.id)}
    >
      <View style={[styles.notificationIcon, { backgroundColor: getTypeColor(item.type) + '15' }]}>
        <IconSymbol name={item.icon as any || getTypeIcon(item.type) as any} size={24} color={getTypeColor(item.type)} />
      </View>
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
          {formatTime(item.createTime)}
        </Text>
      </View>
      {item.imageUrl && (
        <Image source={{ uri: item.imageUrl }} style={styles.notificationImage} />
      )}
    </TouchableOpacity>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>消息通知</Text>
        {unreadCount > 0 ? (
          <TouchableOpacity style={styles.markAllBtn} onPress={handleMarkAllRead}>
            <Text style={[styles.markAllText, { color: colors.primary }]}>全部已读</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.headerRight} />
        )}
      </View>

      {/* Tab 切换 */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabContainer}>
        {TYPE_TABS.map((tab) => (
          <TouchableOpacity
            key={tab.value || 'all'}
            style={[
              styles.tab,
              { borderColor: selectedType === tab.value ? colors.primary : colors.border },
              selectedType === tab.value && { backgroundColor: colors.primary + '10' },
            ]}
            onPress={() => setSelectedType(tab.value)}
          >
            <IconSymbol
              name={tab.icon as any}
              size={14}
              color={selectedType === tab.value ? colors.primary : colors.textSecondary}
            />
            <Text
              style={[
                styles.tabText,
                { color: selectedType === tab.value ? colors.primary : colors.textSecondary },
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* 通知列表 */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : notifications.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="bell.slash" size={48} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>暂无通知</Text>
        </View>
      ) : (
        <FlatList
          data={notifications}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderNotification}
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
