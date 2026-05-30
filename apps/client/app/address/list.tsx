import React, { useState, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { useRouter } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { useFocusEffect } from '@react-navigation/native'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { addressApi } from '@/api/generated'
import { useAuth } from '@/contexts/AuthContext'
import type { MemberAddress } from '@/src/generated/api/models'

export default function AddressListScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { userId } = useAuth()

  const [addresses, setAddresses] = useState<MemberAddress[]>([])
  const [loading, setLoading] = useState(true)

  const loadAddresses = useCallback(async () => {
    if (!userId) return
    setLoading(true)
    try {
      const response = await addressApi.list({ xUserId: userId })
      if (response.data) {
        setAddresses(response.data)
      }
    } catch (error) {
      console.error('Failed to load addresses:', error)
    } finally {
      setLoading(false)
    }
  }, [userId])

  useFocusEffect(
    useCallback(() => {
      loadAddresses()
    }, [loadAddresses])
  )

  const handleAdd = () => {
    router.push('/address/edit')
  }

  const handleEdit = (address: MemberAddress) => {
    router.push({ pathname: '/address/edit', params: { id: address.id } })
  }

  const handleDelete = (id: number) => {
    Alert.alert(
      t('address.delete_confirm_title'),
      t('address.delete_confirm_message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('address.delete'),
          style: 'destructive',
          onPress: async () => {
            try {
              await addressApi.remove({ id })
              loadAddresses()
            } catch (error) {
              Alert.alert(t('common.error'), t('address.delete_failed'))
            }
          },
        },
      ]
    )
  }

  const handleSetDefault = async (address: MemberAddress) => {
    try {
      await addressApi.update({
        id: address.id!,
        memberAddress: { ...address, defaultStatus: true },
      })
      loadAddresses()
    } catch (error) {
      Alert.alert(t('common.error'), t('address.update_failed'))
    }
  }

  const renderAddressItem = ({ item }: { item: MemberAddress }) => {
    const isDefault = item.defaultStatus === true
    return (
      <TouchableOpacity
        style={[styles.addressCard, { backgroundColor: colors.card, borderColor: colors.border }]}
        onPress={() => handleEdit(item)}
      >
        <View style={styles.addressHeader}>
          <View style={styles.userInfo}>
            <Text style={[styles.userName, { color: colors.text }]}>{item.name}</Text>
            <Text style={[styles.userPhone, { color: colors.textSecondary }]}>{item.phoneNumber}</Text>
          </View>
          {isDefault && (
            <View style={[styles.defaultTag, { backgroundColor: colors.primary }]}>
              <Text style={styles.defaultTagText}>{t('address.default')}</Text>
            </View>
          )}
        </View>
        <Text style={[styles.addressText, { color: colors.text }]}>
          {item.province}{item.city}{item.region}{item.detailAddress}
        </Text>
        <View style={styles.addressFooter}>
          {!isDefault && (
            <TouchableOpacity
              style={styles.defaultBtn}
              onPress={() => handleSetDefault(item)}
            >
              <Text style={[styles.defaultBtnText, { color: colors.primary }]}>
                {t('address.set_default')}
              </Text>
            </TouchableOpacity>
          )}
          <View style={styles.actions}>
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleEdit(item)}>
              <IconSymbol name="pencil" size={18} color={colors.primary} />
              <Text style={[styles.actionText, { color: colors.primary }]}>{t('address.edit')}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item.id!)}>
              <IconSymbol name="trash" size={18} color="#FF3B30" />
              <Text style={[styles.actionText, { color: '#FF3B30' }]}>{t('address.delete')}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </TouchableOpacity>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('address.title')}</Text>
        <View style={styles.headerRight} />
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : addresses.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="mappin.and.ellipse" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('address.no_addresses')}</Text>
          <TouchableOpacity style={[styles.addBtn, { backgroundColor: colors.primary }]} onPress={handleAdd}>
            <Text style={styles.addBtnText}>{t('address.add')}</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <FlatList
            data={addresses}
            keyExtractor={(item) => String(item.id)}
            renderItem={renderAddressItem}
            contentContainerStyle={styles.listContent}
          />
          <TouchableOpacity
            style={[styles.bottomAddBtn, { backgroundColor: colors.primary }]}
            onPress={handleAdd}
          >
            <IconSymbol name="plus" size={20} color="#fff" />
            <Text style={styles.bottomAddBtnText}>{t('address.add_new')}</Text>
          </TouchableOpacity>
        </>
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
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
    marginBottom: 24,
  },
  addBtn: {
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 24,
  },
  addBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  listContent: {
    padding: 12,
    paddingBottom: 100,
  },
  addressCard: {
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
  },
  addressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  userInfo: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  userName: { fontSize: 16, fontWeight: '600' },
  userPhone: { fontSize: 14 },
  defaultTag: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  defaultTagText: { color: '#fff', fontSize: 12 },
  addressText: { fontSize: 14, lineHeight: 20, marginBottom: 12 },
  addressFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    paddingTop: 12,
  },
  defaultBtn: { padding: 4 },
  defaultBtnText: { fontSize: 13 },
  actions: { flexDirection: 'row', gap: 16 },
  actionBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  actionText: { fontSize: 13 },
  bottomAddBtn: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 24,
    gap: 8,
  },
  bottomAddBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
})
